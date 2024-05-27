import math
from typing import Any, Dict, List, Optional

import lightning.pytorch as pl
import pyro
import pyro.distributions as dist
import torch
import torch.nn.functional as F
from pyro.infer.autoguide import AutoNormal
from scipy.sparse import csr_matrix

global K  # Number of states
global B  # Number of batches
global C  # Number of types
global R  # Number of groups
global G  # Number of genes

K: int = -1
B: int = -1
C: int = -1
R: int = -1
G: int = -1


DEBUG = False
SUBSMPL = 512
NUM_PARTICLES = 12
MIN_NUM_GLOBAL_UPDATES = 2048
MIN_NUM_CELL_UPDATES = 48


# Use only for debugging.
pyro.enable_validation(DEBUG)


def subset(tensor: Optional[torch.Tensor], idx: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
    if idx is None:
        return tensor
    if tensor is None:
        return None
    return tensor.index_select(0, idx.to(tensor.device))


def denslice(array: csr_matrix, idx: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
    if idx is None:
        return torch.tensor(array.todense())
    if array is None:
        return None
    return torch.tensor(array[idx.cpu(), :].todense())


class UnconditionMessenger(pyro.poutine.messenger.Messenger):
    """A modified version of `pyro.poutine.messenger.UnconditionMessenger`
    where it is possible to specify which sites to uncondition."""

    def __init__(self, sites: List[str] = []):
        self.sites = sites

    def _pyro_sample(self, msg: Dict[str, Any]):
        if msg["name"] in self.sites and msg["is_observed"]:
            # The code below is taken from the
            # original `UnconditionMessenger`.
            msg["is_observed"] = False
            assert msg["infer"] is not None
            msg["infer"]["was_observed"] = True
            msg["infer"]["obs"] = msg["value"]
            msg["value"] = None
            msg["done"] = False


class plTrainHarness(pl.LightningModule):
    def __init__(self, cellavi, lr: float = 0.01):
        super().__init__()
        self.cellavi: Cellavi = cellavi
        self.pyro_model = cellavi.model
        self.pyro_guide = cellavi.guide
        self.lr = lr

        if cellavi.need_to_infer_cell_type:
            self.elbo = pyro.infer.TraceEnum_ELBO(
                num_particles=NUM_PARTICLES,
                vectorize_particles=True,
                max_plate_nesting=2,
                ignore_jit_warnings=True,
            )
        else:
            self.elbo = pyro.infer.Trace_ELBO(
                num_particles=NUM_PARTICLES,
                vectorize_particles=True,
                max_plate_nesting=2,
                ignore_jit_warnings=True,
            )

        # Instantiate parameters. Locals are deterministic.
        self.cellavi.gather_average_profiles()
        self.cellavi.initialize_locals()
        ncells = self.cellavi.ncells
        bsz = self.cellavi.bsz
        idx = torch.randperm(ncells)[:bsz].sort().values
        subX = denslice(self.cellavi.X, idx).to(self.cellavi.device)
        if self.cellavi.need_to_infer_cell_type or self.cellavi.need_to_infer_states:
            # Globals are random, find a good seed if needed.
            seed = self.find_initial_conditions(idx=idx)
            pyro.set_rng_seed(seed)
        self.cellavi.initialize_globals(subX=subX)

        # Instantiate parameters of autoguides.
        self.capture_params()

    def capture_params(self):
        with pyro.poutine.trace(param_only=True):
            self.elbo.differentiable_loss(
                model=self.pyro_model,
                guide=self.pyro_guide,
                # Use just one cell.
                idx=torch.tensor([0]),
            )

    def find_initial_conditions(self, idx, seed=None, sweep=512):
        subX = denslice(self.cellavi.X, idx).to(self.cellavi.device)
        losses = list()
        if seed is None:
            for seed in range(sweep):
                pyro.set_rng_seed(seed)
                self.cellavi.initialize_globals(subX)
                loss = self.elbo.differentiable_loss(self.pyro_model, self.pyro_guide, idx)
                losses.append(float(loss))
            loss, seed = min([(x, i) for (i, x) in enumerate(losses)])
        return seed

    def configure_optimizers(self):
        # Optimize parameters that are not frozen.
        def select(param_name):
            for name in self.cellavi.frozen:
                if name in param_name:
                    return False
            return True

        parameters = self.trainer.model.named_parameters()
        parameters_to_optimize = [param for (name, param) in parameters if select(name)]
        optimizer = torch.optim.Adam(parameters_to_optimize, lr=0.01)

        n_steps = self.trainer.estimated_stepping_batches
        n_warmup_steps = int(0.05 * n_steps)
        n_decay_steps = int(0.95 * n_steps)

        warmup = torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=0.01, end_factor=1.0, total_iters=n_warmup_steps
        )
        decay = torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=1.0, end_factor=0.01, total_iters=n_decay_steps
        )

        scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer=optimizer,
            schedulers=[warmup, decay],
            milestones=[n_warmup_steps],
        )

        return [optimizer], [{"scheduler": scheduler, "interval": "step"}]

    def training_step(self, batch):
        # Note: sorting the indices of the subsample is critical because
        # Pyro messengers subsample in sorted order. Without this line,
        # the parameters are completely randomized in the guide.
        idx = batch.sort().values
        loss = self.elbo.differentiable_loss(self.pyro_model, self.pyro_guide, idx)
        (lr,) = self.lr_schedulers().get_last_lr()
        info = {"loss": loss, "lr": lr}
        self.log_dict(dictionary=info, on_step=True, prog_bar=True, logger=True)
        self.cellavi.training_steps_performed = self.trainer.global_step
        return loss

    def compute_num_training_epochs(self):
        updates_per_epoch = int(self.cellavi.ncells / self.cellavi.bsz)
        num_epochs = int(MIN_NUM_GLOBAL_UPDATES / updates_per_epoch)
        # We need a minimum of `MIN_NUM_GLOBAL_UPDATES` updates for
        # global parameters. If we marginalize and do not compute
        # states (e.g., we infer cell types), there are no per-cell
        # parameters. Otherwise, there are per-cell parameters that
        # require at least `MIN_NUM_CELL_UPDATES` updates.
        if self.cellavi.need_to_infer_states or self.cellavi.marginalize is False:
            return max(MIN_NUM_CELL_UPDATES, num_epochs)
        else:
            return num_epochs


class Cellavi(pyro.nn.PyroModule):
    def __init__(self, data, marginalize=True, freeze=set(), device=None):
        super().__init__()

        # Unpack data.
        (self.ctype, self.batch, self.group, self.label, self.X, masks) = data
        (self.cmask, self.smask) = masks

        self.one_hot_ctype = F.one_hot(self.ctype, num_classes=C).float()
        # Format observed labels. Create one-hot encoding with label smoothing.
        # This is done by assigning value +2.3 or -2.3 so that the logits
        # stand for probabilities equal to 0.01 or 0.99.
        self.one_hot_label = F.one_hot(self.label, num_classes=K)
        self.slabel = 4.6 * self.one_hot_label - 2.3 if K > 1 else self.one_hot_label

        self.device = device if device is not None else self.X.device
        self.ncells = int(self.X.shape[0])

        self.bsz = self.ncells if self.ncells < SUBSMPL else SUBSMPL

        self.marginalize = marginalize
        self.training_steps_performed = 0

        # 1a) Define core parts of the model.
        self.output_scale_factor = self.sample_scale_factor
        self.output_global_base = self.sample_global_base
        self.output_base = self.sample_base
        self.output_scale_z = self.sample_scale_z

        # 1b) Define optional parts of the model.
        if B > 1:
            self.need_to_infer_batch_fx = True
            self.output_batch_fx = self.sample_batch_fx
            self.collect_batch_fx_i = self.compute_batch_fx_i
        else:
            self.need_to_infer_batch_fx = False
            self.output_batch_fx = self.zero
            self.collect_batch_fx_i = self.zero

        if C > 1:
            self.output_scale_tril_unit = self.sample_scale_tril_unit
            self.output_base = self.sample_base
        else:
            self.output_scale_tril_unit = self.one
            self.output_base = self.reshape_global_base

        if K > 1:
            self.need_to_infer_states = True
            self.output_states = self.sample_states
            self.output_theta_i = self.sample_theta_i
            self.collect_states_i = self.compute_states_i
        else:
            self.need_to_infer_states = False
            self.output_states = self.zero
            self.output_theta_i = self.zero
            self.collect_states_i = self.zero

        if self.cmask.all() or C == 1:
            self.need_to_infer_cell_type: bool = False
            self.output_c_indx = self.return_ctype_as_is
            self.collect_base_i = self.compute_base_i_no_enum
        else:
            self.need_to_infer_cell_type: bool = True
            self.output_c_indx = self.sample_c_indx
            self.collect_base_i = self.compute_base_i_enum

        # 2) Register frozen parameters.
        self.frozen = freeze

        # 3) Instantiate autoguide.
        # Local variables must be hidden because they are defined in the guide.
        local_variables = ["log_theta_i", "ctype_i_unobserved", "z_i", "x_i"]
        self.autonormal = AutoNormal(pyro.poutine.block(self.model, hide=local_variables))
        # Instantiate parameters now. Note that redefining `self.autonormal`
        # will destroy the current parameters of the autoguide.
        self.autonormal._setup_prototype(idx=torch.tensor([0]))  # Just one cell.

    def freeze(self, param):
        self.frozen.add(param)

    def unfreeze(self, param):
        self.frozen.remove(param)

    def gather_average_profiles(self):
        self.c_available = torch.unique(self.ctype[self.cmask])
        self.s_available = torch.unique(self.label[self.smask])
        self.avp = torch.ones(K, self.X.shape[-1]).to(self.device) / self.X.shape[-1]
        if self.s_available.any():
            labels = self.label[self.smask]
            subX = denslice(self.X, self.smask).to(self.device)
            for s in self.s_available:
                p = subX[labels == s] / subX[labels == s].sum(dim=-1, keepdim=True)
                self.avp[s] = p.mean(dim=0, keepdim=True)

    def initialize_globals(self, subX):
        param_store = pyro.get_param_store()
        average = (subX + 0.5).mean(dim=0)
        avlog = torch.log(average / average.sum(dim=-1, keepdim=True))
        avlog -= avlog.mean()  # zero-center.

        if "global_base" not in self.frozen:
            param_store["autonormal.locs.global_base"] = avlog

        if "base_0" not in self.frozen:
            if C == 1:
                # Only the global baseline is used in this case.
                base_0 = torch.zeros_like(avlog)
            else:
                # Initialize randomly from the data. Take the average of
                # two cells in order to avoid picking outliers as centroids.
                idx = torch.multinomial(torch.ones(subX.shape[0]) / subX.shape[0], 2 * C)
                two_cells = 0.5 * (subX[idx][:C] + subX[idx][C:]) + 0.5
                base_0 = torch.log(two_cells / two_cells.sum(dim=-1, keepdim=True)) - avlog
                for c in self.c_available:
                    # Pick a cell of the right type with probability 0.9, otherwise keep random pair.
                    if torch.rand(1).item() < 0.9:
                        # TODO: this won't work for very large data sets.
                        all_c = denslice(self.X, (self.ctype == c) & self.cmask).to(self.device)
                        idx = torch.randint(0, all_c.shape[0], (1,)).to(self.device)
                        one_cell = all_c[idx] + 0.5
                        base_0[c] = torch.log(one_cell / one_cell.sum(dim=-1, keepdim=True)) - avlog
                base_0 -= base_0.mean(dim=-1, keepdim=True)
                param_store["autonormal.locs.base_0"] = base_0.transpose(-1, -2)

        avbase_0 = base_0.mean(dim=0, keepdim=True)

        if "states_KR" not in self.frozen:
            # Initialize randomly from the data. Take the average of
            # two cells in order to avoid picking outliers as centroids.
            idx = torch.multinomial(torch.ones(subX.shape[0]) / subX.shape[0], 2 * K)
            two_cells = 0.5 * (subX[idx][:K] + subX[idx][K:]) + 0.5
            rows = torch.log(two_cells / two_cells.sum(dim=-1, keepdim=True)) - avlog - avbase_0
            for s in self.s_available:
                # Pick a cell of the right state with probability 0.9, otherwise keep random pair.
                if torch.rand(1).item() < 0.9:
                    all_s = denslice(self.X, (self.label == s) & self.smask).to(self.device)
                    idx = torch.randint(0, all_s.shape[0], (1,)).to(self.device)
                    one_cell = all_s[idx] + 0.5
                    rows[s] = torch.log(one_cell / one_cell.sum(dim=-1, keepdim=True)) - avlog - avbase_0
            param_store["autonormal.locs.states_KR"] = rows - rows.mean(dim=-1, keepdim=True)

    def initialize_locals(self):
        if self.marginalize is False:
            self.z_i_loc = pyro.nn.module.PyroParam(torch.zeros(G, self.ncells).to(self.device), event_dim=0)
            self.z_i_scale = pyro.nn.module.PyroParam(
                torch.ones(G, self.ncells).to(self.device),
                constraint=torch.distributions.constraints.positive,
                event_dim=0,
            )

        if self.need_to_infer_cell_type:
            import pdb

            pdb.set_trace()  # TODO: just like for states.
            # Initialize `c_indx_probs` randomly for cells where the
            # type is unknown, otherwise initialize with known type.
            # cdist = list()
            self.c_indx_probs = pyro.nn.module.PyroParam(
                torch.where(
                    self.cmask.unsqueeze(-1).expand(self.ncells, C),
                    self.one_hot_ctype,
                    5.0 + torch.rand(self.ncells, C).to(self.device),
                ),
                constraint=torch.distributions.constraints.simplex,
                event_dim=1,
            )

        if self.need_to_infer_states:
            # Initialize `theta` with state if known, otherwise
            # find close profiles if available, otherwise random.
            sdist = list()
            for i in range(0, self.X.shape[0], SUBSMPL):
                idx = torch.arange(self.X.shape[0])[i : i + SUBSMPL]
                subX = denslice(self.X, idx).to(self.device)
                subP = subX / subX.sum(dim=-1, keepdim=True)
                sdist.append(1.0 / torch.square(subP.unsqueeze(1) - self.avp.unsqueeze(0)).sum(dim=-1))
            sdist = torch.cat(sdist, dim=0)
            self.log_theta_i_loc = pyro.nn.module.PyroParam(
                torch.where(
                    self.smask.unsqueeze(-1).expand(self.ncells, K),
                    4.6 * (self.one_hot_label - 0.5),
                    sdist / sdist.sum(dim=-1, keepdim=True),
                ),
                event_dim=1,
            )
            self.log_theta_i_scale = pyro.nn.module.PyroParam(
                torch.ones(self.ncells, K).to(self.device),
                constraint=torch.distributions.constraints.positive,
                event_dim=1,
            )

    #  == Resample ==  #
    def resample(self, num_samples=200, sample_z_from_posterior=False):
        self.marginalize = False
        if sample_z_from_posterior:
            use_prior = []
            assert hasattr(self, "z_i_loc")
            assert hasattr(self, "z_i_scale")
        else:
            use_prior = ["z_i"]
            self.z_i_loc = torch.zeros(1, 1).to(self.device)
            self.z_i_scale = torch.ones(1, 1).to(self.device)
        guide = pyro.plate("samples", 10, dim=-3)(pyro.poutine.block(self.guide, hide=use_prior))
        model = pyro.plate("samples", 10, dim=-3)(pyro.poutine.block(self.model, hide=["x_i"]))
        samples = list()
        total_counts = [int(x) for x in self.X.sum(dim=-1)]
        with UnconditionMessenger(sites=["x_i"]), torch.no_grad():
            # Generate 10 sample at a time.
            for _ in range(0, num_samples, 10):
                guide_trace = pyro.poutine.trace(guide).get_trace()
                model_trace = pyro.poutine.trace(pyro.poutine.replay(model, guide_trace)).get_trace()
                probs_i = model_trace.nodes["probs_i"]["value"]

                def multi(c):
                    return torch.distributions.Multinomial(total_counts[c], probs=probs_i[:, c, :])

                x_i = torch.stack([multi(c).sample() for c in range(self.ncells)])
                samples.append(x_i.transpose(0, 1))
            sample = torch.cat(samples, dim=0)
        return sample

    #  == Helper functions ==  #
    def zero(self, *args, **kwargs):
        return torch.zeros(1).to(self.device)

    def one(self, *args, **kwargs):
        return torch.ones(1).to(self.device)

    #  ==  Model parts ==  #
    def sample_scale_tril_unit(self):
        scale_tril_unit = pyro.sample(
            name="scale_tril_unit",
            # dim(scale_tril_unit): (P x 1) x 1 | C x C
            fn=dist.LKJCholesky(dim=C, concentration=torch.ones(1).to(self.device)),
        )
        return scale_tril_unit

    def sample_scale_factor(self):
        scale_factor = pyro.sample(
            name="scale_factor",
            # dim(scale_factor): (P x 1) x C
            fn=dist.Exponential(
                rate=torch.ones(1).to(self.device),
            ),
        )
        # dim(scale_factor): (P x 1) x 1 x C
        scale_factor = scale_factor.unsqueeze(-2)
        return scale_factor

    def sample_global_base(self):
        global_base = pyro.sample(
            name="global_base",
            # dim(global_base): (P) x 1 x G
            fn=dist.StudentT(
                1.5 * torch.ones(1, 1).to(self.device),
                0.0 * torch.zeros(1, 1).to(self.device),
                1.0 * torch.ones(1, 1).to(self.device),
            ),
        )
        return global_base

    def sample_base(self, global_base, scale_tril):
        base_0 = pyro.sample(
            name="base_0",
            # dim(base): (P x 1) x G | C
            fn=dist.MultivariateNormal(torch.zeros(C).to(self.device), scale_tril=scale_tril),
        )
        # dim(base): (P) x G x C
        base = (global_base.unsqueeze(-1) + base_0).squeeze(-3)
        return base

    def sample_scale_z(self):
        scale_z = pyro.sample(
            name="scale_z",
            # dim(scale_z): (P) x 1 x G
            fn=dist.Exponential(4.0 * torch.ones(1, 1).to(self.device)),
        )
        # dim(scale_z): (P) x G x 1
        scale_z = scale_z.transpose(-1, -2)
        return scale_z

    def reshape_global_base(self, global_base, scale_tril):
        # dim(base): (P) x G x 1
        base = global_base[None, ...].unsqueeze(-1).squeeze(-3)
        return base

    def sample_batch_fx(self):
        batch_fx = pyro.sample(
            name="batch_fx",
            # dim(base): (P) x B x G
            fn=dist.Normal(
                0.00 * torch.zeros(1, 1).to(self.device),
                0.05 * torch.ones(1, 1).to(self.device),
            ),
        )
        return batch_fx

    def sample_states(self):
        states_KR = pyro.sample(
            name="states_KR",
            # dim(states_KR): (P) x KR x G
            fn=dist.Normal(0.0 * torch.zeros(1, 1).to(self.device), 0.7 * torch.ones(1, 1).to(self.device)),
        )
        # dim(states): (P) x K x R x G
        states = states_KR.view(states_KR.shape[:-2] + (K, R, G))
        return states

    def sample_c_indx(self, ctype_i, ctype_i_mask):
        sampled_ctype_i = pyro.sample(
            name="ctype_i",
            # dim(c_indx): C x (P) x 1 x ncells | C
            fn=dist.OneHotCategorical(
                torch.ones(1, 1, C).to(self.device),
            ),
            obs=ctype_i,
            obs_mask=ctype_i_mask,
            infer={"enumerate": "parallel"},
        )
        return sampled_ctype_i

    def return_ctype_as_is(self, ctype_i, cmask_i_mask):
        return ctype_i

    def sample_theta_i(self, slabel_i, slabel_i_mask):
        prior_loc = torch.where(
            slabel_i_mask.unsqueeze(-1).expand(slabel_i.shape),
            slabel_i,
            0.0 * torch.zeros_like(slabel_i),
        ).unsqueeze(-3)
        prior_scale = torch.where(
            slabel_i_mask.unsqueeze(-1).expand(slabel_i.shape),
            0.1 * torch.ones_like(slabel_i),
            1.0 * torch.ones_like(slabel_i),
        ).unsqueeze(-3)
        log_theta_i = pyro.sample(
            name="log_theta_i",
            # dim(log_theta_i): (P) x 1 x ncells | K
            fn=dist.Normal(loc=prior_loc, scale=prior_scale).to_event(1),
        )
        # dim(theta_i): (P) x 1 x ncells x K
        theta_i = log_theta_i.softmax(dim=-1)
        return theta_i

    def compute_base_i_enum(self, c_indx, base):
        # dim(c_indx): z x ncells x C (z = 1 or C)
        c_indx = c_indx.view((c_indx.shape[0],) + c_indx.shape[-2:])
        # dim(base_i): z x (P) x ncells x G (z = 1 or C)
        base_i = torch.einsum("znC,...GC->z...nG", c_indx, base)
        # dim(base_i): (P) x ncells x G or C x (P) x ncells x G
        base_i = base_i.squeeze(0)
        return base_i

    def compute_base_i_no_enum(self, c_indx, base):
        # dim(base_i): (P) x ncells x G
        base_i = torch.einsum("nC,...GC->...nG", c_indx, base)
        return base_i

    def compute_batch_fx_i(self, batch, batch_fx, indx_i):
        # dim(ohg): ncells x B
        ohb = subset(F.one_hot(batch).to(batch_fx), indx_i)
        # dim(batch_fx_i): (P) x ncells x G
        batch_fx_i = torch.einsum("...BG,nB->...nG", batch_fx, ohb)
        return batch_fx_i

    def compute_states_i(self, group, theta_i, states, indx_i, PoE=True):
        # dim(ohg): ncells x R
        ohg = subset(F.one_hot(group).to(states.dtype), indx_i)
        # dim(states_i): (P) x ncells x G
        if PoE:
            # This is the product-of-experts distribution.
            states_i = torch.einsum("...onK,...KRG,nR->...nG", theta_i, states, ohg)
        else:
            # This is the mixture distribution.
            states_ = torch.einsum("...KRG,nR->...nKG", states, ohg)
            log_theta = theta_i.log().squeeze(-3).unsqueeze(-1)
            states_i = torch.logsumexp(log_theta + states_, dim=-2)
        return states_i

    def compute_nu_and_k2(self, x_ij, mu, sg, idx):
        # dim(sg): (P) x 1 x G
        sg = sg.transpose(-1, -2)
        s2 = torch.square(sg)

        if self.need_to_infer_cell_type is False or mu.dim() == 2:
            avmu = mu
        else:
            self._pyro_context.active += 1
            # dim(c_indx_probs): ncells x 1 x C
            c_indx_probs = self.c_indx_probs.detach()
            self._pyro_context.active -= 1
            # dim(avmu): (P) x ncells x G
            if c_indx_probs.dim() > 2:
                avmu = torch.einsum("C...nG,onC->...nG", mu, c_indx_probs)
            else:
                avmu = torch.einsum("C...nG,nC->...nG", mu, c_indx_probs)

        # Remove gradient for Newton-Raphson cycles.
        # dim(mu_): ncells x G
        mu_ = avmu[None].mean(dim=-3).detach()
        # dim(s2_): ncells x G
        s2_ = 1.0 / (1.0 / s2[None].detach()).mean(dim=-3)

        xlog = x_ij.sum(dim=-1, keepdim=True).log()
        nu = -0.5 * torch.ones_like(mu_)

        T_ = xlog - torch.logsumexp(mu_, dim=-1, keepdim=True)
        for _ in range(4):
            k2 = s2_ / (s2_ * x_ij + 1 - nu)
            f = T_ + mu_ + nu + 0.5 * k2 - torch.log(x_ij - nu / s2_)
            df = 1 + 0.5 * s2_ / torch.square(s2_ * x_ij + 1 - nu) + 1.0 / (s2_ * x_ij - nu)
            nu = torch.clamp(nu - f / df, max=x_ij * s2_ - 0.01)

        for _ in range(5):
            nu__ = torch.clamp(nu - 1.0 / df, max=x_ij * s2_ - 0.001)
            k2__ = s2_ / (s2_ * x_ij + 1 - nu__)
            l1 = torch.logsumexp(mu_ + nu + 0.5 * k2, dim=-1, keepdim=True)
            l2 = torch.logsumexp(mu_ + nu__ + 0.5 * k2__, dim=-1, keepdim=True)
            delta = torch.clamp(-(T_ - xlog + l1) / (1 + l2 - l1), min=-1, max=1)
            nu = torch.clamp(nu - delta / df, max=x_ij * s2_ - 0.001)
            k2 = s2_ / (s2_ * x_ij + 1 - nu)
            T_ = xlog - torch.logsumexp(mu_ + nu + 0.5 * k2, dim=-1, keepdim=True)
            f = T_ + mu_ + nu + 0.5 * k2 - torch.log(x_ij - nu / s2_)
            df = 1 + 0.5 * s2_ / torch.square(s2_ * x_ij + 1 - nu) + 1.0 / (s2_ * x_ij - nu)
            nu = torch.clamp(nu - f / df, max=x_ij * s2_ - 0.001)

        k2 = s2_ / (s2_ * x_ij + 1 - nu)

        # Scale estimates over training steps.
        t = 1.0 - math.exp(-2 * (3 * self.training_steps_performed / MIN_NUM_GLOBAL_UPDATES) ** 2)
        nu = (0.1 + 0.9 * t) * nu
        k2 = (0.1 + 0.9 * t) * k2 + (0.9 - 0.9 * t) * torch.ones_like(k2)

        # Put a cap on `nu` and `k2` for numeric stability.
        nu = torch.clamp(nu, min=-256, max=256)
        k2 = torch.clamp(k2, min=-256, max=256)

        return nu, k2

    #  ==  model description == #
    def model(self, idx=None):
        # The correlation between cell types is given by the LKJ
        # distribution with parameter eta = 1, which is a uniform
        # prior over C x C correlation matrices. The parameter
        # `scale_tril_unit` is not the correlation matrix but the
        # lower Cholesky factor of the correlation matrix. It can
        # be passed directly to `MultivariateNormal`.

        # dim(scale_tril_unit): (P x 1) x 1 | C x C
        scale_tril_unit = self.output_scale_tril_unit()

        with pyro.plate("C", C, dim=-1):
            # The parameter `scale_factor` describes the standard
            # deviations for every cell type from the global
            # baseline. The prior is exponential, with 90% weight
            # in the interval (0.05, 3.00). The standard deviation
            # is applied to all the genes so it describes how far
            # the cell type is from the global baseline.

            # dim(scale_factor): (P x 1) x 1 x C
            scale_factor = self.output_scale_factor()

        # Set up `scale_tril` from the correlation and the standard
        # deviation. This is the lower Cholesky factor of the co-
        # variance matrix (can be used directly in `Normal`).

        # dim()scale_tril: (P x 1) x 1 x C x C
        scale_tril = scale_factor[None, ..., None] * scale_tril_unit

        # Per-gene sampling.
        with pyro.plate("G", G, dim=-1):
            # The global baseline represents the prior average
            # expression per gene. The parameters have a Student's t
            # distribution. The distribution is centered on 0,
            # because only the variations between genes are
            # considered here. The prior is chosen so that the
            # parameters have a 90% chance of lying in the interval
            # (-3.5, 3.5), i.e., there is a factor 1000 between the
            # bottom 5% and the top 5%. The distribution has a heavy
            # tail, the top 1% is 60,000 times higher than the
            # average.

            # dim(base): (P) x 1 x G
            global_base = self.output_global_base()

            # The baselines represent the average expression per
            # gene in each cell type. The distribution is centered
            # on 0, because we consider the deviations from the
            # global baseline. The prior is chosen so that the
            # parameters have a 90% chance of lying in the interval
            # (-3.5, 3.5), i.e., there is a factor 1000 between the
            # bottom 5% and the top 5%. The distribution has a heavy
            # tail, the top 1% is 60,000 times higher than the
            # average.

            # dim(base): (P) 1 x G | C
            base = self.output_base(global_base, scale_tril)

            # The variable `scale_z` describes how fuzzy every gene
            # is. It is the scale factor of the variable `z_i` that
            # is either sampled or marginalized. The distribution
            # is exponential with a 1% chance that the value is
            # above 1.15, meaning that 1% of the genes are expected
            # to vary by a factor 30, everything held constant.

            # dim(scale_z): (P) x G x 1
            scale_z = self.output_scale_z()

            # Per-batch, per-gene sampling.
            with pyro.plate("BxG", B, dim=-2):
                # Batch effects have a Gaussian distribution
                # centered on 0. They are weaker than 8% for
                # 95% of the genes.

                # dim(base): (P) x B x G
                batch_fx = self.output_batch_fx()

            # Per-unit, per-type, per-gene sampling.
            with pyro.plate("KRxG", K * R, dim=-2):
                # Effects have a Gaussian distribution
                # centered on 0. They have a 90% chance of
                # lying in the interval (-1.15, 1.15), which
                # corresponds to 3-fold effects.

                # dim(states): (P) x K x R x G
                states = self.output_states()

        # Per-cell sampling.
        with pyro.plate("ncells", self.ncells, dim=-1, subsample=idx, device=self.device) as indx_i:
            # Subset data and mask.
            one_hot_ctype_i = subset(self.one_hot_ctype, indx_i)
            ctype_i_mask = subset(self.cmask, indx_i)
            slabel_i = subset(self.slabel, indx_i)
            slabel_i_mask = subset(self.smask, indx_i)
            x_i = denslice(self.X, indx_i).to(self.device)

            # Cell types as discrete indicators. The prior
            # distiribution is uniform over known cell types.

            # dim(c_indx): C x (P) x 1 x ncells
            one_hot_c_indx = self.output_c_indx(one_hot_ctype_i, ctype_i_mask)

            # Proportion of each state in transcriptomes.
            # The proportions are computed from the softmax
            # of K standard Gaussian variables. This means
            # that there is a 90% chance that two proportions
            # are within a factor 10 of each other.

            # dim(theta_i): (P) x ncells x 1 x K
            theta_i = self.output_theta_i(slabel_i, slabel_i_mask)

            # Deterministic functions to collect per-cell means.

            # dim(base_i): C x (P) x ncells x G
            base_i = self.collect_base_i(one_hot_c_indx, base)

            # dim(batch_fx_i): (P) x ncells x G
            batch_fx_i = self.collect_batch_fx_i(self.batch, batch_fx, indx_i)

            # dim(states_i): (P) x ncells x G
            states_i = self.collect_states_i(self.group, theta_i, states, indx_i, PoE=False)

            # Expected expression of genes in log space.
            # dim(mu_i): (P) x ncells x G
            mu_i = base_i + batch_fx_i + states_i

            if self.marginalize:
                # Marginalize "z_i". Optimize the variational
                # parameters explicitly and add terms to the ELBO.
                nu, k2 = self.compute_nu_and_k2(x_i, mu_i, scale_z, indx_i)

                def log_Px(mu, nu, k2, x_i):
                    T = torch.logsumexp(mu + nu + 0.5 * k2, dim=-1)
                    return (
                        +torch.sum(x_i * (mu + nu), dim=-1)
                        - torch.sum(x_i, dim=-1) * T
                        - torch.lgamma(x_i + 1).sum(dim=-1)
                        + torch.lgamma(x_i.sum(dim=-1) + 1)
                    )

                def log_Pz(s2, nu, k2):
                    return (
                        -0.5 * torch.log(s2)
                        - 0.9189385  # log(sqrt(2pi))
                        - 0.5 * (torch.square(nu) + k2) / s2
                    )

                def log_Qz(s2, nu, k2):
                    return (
                        -0.5 * torch.log(k2)
                        - 0.9189385  # log(sqrt(2pi))
                        - 0.5
                    )

                log_px = log_Px(mu_i, nu, k2, x_i)
                log_pz = log_Pz(torch.square(scale_z.transpose(-1, -2)), nu, k2)
                log_qz = log_Qz(torch.square(scale_z.transpose(-1, -2)), nu, k2)

                with pyro.plate("Gxncells", G):
                    pyro.factor("z_i", (log_pz - log_qz).transpose(-1, -2))

                pyro.factor("x_i", log_px.unsqueeze(-2))

            else:
                # Do not marginalize "z_i". Sample it as a Normal
                # variable as usual.
                with pyro.plate("Gxncells", G):
                    z_i = pyro.sample(
                        name="z_i",
                        # dim(z_i): (P) x G x ncells
                        fn=dist.Normal(
                            loc=torch.zeros(1, 1).to(self.device),
                            scale=scale_z,
                        ),
                    )

                # dim(z_i): (P) x ncells x G
                z_i = z_i.transpose(-1, -2)

                # dim(probs_i): (P) x ncells x G
                probs_i = torch.softmax(mu_i + z_i, dim=-1)
                # Register `probs_i` for prediction purposes.
                pyro.deterministic("probs_i", probs_i)
                # dim(probs_i): (P) x 1 x ncells x G
                probs_i = probs_i.unsqueeze(-3)

                pyro.sample(
                    name="x_i",
                    # dim(x_i): (P) x 1 x ncells | G
                    fn=dist.Multinomial(
                        probs=probs_i,
                        validate_args=False,
                    ),
                    obs=x_i,
                )

    #  ==  guide description == #
    def guide(self, idx=None):
        # Sample all non-cell variables.
        self.autonormal(idx)

        # Per-cell sampling
        with pyro.plate("ncells", self.ncells, dim=-1, subsample=idx, device=self.device) as indx_i:
            # TODO: find canonical way to enter context of the module.
            self._pyro_context.active += 1

            # Subset data and mask.
            ctype_i_mask = subset(self.cmask, indx_i)

            # If more than one unit, sample them here.
            if self.need_to_infer_states:
                # Posterior distribution of `log_theta_i`.
                log_theta_i = pyro.sample(
                    name="log_theta_i",
                    # dim(log_theta_i): (P) x 1 x ncells | K
                    fn=dist.Normal(
                        self.log_theta_i_loc,
                        self.log_theta_i_scale,
                    ).to_event(1),
                )
                log_theta_i.shape

            if self.marginalize is False:
                with pyro.plate("Gxncells", G):
                    pyro.sample(
                        name="z_i",
                        # dim(z_i): (P) x G x ncells
                        fn=dist.Normal(
                            loc=self.z_i_loc,
                            scale=self.z_i_scale,
                        ),
                    )

            # If some cell types are unknown, sample them here.
            if self.need_to_infer_cell_type:
                with pyro.poutine.mask(mask=~ctype_i_mask):
                    pyro.sample(
                        name="ctype_i_unobserved",
                        # dim(c_indx): C x 1 x 1 x 1 | C
                        fn=dist.OneHotCategorical(
                            self.c_indx_probs,
                        ),
                        infer={"enumerate": "parallel"},
                    )

            self._pyro_context.active -= 1
