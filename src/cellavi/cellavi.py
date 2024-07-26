import math
import warnings
from typing import Any, Dict, List

import lightning.pytorch as pl
import pyro
import pyro.distributions as dist
import torch
import torch.nn.functional as F
from cellavi_initializer import initialize_parameters
from pyro.infer.autoguide import AutoNormal

global K  # Number of topics
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
MIN_NUM_EPOCHS = 24


# Use only for debugging.
pyro.enable_validation(DEBUG)


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

    def capture_params(self):
        # just_one_cell = self.cellavi.expr[0].to(self.device)
        # just_one_idx = torch.tensor([0]).to(self.device)
        with pyro.poutine.trace(param_only=True):
            self.elbo.differentiable_loss(
                model=self.pyro_model,
                guide=self.pyro_guide,
                # idx=torch.tensor([0]),
                ddata_i=self.cellavi.ddata[0],
            )

    def configure_optimizers(self):
        # Optimize parameters that are not frozen.
        def pick(param_name):
            for name in self.cellavi.frozen:
                if name in param_name:
                    return False
            return True

        model_parameters = set(self.trainer.model.named_parameters())
        trainable_params = set([param for (name, param) in model_parameters if pick(name)])
        optimizer = torch.optim.AdamW(trainable_params, lr=0.01)

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

    def setup(self, stage=None):
        if stage == "fit":
            with torch.device(self.cellavi.device):
                data = self.cellavi.ddata
                amortizer = self.cellavi.amortizer
                initialize_parameters(data, amortizer)

            # Instantiate parameters of autoguides.
            self.capture_params()

        if stage == "test":
            if self.cellavi.amortize is True:
                self.log_theta_i_loc = list()
                self.log_theta_i_scale = list()

    def training_step(self, batch):
        # Note: sorting the indices of the subsample is critical because
        # Pyro messengers subsample in sorted order. Without this line,
        # the parameters are completely randomized in the guide.
        loss = self.elbo.differentiable_loss(self.pyro_model, self.pyro_guide, batch)
        learning_rates = self.lr_schedulers().get_last_lr()
        info = {"loss": loss, "lr": learning_rates[0]}
        self.log_dict(dictionary=info, on_step=True, prog_bar=True, logger=True)
        self.cellavi.training_steps_performed = self.trainer.global_step
        return loss

    def test_step(self, batch):
        if self.cellavi.amortize is False:
            warnings.warn("Amortization is disabled.")
            return
        # Use amortizer data to compute `log_theta_i_loc` and `log_theta_i_scale`.
        x_i = batch.x.to(self.cellavi.device)
        ohb_i = batch.one_hot_batch
        ohc_i = batch.one_hot_ctype
        ohg_i = batch.one_hot_group
        freq_i = x_i / x_i.sum(dim=-1, keepdim=True)
        bcgf_i = torch.cat([ohb_i, ohc_i, ohg_i, freq_i], dim=-1)
        loc_i, scale_i = self.cellavi.amortizer(bcgf_i)
        self.log_theta_i_loc.append(loc_i)
        self.log_theta_i_scale.append(scale_i)

    def on_test_end(self):
        if self.cellavi.amortize is False:
            return
        pyro.param(name="log_theta_i_loc", init_tensor=torch.cat(self.log_theta_i_loc, dim=0))
        pyro.param(
            name="log_theta_i_scale",
            init_tensor=torch.cat(self.log_theta_i_scale, dim=0),
            constraint=torch.distributions.constraints.positive,
        )

    def compute_num_training_epochs(self):
        updates_per_epoch = int(self.cellavi.ncells / self.cellavi.bsz)
        num_epochs = 1 + int(MIN_NUM_GLOBAL_UPDATES / updates_per_epoch)
        # With amortization, there are no local (i.e., per-cell) parameters
        # so se we just need to train the amortizer and the global parameters.
        # Otherwise, we must make sure that every cell goes through a certain
        # number of updates so that the local parameters are optimized enough.
        if self.cellavi.amortize is False:
            num_epochs = max(num_epochs, MIN_NUM_EPOCHS)
        return num_epochs


class InferenceNetwork(pyro.nn.PyroModule):
    def __init__(self, input_size, hidden_size, output_size, dropout_rate=0.3):
        super().__init__()
        self.layer1 = pyro.nn.PyroModule[torch.nn.Linear](input_size, hidden_size)
        self.layer2 = pyro.nn.PyroModule[torch.nn.Linear](hidden_size, hidden_size)
        self.dropout = torch.nn.Dropout(dropout_rate)
        self.layer_loc = pyro.nn.PyroModule[torch.nn.Linear](hidden_size, output_size)
        self.layer_scale = pyro.nn.PyroModule[torch.nn.Linear](hidden_size, output_size)
        # Make sure that scale is initially close to 0.
        torch.nn.init.constant_(self.layer_scale.bias, -3.0)

    def forward(self, x):
        x = F.gelu(self.layer1(x))
        x = self.dropout(x)
        x = F.gelu(self.layer2(x))
        x = self.dropout(x)
        loc = self.layer_loc(x)
        scale = F.softplus(self.layer_scale(x))
        return loc, scale


class Cellavi(pyro.nn.PyroModule):
    def __init__(self, ddata, PoE=False, collapse=True, amortize=True, freeze=set(), device="cuda:0"):
        super().__init__()

        self.ddata = ddata
        self.PoE = PoE

        self.device = device
        self.ncells = int(self.ddata.x.shape[0])

        self.bsz = self.ncells if self.ncells < SUBSMPL else SUBSMPL

        self.collapse = collapse
        self.amortize = amortize
        self.training_steps_performed = 0

        # 1a) Define core parts of the model.
        self.output_scale_factor = self.sample_scale_factor
        self.output_global_base = self.sample_global_base
        self.output_scale_z = self.sample_scale_z
        self.output_topics = self.sample_topics
        self.output_theta_i = self.sample_theta_i
        self.collect_topics_i = self.compute_topics_i

        # 1b) Define optional parts of the model.
        if B == 1:
            self.need_to_infer_batch_fx = False
            self.output_batch_fx = self.zero
            self.collect_batch_fx_i = self.zero
        else:
            self.need_to_infer_batch_fx = True
            self.output_batch_fx = self.sample_batch_fx
            self.collect_batch_fx_i = self.compute_batch_fx_i

        if C == 1:
            self.need_to_infer_cell_type = False
            self.output_ctype_fx = self.zero
            self.collect_ctype_fx_i = self.zero
            self.output_c_indx = self.return_ctype_as_is
        else:
            self.output_ctype_fx = self.sample_ctype_fx
            self.collect_ctype_fx_i = self.compute_ctype_fx_i_no_enum
            if ddata.cmask.all():
                # All cell types are known.
                self.need_to_infer_cell_type = False
                self.output_c_indx = self.return_ctype_as_is
            else:
                # Some cell types are unknown.
                self.need_to_infer_cell_type = True
                self.output_c_indx = self.sample_c_indx

        # 2) Register frozen parameters.
        self.frozen = freeze

        # 3) Instantiate autoguide.
        # Local variables must be hidden because they are defined in the guide.
        local_variables = ["log_theta_i", "ctype_i_unobserved", "z_i", "x_i"]
        self.autonormal = AutoNormal(pyro.poutine.block(self.model, hide=local_variables))
        # Instantiate parameters now. Note that redefining `self.autonormal`
        # will destroy the current parameters of the autoguide.
        # TODO: test if this is really useful (doing something).
        self.autonormal._setup_prototype(self.ddata[0])

        # 4) Instantiate inference network
        if self.amortize is True:
            self.amortizer = InferenceNetwork(
                # Make space for batch, cell type, and group.
                input_size=G + B + C + R,
                hidden_size=128,
                output_size=K,
                dropout_rate=0.3,
            ).to(self.device)

        # When collapsed variational inference is disabled, we need to
        # instantiate and initialize the latent parameters of the
        # Poisson-LogNormal / Logistic-Normal distribution `self.z_i_loc`,
        # `self.z_i_scale`.
        if self.collapse is False:
            self.z_i_loc = pyro.nn.module.PyroParam(torch.zeros(G, self.ncells).to(self.device), event_dim=0)
            self.z_i_scale = pyro.nn.module.PyroParam(
                torch.ones(G, self.ncells).to(self.device),
                constraint=torch.distributions.constraints.positive,
                event_dim=0,
            )

    def freeze(self, param):
        self.frozen.add(param)

    def unfreeze(self, param):
        self.frozen.remove(param)

    #  == Resample ==  #
    def resample(self, num_samples=200, sample_z_from_posterior=False):
        self.collapse = False
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
        total_counts = [int(x) for x in self.ddata.x.sum(dim=-1)]
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
    def sample_scale_factor(self):
        scale_factor = pyro.sample(
            name="scale_factor",
            # dim(scale_factor): (P x 1) x C
            fn=dist.Exponential(
                rate=torch.ones(1).to(self.device),
            ),
        )
        return scale_factor

    def sample_global_base(self):
        global_base = pyro.sample(
            name="global_base",
            # dim(global_base): (P) x 1 x G
            fn=dist.StudentT(
                df=1.5 * torch.ones(1, 1).to(self.device),
                loc=0.0 * torch.zeros(1, 1).to(self.device),
                scale=1.0 * torch.ones(1, 1).to(self.device),
            ),
        )
        return global_base

    def sample_ctype_fx(self, scale_factor):
        ctype_fx = pyro.sample(
            name="ctype_fx",
            # dim(ctype_fx): (P) x C x G
            fn=dist.Normal(
                torch.zeros(1, 1).to(self.device),
                scale_factor,
            ),
        )
        return ctype_fx

    def sample_scale_z(self):
        scale_z = pyro.sample(
            name="scale_z",
            # dim(scale_z): (P) x 1 x G
            fn=dist.Exponential(4.0 * torch.ones(1, 1).to(self.device)),
        )
        # dim(scale_z): (P) x G x 1
        scale_z = scale_z.transpose(-1, -2)
        return scale_z

    def sample_batch_fx(self):
        batch_fx = pyro.sample(
            name="batch_fx",
            # dim(batch_fx): (P) x B x G
            fn=dist.Normal(
                0.00 * torch.zeros(1, 1).to(self.device),
                0.05 * torch.ones(1, 1).to(self.device),
            ),
        )
        return batch_fx

    def sample_topics(self):
        # topics_KR = pyro.sample(
        #     name="topics_KR",
        #     # dim(topics_KR): (P) x KR x G
        #     fn=dist.Normal(
        #         0.0 * torch.zeros(1, 1).to(self.device),
        #         0.7 * torch.ones(1, 1).to(self.device)
        #     ),
        # )
        # Student t prior.
        topics_KR = pyro.sample(
            name="topics_KR",
            # dim(topics_KR): (P) x KR x G
            fn=dist.StudentT(
                df=1.0 * torch.ones(1, 1).to(self.device),
                loc=0.0 * torch.zeros(1, 1).to(self.device),
                scale=0.01 * torch.ones(1, 1).to(self.device),
            ),
        )
        # dim(topics): (P) x K x R x G
        topics = topics_KR.view(topics_KR.shape[:-2] + (K, R, G))
        return topics

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
        # Tweak the prior, depending on whether labels are specified.
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

    def compute_ctype_fx_i_enum(self, ctype_fx, c_indx):
        # dim(c_indx): z x ncells x C (z = 1 or C)
        c_indx = c_indx.view((c_indx.shape[0],) + c_indx.shape[-2:])
        # dim(ctype_fx_i): z x (P) x ncells x G (z = 1 or C)
        ctype_fx_i = torch.einsum("znC,...CG->z...nG", c_indx, ctype_fx)
        # dim(ctype_fx_i): (P) x ncells x G or C x (P) x ncells x G
        ctype_fx_i = ctype_fx_i.squeeze(0)
        return ctype_fx_i

    def compute_ctype_fx_i_no_enum(self, c_indx, ctype_fx):
        # dim(ctype_fx_i): (P) x ncells x G
        ctype_fx_i = torch.einsum("...CG,nC->...nG", c_indx, ctype_fx)
        return ctype_fx_i

    def compute_batch_fx_i(self, batch_fx, one_hot_batch_i):
        # dim(batch_fx_i): (P) x ncells x G
        batch_fx_i = torch.einsum("...BG,nB->...nG", batch_fx, one_hot_batch_i)
        return batch_fx_i

    def compute_topics_i(self, one_hot_group_i, theta_i, topics, PoE=False):
        # dim(topics_i): (P) x ncells x G
        if PoE:
            # This is the product-of-experts distribution.
            topics_i = torch.einsum("...onK,...KRG,nR->...nG", theta_i, topics, one_hot_group_i)
        else:
            # This is the mixture distribution.
            topics_ = torch.einsum("...KRG,nR->...nKG", topics, one_hot_group_i)
            log_theta = theta_i.log().squeeze(-3).unsqueeze(-1)
            topics_i = torch.logsumexp(log_theta + topics_, dim=-2)
        return topics_i

    def compute_nu_and_k2(self, x_ij, mu, sg):
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

        min_nu = -32 * torch.ones_like(x_ij)
        max_nu = torch.clamp(x_ij * s2_ - 0.001, max=32)

        T_ = xlog - torch.logsumexp(mu_, dim=-1, keepdim=True)
        for _ in range(4):
            k2 = torch.clamp(s2_ / (s2_ * x_ij + 1 - nu), max=32)
            f = T_ + mu_ + nu + 0.5 * k2 - torch.log(x_ij - nu / s2_)
            df = 1 + 0.5 * s2_ / torch.square(s2_ * x_ij + 1 - nu) + 1.0 / (s2_ * x_ij - nu)
            nu = torch.clamp(nu - f / df, min=min_nu, max=max_nu)

        for _ in range(5):
            nu__ = torch.clamp(nu - 1.0 / df, min=min_nu, max=max_nu)
            k2__ = torch.clamp(s2_ / (s2_ * x_ij + 1 - nu__), max=32)
            l1 = torch.logsumexp(mu_ + nu + 0.5 * k2, dim=-1, keepdim=True)
            l2 = torch.logsumexp(mu_ + nu__ + 0.5 * k2__, dim=-1, keepdim=True)
            delta = torch.clamp(-(T_ - xlog + l1) / (1 + l2 - l1), min=-1, max=1)
            nu = torch.clamp(nu - delta / df, min=min_nu, max=max_nu)
            k2 = torch.clamp(s2_ / (s2_ * x_ij + 1 - nu), max=32)
            T_ = xlog - torch.logsumexp(mu_ + nu + 0.5 * k2, dim=-1, keepdim=True)
            f = T_ + mu_ + nu + 0.5 * k2 - torch.log(x_ij - nu / s2_)
            df = 1 + 0.5 * s2_ / torch.square(s2_ * x_ij + 1 - nu) + 1.0 / (s2_ * x_ij - nu)
            nu = torch.clamp(nu - f / df, min=min_nu, max=max_nu)

        k2 = torch.clamp(s2_ / (s2_ * x_ij + 1 - nu), max=32)

        # Scale estimates over training steps.
        t = 1.0 - math.exp(-2 * (3 * self.training_steps_performed / MIN_NUM_GLOBAL_UPDATES) ** 2)
        nu = (0.1 + 0.9 * t) * nu
        k2 = (0.1 + 0.9 * t) * k2 + (0.9 - 0.9 * t) * torch.ones_like(k2)

        # Put a cap on `nu` and `k2` for numeric stability.
        nu = torch.clamp(nu, min=min_nu, max=max_nu)
        k2 = torch.clamp(k2, min=1e-6, max=32)

        return nu, k2

    #  ==  model description == #
    def model(self, ddata_i=None):
        with pyro.plate("C", C, dim=-2):
            # The parameter `scale_factor` describes the standard
            # deviations for every cell type from the global
            # baseline. The prior is exponential, with 90% weight
            # in the interval (0.05, 3.00). The standard deviation
            # is applied to all the genes so it describes how far
            # the cell type is from the global baseline.

            # dim(scale_factor): (P) x C x 1
            scale_factor = self.output_scale_factor()

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

            # dim(global_base): (P) x 1 x G
            global_base = self.output_global_base()

            # The variable `scale_z` describes how fuzzy every gene
            # is. It is the scale factor of the variable `z_i` that
            # is either sampled or collapsed. The distribution
            # is exponential with a 1% chance that the value is
            # above 1.15, meaning that 1% of the genes are expected
            # to vary by a factor 30, everything held constant.

            # dim(scale_z): (P) x G x 1
            scale_z = self.output_scale_z()

            with pyro.plate("CxG", C, dim=-2):
                # The cell type effects have a Gaussian distribution
                # centered on 0. The dispersion is set by the hyper-
                # parameter `scale_factor`.

                # dim(base): (P) x C x G
                ctype_fx = self.output_ctype_fx(scale_factor)

            # Per-batch, per-gene sampling.
            with pyro.plate("BxG", B, dim=-2):
                # Batch effects have a Gaussian distribution centered
                # on 0. They are weaker than 8% for 95% of the genes.

                # dim(batch_fx): (P) x B x G
                batch_fx = self.output_batch_fx()

            # Per-topic, per-group, per-gene sampling.
            with pyro.plate("KRxG", K * R, dim=-2):
                # Topics have a Gaussian distribution
                # centered on 0. They have a 90% chance of
                # lying in the interval (-1.15, 1.15), which
                # corresponds to 3-fold effects.

                # dim(topics): (P) x K x R x G
                topics = self.output_topics()

        # Per-cell sampling.
        idx = ddata_i.idx_i if ddata_i is not None else None
        with pyro.plate("ncells", self.ncells, dim=-1, subsample=idx, device=self.device):
            # Subset data and mask.
            x_i = ddata_i.x.to(self.device)
            one_hot_ctype_i = ddata_i.one_hot_ctype.to(self.device)
            one_hot_batch_i = ddata_i.one_hot_batch.to(self.device)
            one_hot_group_i = ddata_i.one_hot_group.to(self.device)
            ctype_i_mask = ddata_i.cmask.to(self.device)
            slabel_i = ddata_i.stopic.to(self.device)
            slabel_i_mask = ddata_i.smask.to(self.device)

            # dim(c_indx): C x (P) x 1 x ncells
            one_hot_c_indx = self.output_c_indx(one_hot_ctype_i, ctype_i_mask)

            # Proportion of each topic in transcriptomes.
            # The proportions are computed from the softmax
            # of K standard Gaussian variables. This means
            # that there is a 90% chance that two proportions
            # are within a factor 10 of each other.

            # dim(theta_i): (P) x ncells x 1 x K
            theta_i = self.output_theta_i(slabel_i, slabel_i_mask)

            # Deterministic functions to collect per-cell means.

            # dim(ctype_fx_i): C x (P) x ncells x G
            ctype_fx_i = self.collect_ctype_fx_i(ctype_fx, one_hot_c_indx)

            # dim(batch_fx_i): (P) x ncells x G
            batch_fx_i = self.collect_batch_fx_i(batch_fx, one_hot_batch_i)

            # dim(topics_i): (P) x ncells x G
            topics_i = self.collect_topics_i(one_hot_group_i, theta_i, topics, PoE=self.PoE)

            # Expected expression of genes in log space.
            # dim(mu_i): (P) x ncells x G
            mu_i = global_base + ctype_fx_i + batch_fx_i + topics_i

            if self.collapse:
                # Collapse "z_i". Optimize the variational
                # parameters explicitly and add terms to the ELBO.
                nu, k2 = self.compute_nu_and_k2(x_i, mu_i, scale_z)

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
                    return -0.5 * torch.log(k2) - 0.9189385 - 0.5  # log(sqrt(2pi))

                log_px = log_Px(mu_i, nu, k2, x_i)
                log_pz = log_Pz(torch.square(scale_z.transpose(-1, -2)), nu, k2)
                log_qz = log_Qz(torch.square(scale_z.transpose(-1, -2)), nu, k2)

                with pyro.plate("Gxncells", G):
                    pyro.factor("z_i", (log_pz - log_qz).transpose(-1, -2))

                pyro.factor("x_i", log_px.unsqueeze(-2))

            else:
                # Do not collapse "z_i", sample it as usual.
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
    def guide(self, ddata_i=None):
        # Sample all non-cell variables.
        self.autonormal(ddata_i)

        # Per-cell sampling
        idx = ddata_i.idx_i if ddata_i is not None else None
        with pyro.plate("ncells", self.ncells, dim=-1, subsample=idx, device=self.device):
            x_i = ddata_i.x.to(self.device)
            # TODO: find canonical way to enter context of the module.
            self._pyro_context.active += 1

            # Subset data and mask.
            ctype_i_mask = ddata_i.cmask.to(self.device)

            # Get topic-breakdown parameters by either calling the amortizer
            # on the input data or by pulling learnable parameters.
            if self.amortize is True:
                ohb_i = ddata_i.one_hot_batch.to(self.device)
                ohc_i = ddata_i.one_hot_ctype.to(self.device)
                ohg_i = ddata_i.one_hot_group.to(self.device)
                freq_i = x_i / x_i.sum(dim=-1, keepdim=True)
                bcgf_i = torch.cat([ohb_i, ohc_i, ohg_i, freq_i], dim=-1)
                (log_theta_i_loc, log_theta_i_scale) = self.amortizer(bcgf_i)
            else:
                log_theta_i_loc = self.log_theta_i_loc
                log_theta_i_scale = self.log_theta_i_scale

            # Posterior distribution of `log_theta_i`.
            pyro.sample(
                name="log_theta_i",
                # dim(log_theta_i): (P) x 1 x ncells | K
                fn=dist.Normal(
                    log_theta_i_loc,
                    log_theta_i_scale,
                ).to_event(1),
            )

            if self.collapse is False:
                with pyro.plate("Gxncells", G):
                    pyro.sample(
                        name="z_i",
                        # dim(z_i): (P) x 1 x G x ncells
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
