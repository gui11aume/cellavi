import sys

import lightning.pytorch as pl
import pyro
import pyro.distributions as dist
import torch
import torch.nn.functional as F
from misc_cellavi import (
    read_dense_matrix,
    read_info_from_file,
)
from pyro.infer.autoguide import (
    AutoNormal,
)

global K  # Number of moduls / set by user.
global B  # Number of batches / from data.
global C  # Number of types / from data.
global R  # Number of groups / from data.
global G  # Number of genes / from data.


DEBUG = False
SUBSMPL = 512
NUM_PARTICLES = 12
NUM_EPOCHS = 2048

# Use only for debugging.
pyro.enable_validation(DEBUG)


def subset(tensor, idx):
    if idx is None:
        return tensor
    if tensor is None:
        return None
    return tensor.index_select(0, idx.to(tensor.device))


class plTrainHarness(pl.LightningModule):
    def __init__(self, cellavi, lr=0.01):
        super().__init__()
        self.cellavi = cellavi
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

        # Instantiate parameters of autoguides.
        if self.cellavi.need_to_infer_moduls or self.cellavi.need_to_infer_cell_type:
            self.find_initial_conditions()

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

    def find_initial_conditions(self, seed=None, sweep=200):
        ncells = self.cellavi.ncells
        bsz = self.cellavi.bsz
        idx = torch.randperm(ncells)[:bsz].sort().values
        losses = list()
        if seed is None:
            for seed in range(sweep):
                pyro.set_rng_seed(seed)
                self.cellavi.reinitialize_auto_guide()
                loss = self.elbo.differentiable_loss(self.pyro_model, self.pyro_guide, idx)
                losses.append(float(loss))
            loss, seed = min([(x, i) for (i, x) in enumerate(losses)])
        pyro.set_rng_seed(seed)
        self.cellavi.reinitialize_auto_guide()

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.trainer.model.parameters(),
            lr=0.01,
        )

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
        return loss


class Cellavi(pyro.nn.PyroModule):
    def __init__(self, data):
        super().__init__()

        # Unpack data.
        self.ctype, self.batch, self.group, self.label, self.X, masks = data
        self.cmask, self.smask = masks

        self.one_hot_ctype = F.one_hot(self.ctype, num_classes=C).float()
        # Format observed labels. Create one-hot encoding with label smoothing.
        # This is done by assigning value +2.3 or -2.3 so that the logits
        # stand for probabilities equal to 0.01 or 0.99.
        self.one_hot_label = F.one_hot(self.label, num_classes=K).to(self.X.dtype)
        self.slabel = 4.6 * self.one_hot_label - 2.3 if K > 1 else 0.0

        self.device = self.X.device
        self.ncells = int(self.X.shape[0])

        self.bsz = self.ncells if self.ncells < SUBSMPL else SUBSMPL

        # 1a) Define core parts of the model.
        self.output_scale_tril_unit = self.sample_scale_tril_unit
        self.output_scale_factor = self.sample_scale_factor
        self.output_global_base = self.sample_global_base
        self.output_base = self.sample_base

        # 1b) Define optional parts of the model.
        if B > 1:
            self.need_to_infer_batch_fx = True
            self.output_batch_fx_scale = self.sample_batch_fx_scale
            self.output_batch_fx = self.sample_batch_fx
            self.collect_batch_fx_i = self.compute_batch_fx_i
        else:
            self.need_to_infer_batch_fx = False
            self.output_batch_fx_scale = self.zero
            self.output_batch_fx = self.zero
            self.collect_batch_fx_i = self.zero

        if C > 1:
            self.output_scale_tril_unit = self.sample_scale_tril_unit
            self.output_base = self.sample_base
        else:
            self.output_scale_tril_unit = self.one
            self.output_base = self.reshape_global_base

        if K > 1:
            self.need_to_infer_moduls = True
            self.output_moduls = self.sample_moduls
            self.output_theta_i = self.sample_theta_i
            self.collect_probs_i = self.compute_probs_i
        else:
            self.need_to_infer_moduls = False
            self.output_moduls = self.zero
            self.output_theta_i = self.zero
            self.collect_probs_i = self.zero

        if cmask.all() or C == 1:
            self.need_to_infer_cell_type = False
            self.output_c_indx = self.return_ctype_as_is
            self.collect_base_i = self.compute_base_i_no_enum
        else:
            self.need_to_infer_cell_type = True
            self.output_c_indx = self.sample_c_indx
            self.collect_base_i = self.compute_base_i_enum

        # 2) Define the autoguide.
        self.autonormal = AutoNormal(
            pyro.poutine.block(self.model, hide=["log_theta_i_unobserved", "ctype_i_unobserved", "z_i"])
        )

        # 3) Define the guide parameters.
        self.z_i_loc = pyro.nn.module.PyroParam(torch.zeros(G, self.ncells).to(self.device), event_dim=0)
        self.z_i_scale = pyro.nn.module.PyroParam(
            torch.ones(G, self.ncells).to(self.device),
            constraint=torch.distributions.constraints.positive,
            event_dim=0,
        )

        if self.need_to_infer_cell_type:
            # Initialize `c_indx_probs` randomly for cells where the
            # type is unknown, otherwise initialize with known type.
            self.c_indx_probs = pyro.nn.module.PyroParam(
                torch.where(
                    self.cmask.unsqueeze(-1).expand(self.ncells, C),
                    self.one_hot_ctype,
                    5.0 + torch.rand(self.ncells, C).to(self.device),
                ),
                constraint=torch.distributions.constraints.simplex,
                event_dim=1,
            )

        if self.need_to_infer_moduls:
            # Initialize `theta` randomly for cells where the state
            # is unknown, otherwise initialize with known state.
            self.log_theta_i_loc = pyro.nn.module.PyroParam(
                torch.where(
                    self.smask.unsqueeze(-1).expand(self.ncells, K),
                    4.6 * (self.one_hot_label - 0.5),
                    0.1 * torch.randn(self.ncells, K).to(self.device),
                ),
                event_dim=1,
            )
            self.log_theta_i_scale = pyro.nn.module.PyroParam(
                torch.ones(self.ncells, K).to(self.device),
                constraint=torch.distributions.constraints.positive,
                event_dim=1,
            )

    def init_loc_fn(self, site):
        if site["name"] == "moduls_KR":
            idx = torch.multinomial(torch.ones(self.ncells) / self.ncells, K * R).to(self.device)
            rows = subset(torch.log(0.5 + self.X), idx)
            return rows - rows.mean(dim=0, keepdim=True)
        if site["name"] == "base_0":
            avlog = torch.log(0.5 + self.X).mean(dim=0, keepdim=True)
            if self.cmask.all():
                zero = torch.zeros(C, G).to(self.device)
                index = self.ctype.unsqueeze(-1).expand(self.X.shape)
                base_0 = zero.scatter_reduce(0, index, torch.log(0.5 + self.X), "mean")
                return (base_0 - avlog).transpose(-1, -2)
            else:
                idx = torch.multionomial(torch.ones(self.ncells) / self.ncells, C).to(self.device)
                base_0 = subset(torch.log(0.5 + self.X), idx)
                return (base_0 - avlog).transpose(-1, -2)
        if site["name"] == "global_base":
            avlog = torch.log(0.5 + self.X).mean(dim=0, keepdim=True)
            return avlog - avlog.mean()

    def reinitialize_auto_guide(self):
        # Reinitialize the parameters of the autoguide.
        self.autonormal = AutoNormal(
            pyro.poutine.block(self.model, hide=["log_theta_i_unobserved", "ctype_i_unobserved", "z_i"]),
            init_loc_fn=self.init_loc_fn,
        )

    #  == Helper functions == #
    def zero(self, *args, **kwargs):
        return torch.zeros(1).to(self.device)

    def one(self, *args, **kwargs):
        return torch.ones(1).to(self.device)

    #  ==  Model parts == #
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

    def sample_batch_fx_scale(self):
        batch_fx_scale = pyro.sample(
            name="batch_fx_scale",
            # dim(base): (P) x 1 x B
            fn=dist.HalfNormal(
                0.05 * torch.ones(1, 1).to(self.device),
            ),
        )
        return batch_fx_scale

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

    def reshape_global_base(self, global_base, scale_tril):
        # dim(base): (P) x G x 1
        base = global_base[None, ...].unsqueeze(-1).squeeze(-3)
        return base

    def sample_batch_fx(self, scale):
        batch_fx = pyro.sample(
            name="batch_fx",
            # dim(base): (P) x B x G
            fn=dist.Normal(torch.zeros(1, 1).to(self.device), scale),
        )
        return batch_fx

    def sample_moduls(self):
        moduls_KR = pyro.sample(
            name="moduls_KR",
            # dim(moduls_KR): (P) x KR x G
            fn=dist.Normal(0.0 * torch.zeros(1, 1).to(self.device), 0.7 * torch.ones(1, 1).to(self.device)),
        )
        # dim(moduls): (P) x K x R x G
        moduls = moduls_KR.view(moduls_KR.shape[:-2] + (K, R, G))
        return moduls

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
        log_theta_i = pyro.sample(
            name="log_theta_i",
            # dim(log_theta_i): (P) x 1 x ncells | K
            fn=dist.Normal(
                loc=torch.zeros(1, 1, K).to(self.device), scale=torch.ones(1, 1, K).to(self.device)
            ).to_event(1),
            obs=slabel_i,
            obs_mask=slabel_i_mask,
        )
        # dim(theta_i): (P) x 1 x ncells x K
        theta_i = log_theta_i.softmax(dim=-1)
        return theta_i

    def compute_base_i_enum(self, c_indx, base):
        # dim(base_i): z x (P) x ncells x G (z = 1 or C)
        base_i = torch.einsum("znC,...GC->z...nG", c_indx, base)
        # dim(base_i): (P) x ncells x G or C x (P) x ncells x G
        base_i = base_i.squeeze(0)
        return base_i

    def compute_base_i_no_enum(self, c_indx, base):
        # dim(base_i): (P) x ncells x G
        base_i = torch.einsum("nC,...GC->...nG", c_indx, base)
        return base_i

    def compute_batch_fx_i(self, batch, batch_fx, indx_i, dtype):
        # dim(ohg): ncells x B
        ohb = subset(F.one_hot(batch).to(dtype), indx_i)
        # dim(batch_fx_i): (P) x ncells x G
        batch_fx_i = torch.einsum("...BG,nB->...nG", batch_fx, ohb)
        return batch_fx_i

    def compute_probs_i(self, group, theta_i, base, moduls, indx_i):
        # dim(ohg): ncells x R
        ohg = subset(F.one_hot(group).to(moduls.dtype), indx_i)
        moduls_i = torch.einsum("...KRG,nR->...KnG", moduls, ohg)
        profiles_i = (moduls_i + base.unsqueeze(-3)).softmax(dim=-1)
        # dim(probs_i): (P) x ncells x G
        probs_i = torch.einsum("...onK,...KnG->...nG", theta_i, profiles_i)
        return probs_i

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

        with pyro.plate("B", B, dim=-2):
            # The parameter `batch_fx_scale` describes the standard
            # deviations for every batch from the transcriptome of
            # the cell type. The prior is half-normal, the quantiles
            # 0.9, .99, .9999 equal to .064, .116 and .186. Beyond
            # this, the probabilities drop fast, so that values
            # above 0.25 are exceedingly unlikely. The standard
            # deviation is applied to all the genes so it describes
            # how far the batch is from the baseline transcriptome.
            # A value of 0.25 means that every gene in the batch may
            # vary by a factor ~2 compared to the baseline.

            # dim(batch_fx_scale): (P) x B x 1
            batch_fx_scale = self.output_batch_fx_scale()

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

            # TODO: give a proper name to this variable and
            # describe the prior.
            the_scale = pyro.sample(
                name="the_scale",
                # dim(the_scale): (P) x 1 x G
                fn=dist.Exponential(6.0 * torch.ones(1, 1).to(self.device)),
            )
            # dim(the_scale): (P) x G x 1
            the_scale = the_scale.transpose(-1, -2)

            # Per-batch, per-gene sampling.
            with pyro.plate("BxG", B, dim=-2):
                # Batch effects have a Gaussian distribution
                # centered on 0. They are weaker than 8% for
                # 95% of the genes.

                # dim(base): (P) x B x G
                batch_fx = self.output_batch_fx(batch_fx_scale)

            # Per-unit, per-type, per-gene sampling.
            with pyro.plate("KRxG", K * R, dim=-2):
                # Unit effects have a Gaussian distribution
                # centered on 0. They have a 90% chance of
                # lying in the interval (-1.15, 1.15), which
                # corresponds to 3-fold effects.

                # dim(moduls): (P) x K x R x G
                moduls = self.output_moduls()

        # Per-cell sampling.
        with pyro.plate("ncells", self.ncells, dim=-1, subsample=idx, device=self.device) as indx_i:
            # Subset data and mask.
            one_hot_ctype_i = subset(self.one_hot_ctype, indx_i)
            ctype_i_mask = subset(self.cmask, indx_i)
            slabel_i = subset(self.slabel, indx_i)
            slabel_i_mask = subset(self.smask, indx_i)
            x_i = subset(self.X, indx_i).to_dense()

            # Cell types as discrete indicators. The prior
            # distiribution is uniform over known cell types.

            # dim(c_indx): C x (P) x 1 x ncells
            one_hot_c_indx = self.output_c_indx(one_hot_ctype_i, ctype_i_mask)

            # Proportion of each modul in transcriptomes.
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
            batch_fx_i = self.collect_batch_fx_i(batch, batch_fx, indx_i, base.dtype)

            # Expected expression of gene in log space (logits).
            # dim(mu_i): (P) x ncells x G
            mu_i = base_i + batch_fx_i

            with pyro.plate("Gxncells", G):
                z_i = pyro.sample(
                    name="z_i",
                    # dim(z_i): (P) x G x ncells
                    fn=dist.Normal(
                        loc=torch.zeros(1, 1).to(self.device),
                        scale=the_scale,
                    ),
                )

            # dim(z_i): (P) x ncells x G
            z_i = z_i.transpose(-1, -2)
            # dim(base_prob): (P) x ncells x G
            mu_i_plus_z_i = mu_i + z_i

            # dim(probs_i): (P) x ncells x G
            probs_i = self.collect_probs_i(group, theta_i, mu_i_plus_z_i, moduls, indx_i)

            x = pyro.sample(
                name="x_i",
                # dim(x_i): ncells | G
                fn=dist.Multinomial(
                    probs=probs_i,
                    validate_args=False,
                ),
                obs=x_i,
            )

            return x

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
            if self.need_to_infer_moduls:
                # Posterior distribution of `log_theta_i`.
                pyro.sample(
                    name="log_theta_i_unobserved",
                    # dim(log_theta_i): (P) x 1 x ncells | K
                    fn=dist.Normal(
                        self.log_theta_i_loc,
                        self.log_theta_i_scale,
                    ).to_event(1),
                )

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


def validate(data):
    (ctype, batch, group, modul, X, (cmask, smask)) = data
    ncells = X.shape[0]
    assert len(ctype) == ncells
    assert len(batch) == ncells
    assert len(group) == ncells
    assert len(modul) == ncells
    assert len(cmask) == ncells
    assert len(smask) == ncells


if __name__ == "__main__":
    pl.seed_everything(123)
    torch.set_float32_matmul_precision("high")

    device = "cuda"

    K = int(sys.argv[1])
    info_path = sys.argv[2]
    expr_path = sys.argv[3]
    out_path = sys.argv[4]

    info = read_info_from_file(info_path)

    ctype = info[0].to(device)
    batch = info[1].to(device)
    group = info[2].to(device)
    modul = info[3].to(device)
    cmask = info[4].to(device)
    smask = info[5].to(device)

    X = read_dense_matrix(expr_path)
    X = X.to(device)

    # Set the dimensions.
    B = int(batch.max() + 1)
    C = int(ctype.max() + 1)
    R = int(group.max() + 1)
    G = int(X.shape[-1])

    data = (ctype, batch, group, modul, X, (cmask, smask))
    data_idx = range(X.shape[0])
    validate(data)

    data_loader = torch.utils.data.DataLoader(
        dataset=data_idx,
        shuffle=True,
        batch_size=SUBSMPL,
    )

    pyro.clear_param_store()
    cellavi = Cellavi(data)
    harnessed = plTrainHarness(cellavi)

    trainer = pl.Trainer(
        default_root_dir=".",
        strategy=pl.strategies.DeepSpeedStrategy(stage=2),
        accelerator="gpu",
        gradient_clip_val=1.0,
        max_epochs=NUM_EPOCHS,
        enable_progress_bar=True,
        enable_model_summary=True,
        logger=pl.loggers.CSVLogger("."),
        enable_checkpointing=False,
    )

    trainer.fit(harnessed, data_loader)

    # Save output to file.
    param_store = pyro.get_param_store().get_state()
    for key, value in param_store["params"].items():
        param_store["params"][key] = value.clone().squeeze().cpu()
    torch.save(param_store, out_path)
