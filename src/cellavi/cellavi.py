import math
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


DEBUG = True
SUBSMPL = 512
NUM_PARTICLES = 12
MIN_CELL_UPDATES = 16
MIN_TOT_UPDATES = 2048

DEBUG_COUNTER = 0

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
        self.capture_params()

    def capture_params(self):
        with pyro.poutine.trace(param_only=True):
            self.elbo.differentiable_loss(
                model=self.pyro_model,
                guide=self.pyro_guide,
                # Use just one cell.
                idx=torch.tensor([0]),
            )

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

    def training_step(self, batch, batch_idx):
        # idx = batch.sort().values
        loss = self.elbo.differentiable_loss(self.pyro_model, self.pyro_guide, batch)
        (lr,) = self.lr_schedulers().get_last_lr()
        info = {"loss": loss, "lr": lr}
        self.log_dict(dictionary=info, on_step=True, prog_bar=True, logger=True)
        return loss


class Cellavi(pyro.nn.PyroModule):
    def __init__(self, data, marginalize=True):
        super().__init__()

        # Unpack data.
        self.ctype, self.batch, self.group, self.label, self.X, masks = data
        self.cmask, self.mmask = masks
        self.gmask = self.X.isnan()

        self.ctype = F.one_hot(self.ctype.view(-1, 1), num_classes=C).float()
        self.cmask = self.cmask.view(-1, 1)
        self.mmask = self.mmask.view(-1, 1)

        self.device = self.X.device
        self.ncells = int(self.X.shape[0])

        self.bsz = self.ncells if self.ncells < SUBSMPL else SUBSMPL

        # Format observed labels. Create one-hot encoding with label smoothing.
        # TODO: allow unit labels.
        #      oh = F.one_hot(self.label, num_classes=K).to(self.X.dtype)
        #      self.smooth_lab = ((.99-.01/(K-1)) * oh + .01/(K-1)).view(-1,1,K) if K > 1 else 0.
        self.smooth_lab = None

        self.marginalize = marginalize

        # 1a) Define core parts of the model.
        self.output_scale_factor = self.sample_scale_factor
        self.output_gene_fuzz = self.sample_gene_fuzz

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
            self.output_global_base = self.sample_global_base
            self.output_scale_tril_unit = self.sample_scale_tril_unit
            self.output_base = self.sample_base
        else:
            self.output_global_base = self.zero
            self.output_scale_tril_unit = self.one
            self.output_base = self.sample_base_0

        if K > 1:
            self.need_to_infer_moduls = True
            self.output_moduls = self.sample_moduls
            self.output_scale_moduls = self.sample_scale_moduls
            self.output_theta_i = self.sample_theta_i
            self.collect_moduls_i = self.compute_moduls_i
        else:
            self.need_to_infer_moduls = False
            self.output_scale_moduls = self.zero
            self.output_moduls = self.zero
            self.output_theta_i = self.zero
            self.collect_moduls_i = self.zero

        if cmask.all():
            self.need_to_infer_cell_type = False
            self.output_c_indx = self.return_ctype_as_is
            self.collect_base_i = self.compute_base_i_no_enum
        else:
            self.need_to_infer_cell_type = True
            self.output_c_indx = self.sample_c_indx
            self.collect_base_i = self.compute_base_i_enum

        # 2) Define the autoguide.
        self.autonormal = AutoNormal(
            pyro.poutine.block(self.model, hide=["log_theta_i", "cell_type_unobserved", "z_i"])
        )

        # 3) Define the guide parameters.
        if self.need_to_infer_cell_type:
            self.c_indx_probs = pyro.nn.module.PyroParam(
                0.1 * torch.ones(self.ncells, 1, C).to(self.device),
                constraint=torch.distributions.constraints.simplex,
                event_dim=1,
            )

        if self.need_to_infer_moduls:
            self.log_theta_i_loc = pyro.nn.module.PyroParam(torch.zeros(self.ncells, 1, K).to(self.device), event_dim=1)
            self.log_theta_i_scale = pyro.nn.module.PyroParam(
                0.1 * torch.ones(self.ncells, 1, K).to(self.device),
                constraint=torch.distributions.constraints.positive,
                event_dim=1,
            )

        if self.marginalize is False:
            self.z_i_loc = pyro.nn.module.PyroParam(torch.zeros(self.ncells, G).to(self.device), event_dim=0)
            self.z_i_scale = pyro.nn.module.PyroParam(
                0.5 * torch.ones(self.ncells, G).to(self.device),
                constraint=torch.distributions.constraints.positive,
                event_dim=0,
            )

    #  == Helper functions == #
    def one(self, *args, **kwargs):
        return 1.0

    def zero(self, *args, **kwargs):
        return 0.0

    #  ==  Model parts == #
    def sample_scale_moduls(self):
        scale_moduls = pyro.sample(
            name="scale_moduls",
            # dim(scale_tril_unit): (P x 1) x 1
            fn=dist.LogNormal(loc=-1.0 * torch.ones(1).to(self.device), scale=0.5 * torch.ones(1).to(self.device)),
        )
        return scale_moduls

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
            fn=dist.Exponential(
                5.0 * torch.ones(1, 1).to(self.device),
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
        # dim(base): (P) x G x C
        base = self.sample_base_0(global_base, scale_tril)
        base = base + global_base.unsqueeze(-1).squeeze(-3)
        return base

    def sample_base_0(self, global_base, scale_tril):
        base_0 = pyro.sample(
            name="base_0",
            # dim(base): (P) x 1 x G | C
            fn=dist.MultivariateNormal(torch.zeros(1, 1, C).to(self.device), scale_tril=scale_tril),
        )
        # dim(base): (P) x G x C
        base = base_0.squeeze(-3)
        return base

    def sample_gene_fuzz(self):
        gene_fuzz = pyro.sample(
            name="gene_fuzz",
            # dim(fuzz): (P) x 1 x G
            fn=dist.LogNormal(-2.0 * torch.zeros(1).to(self.device), torch.ones(1).to(self.device)),
        )
        return gene_fuzz

    def sample_batch_fx(self, scale):
        batch_fx = pyro.sample(
            name="batch_fx",
            # dim(base): (P) x B x G
            fn=dist.Normal(torch.zeros(1, 1).to(self.device), scale),
        )
        return batch_fx

    def sample_moduls(self, scale):
        moduls_KR = pyro.sample(
            name="moduls_KR",
            # dim(moduls_KR): (P) x KR x G
            fn=dist.Normal(torch.zeros(1, 1).to(self.device), scale),
        )
        # dim(moduls): (P) x K x R x G
        moduls = moduls_KR.view(moduls_KR.shape[:-2] + (K, R, G))
        return moduls

    def sample_c_indx(self, ctype_i, ctype_i_mask):
        c_indx = pyro.sample(
            name="cell_type",
            # dim(c_indx): C x (P) x ncells x 1 | C
            fn=dist.OneHotCategorical(
                torch.ones(1, 1, C).to(self.device),
            ),
            obs=ctype_i,
            obs_mask=ctype_i_mask,
            infer={"enumerate": "parallel"},
        )
        return c_indx

    def return_ctype_as_is(self, ctype_i, cmask_i_mask):
        return ctype_i

    def sample_theta_i(self, scale):
        log_theta_i = pyro.sample(
            name="log_theta_i",
            # dim(log_theta_i): (P) x ncells x 1 | K
            fn=dist.Normal(torch.zeros(1, 1, K).to(self.device), scale * torch.ones(1, 1, K).to(self.device)).to_event(
                1
            ),
        )
        # dim(theta_i): (P) x ncells x 1 x K
        theta_i = log_theta_i.softmax(dim=-1)
        return theta_i

    def compute_base_i_enum(self, c_indx, base):
        # dim(c_indx): z x ncells x C (z = 1 or C)
        c_indx = c_indx.view((-1,) + c_indx.shape[-3:]).squeeze(-2)
        # dim(base_i): z x (P) x ncells x G (z = 1 or C)
        base_i = torch.einsum("znC,...GC->z...nG", c_indx, base)
        return base_i

    def compute_base_i_no_enum(self, c_indx, base):
        # dim(c_indx): ncells x C
        c_indx = c_indx.squeeze(-2)
        # dim(base_i): (P) x ncells x G
        base_i = torch.einsum("nC,...GC->...nG", c_indx, base)
        return base_i

    def compute_batch_fx_i(self, batch, batch_fx, indx_i, dtype):
        # dim(ohg): ncells x B
        ohb = subset(F.one_hot(batch).to(dtype), indx_i)
        # dim(batch_fx_i): (P) x ncells x G
        batch_fx_i = torch.einsum("...BG,nB->...nG", batch_fx, ohb)
        return batch_fx_i

    def compute_moduls_i(self, group, theta_i, moduls, indx_i):
        # dim(ohg): ncells x R
        ohg = subset(F.one_hot(group).to(moduls.dtype), indx_i)
        # dim(moduls_i): (P) x ncells x G
        moduls_i = torch.einsum("...noK,...KRG,nR->...nG", theta_i, moduls, ohg)
        return moduls_i

    def compute_ELBO_z_i(self, x_ij, mu, sg, x_ij_mask, idx):
        # Parameters `mu` and `sg` are the prior parameters of the Poisson
        # LogNormal distribution. The variational posterior parameters
        # given the observations `x_ij` are `xi` and `w2_i`. In this case
        # we can compute the ELBO analytically and maximize it with respect
        # to `xi` and `w2_i` so as to pass the gradient to `mu` and `sg`.
        # This allows us to compute the ELBO efficiently without having
        # to store parameters and gradients for `xi` and `w2_i`.

        # FIXME: compute something during prototyping?
        if mu.dim() < 4:
            return

        self._pyro_context.active += 1
        # dim(c_indx_probs): ncells x 1 x C
        c_indx_probs = self.c_indx_probs.detach()
        self._pyro_context.active -= 1

        # dim(c_indx_probs): C x 1 x ncells x 1
        log_probs = c_indx_probs.detach().permute(2, 0, 1).unsqueeze(-3).log()
        log_P = math.log(mu.shape[-3])

        #      shift_i = x_ij.sum(dim=-1, keepdim=True).log() - \
        #         torch.logsumexp(mu.detach() + log_probs - log_P, dim=(-1,-3,-4)).unsqueeze(-1)
        C_ij = torch.logsumexp(mu.detach() + log_probs - log_P, dim=(-3, -4))
        # Harmonic mean of the variances.
        w2 = 1 / (1 / torch.square(sg.detach())).mean(dim=-3)
        xlog = x_ij.sum(dim=-1, keepdim=True).log()

        # dim(m): C x ncells x G
        # Initialize `xi` with dim: ncells x G.
        xi = (x_ij * w2 - 2.0) * torch.ones_like(C_ij)
        # Perform Newton-Raphson iterations.
        for _ in range(25):
            kap = w2 * x_ij + 1 - xi
            shift_i = torch.clamp(
                xlog - torch.logsumexp(C_ij + xi + 0.5 * w2 / kap, dim=-1, keepdim=True), min=-4, max=4
            )
            f = C_ij + shift_i + xi + 0.5 * w2 / kap - torch.log(x_ij - xi / w2)
            df = 1 + 0.5 * w2 / torch.square(kap) + 1.0 / (w2 * x_ij - xi)
            xi = torch.clamp(xi - f / df, max=x_ij * w2 - 0.01)

        # Set the optimal `w2_i` from the optimal `xi`.
        w2_i = w2 / (w2 * x_ij + 1 - xi)

        # Compute ELBO term as a function of `mu` and `sg`.
        def mini_ELBO_fn(mu, sg, xi, w2_i, x_ij, shift_i):
            return (
                -torch.exp(mu + shift_i + xi + 0.5 * w2_i)
                + x_ij * (mu + xi)
                - torch.log(sg)
                + 0.5 * torch.log(w2_i)
                - 0.5 * (xi * xi + w2_i) / (sg * sg)
            )

        mini_ELBO = mini_ELBO_fn(mu, sg, xi, w2_i, x_ij, shift_i)

        pyro.factor("PLN_ELBO_term", mini_ELBO.sum(dim=-1, keepdim=True))
        return x_ij

    #  ==  model description == #
    def model(self, idx=None):
        # TODO: describe prior.

        # dim(scale_moduls): (P x 1) x 1
        scale_log_theta = pyro.sample(
            name="scale_log_theta",
            # dim(scale_log_theta): (P x 1) x 1
            fn=dist.LogNormal(loc=-1.0 * torch.ones(1).to(self.device), scale=0.5 * torch.ones(1).to(self.device)),
        )
        # dim(scale_log_theta): (P x 1) x 1 x 1
        scale_log_theta = scale_log_theta.unsqueeze(-2)

        # The characteristic effect of the transcriptional modules
        # is give by the scale parameter. `scale_moduls` has a
        # log-normal distribution with 90% weight in the interval
        # (0.15, 0.85).

        # dim(scale_moduls): (P x 1) x 1
        scale_moduls = self.output_scale_moduls()

        # The correlation between cell types is given by the LKJ
        # distribution with parameter eta = 1, which is a uniform
        # prior over C x C correlation matrices. The parameter
        # `scale_tril_unit` is not the correlation matrix but the
        # lower Cholesky factor of the correlation matrix. It can
        # be passed directly to `MultivariateNormal`.

        # dim(scale_tril_unit): (P x 1) x 1 | C x C
        scale_tril_unit = self.output_scale_tril_unit()

        with pyro.plate("B", B, dim=-2):
            # The parameter `batch_fx_scale` describes the standard
            # deviations for every batch from the transcriptome of
            # the cell type. The prior is exponential, with 90% weight
            # in the interval (0.01, 0.60). The standard deviation
            # is applied to all the genes so it describes how far
            # the batch is from the prototype transcriptome.

            # dim(batch_fx_scale): (P) x B x 1
            batch_fx_scale = self.output_batch_fx_scale()

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

        # dim(scale_tril): (P x 1) x 1 x C x C
        scale_tril = scale_factor.unsqueeze(-1) * scale_tril_unit

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

            # dim(base): (P) x 1 x G | C
            base = self.output_base(global_base, scale_tril)

            # The parameter `gene_fuzz` is the standard deviation
            # for genes in the transcriptome. The prior is log-normal
            # with location parameter
            # For every gene, the
            # standard deviation is applied to all the cells, so it
            # describes how "fuzzy" a gene is, or on the contrary how
            # much it is determined by the cell type and its break down
            # in transcriptional modules.

            # dim(fuzz): (P) x 1 x G
            gene_fuzz = self.output_gene_fuzz()

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
                moduls = self.output_moduls(scale_moduls)

        # Per-cell sampling (on dimension -2).
        #        with pyro.plate("ncells", self.ncells, dim=-2, subsample=idx, device=self.device) as indx_i:
        with pyro.plate("ncells", self.ncells, dim=-2, device=self.device) as indx_i:
            # Subset data and mask.
            ctype_i = subset(self.ctype, indx_i)
            ctype_i_mask = subset(self.cmask, indx_i)
            x_i = subset(self.X, indx_i).to_dense()
            x_i_mask = subset(self.gmask, indx_i)

            # Cell types as discrete indicators. The prior
            # distiribution is uniform over known cell types.

            # dim(c_indx): C x (P) x ncells x 1
            c_indx = self.output_c_indx(ctype_i, ctype_i_mask)

            # Proportion of each unit in transcriptomes.
            # The proportions are computed from the softmax
            # of K Gaussian variables. The parameters are chosen
            # so that there is a 90% chance that two proportions
            # are within a factor 5 or each other.

            # dim(theta_i): (P) x ncells x 1 x K
            theta_i = self.output_theta_i(scale_log_theta)

            # Deterministic functions to collect per-cell means.

            # dim(base_i): C x (P) x ncells x G
            base_i = self.collect_base_i(c_indx, base)

            # dim(batch_fx_i): (P) x ncells x G
            batch_fx_i = self.collect_batch_fx_i(batch, batch_fx, indx_i, base.dtype)

            # dim(moduls_i): (P) x ncells x G
            moduls_i = self.collect_moduls_i(group, theta_i, moduls, indx_i)

            # Expected expression of gene in log space.
            mu_i = base_i + batch_fx_i + moduls_i

            if self.marginalize:
                return self.compute_ELBO_z_i(x_i, mu_i, gene_fuzz, x_i_mask, indx_i)

            else:
                # Per-cell, per-gene sampling.
                with pyro.plate("ncellsxG", G, dim=-1):
                    z_i = pyro.sample(
                        name="z_i",
                        # dim(z_i): ncells x G
                        fn=dist.Normal(
                            torch.zeros(1, 1).to(self.device),
                            0.1 * torch.ones(1, 1).to(self.device),
                            #                            gene_fuzz,
                        ),
                    )

                if self.need_to_infer_cell_type:
                    self._pyro_context.active += 1
                    # dim(c_indx_probs): ncells x 1 x C
                    c_indx_probs = self.c_indx_probs.detach()
                    self._pyro_context.active -= 1
                    # dim(c_indx_probs): C x 1 x ncells x 1
                    log_probs = c_indx_probs.detach().permute(2, 0, 1).unsqueeze(-3).log()
                else:
                    log_probs = 0.0

                # Pad with two singletons to make sure `m` has enough dimensions,
                # even when the model is called without parallel particles and
                # without enumeration.
                m = (mu_i.detach() + log_probs + z_i)[None, None, :]
                shift_i = x_i.sum(dim=-1, keepdim=True).log() - torch.logsumexp(
                    m[None, None, :], dim=(-1, -4), keepdim=True
                )

                rate_i = torch.clamp(torch.exp(z_i + mu_i + shift_i), max=1e6)

                pyro.sample(
                    name="x_i",
                    # dim(x_i): ncells x 1 | G
                    fn=dist.Poisson(
                        rate=rate_i.unsqueeze(-2),
                        validate_args=False,
                    ).to_event(1),
                    obs=x_i.unsqueeze(-2),
                )

                return

    #  ==  guide description == #
    def guide(self, idx=None):
        # Sample all non-cell variables.
        self.autonormal(idx)

        # Per-cell sampling (on dimension -2).
        #        with pyro.plate("ncells", self.ncells, dim=-2, subsample=idx, device=self.device) as indx_i:
        with pyro.plate("ncells", self.ncells, dim=-2, device=self.device) as indx_i:
            # Subset data and mask.
            ctype_i_mask = subset(self.cmask, indx_i)

            # TODO: find canonical way to enter context of the module.
            self._pyro_context.active += 1

            # If more than one unit, sample them here.
            if self.need_to_infer_moduls:
                # Posterior distribution of `log_theta_i`.
                pyro.sample(
                    name="log_theta_i",
                    # dim(log_theta_i): (P) x ncells x 1 | K
                    fn=dist.Normal(
                        self.log_theta_i_loc,
                        self.log_theta_i_scale,
                    ).to_event(1),
                )

            # If some cell types are unknown, sample them here.
            if self.need_to_infer_cell_type:
                with pyro.poutine.mask(mask=~ctype_i_mask):
                    pyro.sample(
                        name="cell_type_unobserved",
                        # dim(c_indx): C x 1 x 1 x 1 | C
                        fn=dist.OneHotCategorical(
                            self.c_indx_probs  # dim: ncells x 1 | C
                        ),
                        infer={"enumerate": "parallel"},
                    )

            # If `z_i` are not marginalized, sample them here.
            if self.marginalize is False:
                # Per-cell, per-gene sampling.
                with pyro.plate("ncellsxG", G, dim=-1):
                    # Posterior distribution of `z_i`.
                    pyro.sample(
                        name="z_i",
                        # dim(z_i): n x G
                        fn=dist.Normal(
                            self.z_i_loc,
                            self.z_i_scale,
                        ),
                    )

            self._pyro_context.active -= 1


def validate(data):
    (ctype, batch, group, modul, X, (cmask, mmask)) = data
    ncells = X.shape[0]
    assert len(ctype) == ncells
    assert len(batch) == ncells
    assert len(group) == ncells
    assert len(modul) == ncells
    assert len(cmask) == ncells
    assert len(mmask) == ncells


if __name__ == "__main__":
    pl.seed_everything(123)
    pyro.set_rng_seed(123)

    torch.set_float32_matmul_precision("medium")

    # TODO: Allow user to specify device.
    device = "cuda:0"

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
    mmask = info[5].to(device)

    X = read_dense_matrix(expr_path)
    X = X.to(device)

    # Set the dimensions.
    B = int(batch.max() + 1)
    C = int(ctype.max() + 1)
    R = int(group.max() + 1)
    G = int(X.shape[-1])

    data = (ctype, batch, group, modul, X, (cmask, mmask))
    data_idx = range(X.shape[0])
    validate(data)

    data_loader = torch.utils.data.DataLoader(
        dataset=data_idx,
        shuffle=True,
        batch_size=SUBSMPL,
    )

    pyro.clear_param_store()
    cellavi = Cellavi(data, marginalize=False)
    harnessed = plTrainHarness(cellavi)

    updates_per_epoch = math.ceil(X.shape[0] / SUBSMPL)
    num_epochs = math.ceil(max(MIN_CELL_UPDATES, MIN_TOT_UPDATES / updates_per_epoch))

    trainer = pl.Trainer(
        default_root_dir=".",
        strategy=pl.strategies.DeepSpeedStrategy(stage=2),
        accelerator="gpu",
        devices=[0],
        gradient_clip_val=1.0,
        max_epochs=num_epochs,
        enable_progress_bar=True,
        enable_model_summary=True,
        logger=pl.loggers.CSVLogger("."),
        enable_checkpointing=False,
    )

    trainer.fit(harnessed, data_loader)

    # Save output to file.
    param_store = pyro.get_param_store().get_state()
    for key, value in param_store["params"].items():
        param_store["params"][key] = value.clone().cpu().squeeze()
    torch.save(param_store, out_path)
