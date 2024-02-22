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
SUBSMPL = 524
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
    def __init__(self, prodlda, lr=0.01):
        super().__init__()
        self.prodlda = prodlda
        self.pyro_model = prodlda.model
        self.pyro_guide = prodlda.guide
        self.lr = lr

        self.elbo = pyro.infer.Trace_ELBO(
            num_particles=NUM_PARTICLES,
            vectorize_particles=True,
            max_plate_nesting=2,
            ignore_jit_warnings=True,
        )

        # Instantiate parameters of autoguides.
        if self.prodlda.need_to_infer_moduls:
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
        ncells = self.prodlda.ncells
        bsz = self.prodlda.bsz
        idx = torch.randperm(ncells)[:bsz].sort().values
        losses = list()
        if seed is None:
            for seed in range(sweep):
                pyro.set_rng_seed(seed)
                self.prodlda.reinitialize_auto_guide()
                loss = self.elbo.differentiable_loss(self.pyro_model, self.pyro_guide, idx)
                losses.append(float(loss))
            loss, seed = min([(x, i) for (i, x) in enumerate(losses)])
        pyro.set_rng_seed(seed)
        self.prodlda.reinitialize_auto_guide()

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


class prodLDA(pyro.nn.PyroModule):
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

        # 1b) Define optional parts of the model.
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

        # 2) Define the autoguide.
        self.autonormal = AutoNormal(pyro.poutine.block(self.model, hide=["log_theta_i_unobserved"]))

        # 3) Define the guide parameters.
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

    def reinitialize_auto_guide(self):
        # Reinitialize the parameters of the autoguide.
        self.autonormal = AutoNormal(
            pyro.poutine.block(self.model, hide=["log_theta_i_unobserved"]), init_loc_fn=self.init_loc_fn
        )

    #  == Predict ==  #
    def predict(self, num_samples=200):
        guide = pyro.plate("samples", 10, dim=-3)(self.guide)
        model = pyro.plate("samples", 10, dim=-3)(self.model)
        samples = list()
        with torch.no_grad():
            for _ in range(0, num_samples, 10):
                guide_trace = pyro.poutine.trace(guide).get_trace()
                model_trace = pyro.poutine.trace(pyro.poutine.replay(model, guide_trace)).get_trace()
                samples.append(model_trace.nodes["probs_i"]["value"])
            sample = torch.cat(samples, dim=0).squeeze()
        return sample

    #  == Helper functions ==  #
    def zero(self, *args, **kwargs):
        return torch.zeros(1).to(self.device)

    def one(self, *args, **kwargs):
        return torch.ones(1).to(self.device)

    #  ==  Model parts ==  #
    def sample_moduls(self):
        moduls_KR = pyro.sample(
            name="moduls_KR",
            # dim(moduls_KR): (P) x KR x G
            fn=dist.Normal(0.0 * torch.zeros(1, 1).to(self.device), 0.7 * torch.ones(1, 1).to(self.device)),
        )
        # dim(moduls): (P) x K x R x G
        moduls = moduls_KR.view(moduls_KR.shape[:-2] + (K, R, G))
        return moduls

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

    def compute_probs_i(self, group, theta_i, moduls, indx_i):
        # dim(ohg): ncells x R
        ohg = subset(F.one_hot(group).to(moduls.dtype), indx_i)
        moduls_i = torch.einsum("...KRG,nR->...KnG", moduls, ohg)
        profiles_i = moduls_i.softmax(dim=-1)
        # dim(probs_i): (P) x ncells x G
        probs_i = torch.einsum("...onK,...KnG->...nG", theta_i, profiles_i)
        return probs_i

    #  ==  model description == #
    def model(self, idx=None):
        # Per-gene sampling.
        with pyro.plate("G", G, dim=-1):
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
            slabel_i = subset(self.slabel, indx_i)
            slabel_i_mask = subset(self.smask, indx_i)
            x_i = subset(self.X, indx_i).to_dense()

            # Proportion of each modul in transcriptomes.
            # The proportions are computed from the softmax
            # of K standard Gaussian variables. This means
            # that there is a 90% chance that two proportions
            # are within a factor 10 of each other.

            # dim(theta_i): (P) x ncells x 1 x K
            theta_i = self.output_theta_i(slabel_i, slabel_i_mask)

            # dim(probs_i): (P) x ncells x G
            probs_i = self.collect_probs_i(group, theta_i, moduls, indx_i)

            # Register `probs_i` for prediction purposes.
            pyro.deterministic("probs_i", probs_i)

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
        with pyro.plate("ncells", self.ncells, dim=-1, subsample=idx, device=self.device):
            # TODO: find canonical way to enter context of the module.
            self._pyro_context.active += 1

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

    pyro.clear_param_store()
    prodlda = prodLDA(data)
    if len(sys.argv) > 5:
        # Prediction.
        store = torch.load(sys.argv[5])
        for key in store["params"]:
            store["params"][key] = store["params"][key].to(device)
        pyro.get_param_store().set_state(store)
        sample = prodlda.predict().cpu()
        torch.save(sample, out_path)
    else:
        # Fitting.
        data_loader = torch.utils.data.DataLoader(
            dataset=data_idx,
            shuffle=True,
            batch_size=SUBSMPL,
        )

        harnessed = plTrainHarness(prodlda)

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
