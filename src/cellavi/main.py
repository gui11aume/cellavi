import argparse
import sys
import warnings

import cellavi
import lightning.pytorch as pl
import pyro
import torch
from cellavi import Cellavi, plTrainHarness
from cellavi_data import CellaviCollator, CellaviData
from lightning.pytorch.callbacks import Callback
from misc_cellavi import load_parameters, read_h5ad, read_meta_from_file, read_mtx, read_text_matrix, update_ctmap
from tqdm.auto import tqdm

SUBSMPL = 512

# Suppress the specific warning about the number of workers.
warnings.filterwarnings("ignore", message=".*does not have many workers which may be a bottleneck.*")
warnings.filterwarnings("ignore", message=".*The epoch parameter in `scheduler.step()` was not necessary.*")


class CustomProgressBar(Callback):
    def __init__(self):
        self.train_bar = None
        self.val_bar = None

    def on_train_start(self, trainer, pl_module):
        self.train_bar = tqdm(total=trainer.max_epochs, position=0, desc="Training", leave=True)

    def on_train_epoch_end(self, trainer, pl_module):
        self.train_bar.update(1)
        self.train_bar.set_description(f"Training: Epoch {trainer.current_epoch + 1}/{trainer.max_epochs}")

    def on_validation_epoch_start(self, trainer, pl_module):
        if self.val_bar is None:
            self.val_bar = tqdm(total=trainer.max_epochs, position=1, desc="Validation", leave=True)

    def on_validation_epoch_end(self, trainer, pl_module):
        self.val_bar.update(1)
        self.val_bar.set_description(f"Validation: Epoch {trainer.current_epoch + 1}/{trainer.max_epochs}")

    def on_train_end(self, trainer, pl_module):
        self.train_bar.close()
        if self.val_bar:
            self.val_bar.close()


def main():
    parser = argparse.ArgumentParser(description="Cellavi")
    parser.add_argument("-K", type=int, default=1, help="number of topics (default: 1)")
    parser.add_argument("-C", type=int, default=0, help="number of cell types (default: auto)")
    parser.add_argument("--data_path", type=str, required=True, help="path to data file")
    parser.add_argument("--meta_path", type=str, help="path to metadata file")
    parser.add_argument("--out_path", type=str, required=True, help="path to output file")
    parser.add_argument("--product_of_experts", action="store_true", help="use product of expert")
    parser.add_argument("--parameters", type=str, help="path to file with parameters (optional)")
    parser.add_argument(
        "--mode", type=str, default="train", help="one of 'train', 'sample', 'freeze' (default: 'train')"
    )

    args = parser.parse_args()

    pyro.clear_param_store()
    torch.set_default_dtype(torch.float32)
    torch.set_float32_matmul_precision("high")

    meta_path = args.meta_path
    data_path = args.data_path
    out_path = args.out_path

    #######################################################
    if data_path.endswith(".h5ad"):
        X, meta = read_h5ad(data_path)
    elif data_path.endswith(".mtx"):
        read_mtx(data_path)
    else:
        X = read_text_matrix(data_path)

    # Overwrite h5ad metadata if another file is specified.
    if meta_path:
        meta = read_meta_from_file(meta_path)

    ctype = meta.ctypes_tensor
    batch = meta.batches_tensor
    group = meta.groups_tensor
    topic = meta.topics_tensor
    cmask = meta.ctype_mask_tensor
    smask = meta.topic_mask_tensor
    ctmap = meta.unique_ctypes

    # Make sure that the total number of topics is no smaller than
    # the number of known (specified) topics.
    if args.K < len(torch.unique(topic)):
        sys.exit("-K is less than the number of existing topics")

    # Make sure the total number of cell types is no smaller than
    # the number of known (specified) cell types.
    if (args.C > 0) and (args.C < len(ctmap)):
        sys.exit("-C is less than the number of existing cell types")

    PoE = args.product_of_experts

    #######################################################

    if args.parameters is not None:
        loaded_ctmap = load_parameters(args.parameters)
        ctmap, ctype = update_ctmap(ctmap, loaded_ctmap, ctype)

    # Set the dimensions.
    cellavi.K = args.K
    cellavi.C = args.C if args.C > 0 else len(ctmap)  # Number of cell types.
    cellavi.B = int(batch.max() + 1)  # Number of batches.
    cellavi.R = int(group.max() + 1)  # Number of groups.
    cellavi.G = int(X.shape[-1])  # Number of genes.

    ddata = CellaviData(
        x=X,
        ctype=ctype,
        batch=batch,
        group=group,
        topic=topic,
        cmask=cmask,
        smask=smask,
        chunk_size=SUBSMPL,
        K=cellavi.K,
        C=cellavi.C,
        B=cellavi.B,
        R=cellavi.R,
    )

    sdata = ddata.subsample_to(8192)

    model = Cellavi(ddata=sdata, PoE=PoE, amortize=False, collapse=False)

    if args.mode == "sample":
        sample = model.resample().cpu()
        torch.save(sample, out_path)
        return
    elif args.mode == "freeze":
        model.freeze("global_base")
        model.freeze("topics_KR")

    # The train data loaders are dummy lists of indices and the
    # collators return the corresponding rows of the data.
    # This is required because Pyro needs to know how to subset
    # the corresponding parameters.
    phase_1_data_loader = torch.utils.data.DataLoader(
        dataset=torch.arange(len(sdata)),
        shuffle=True,
        batch_size=cellavi.SUBSMPL,
        collate_fn=CellaviCollator(sdata),
    )

    phase_2_data_loader = torch.utils.data.DataLoader(
        dataset=torch.arange(len(ddata)),
        shuffle=True,
        batch_size=cellavi.SUBSMPL,
        collate_fn=CellaviCollator(ddata),
    )

    # The test data loader is the same dummy list of indices
    # but shuffling is turned off so that cells are processed in
    # the same order as in the input data. We also make the batch
    # size 64 times larger because we just call the amortizer
    # (no gradient updates are performed).
    test_data_loader = torch.utils.data.DataLoader(
        dataset=torch.arange(len(ddata)),
        shuffle=False,
        batch_size=64 * cellavi.SUBSMPL,
        collate_fn=CellaviCollator(ddata),
    )

    harnessed = plTrainHarness(model)

    trainer_args = {
        "default_root_dir": ".",
        "accelerator": "gpu",
        "gradient_clip_val": 1.0,
        "max_epochs": harnessed.compute_num_training_epochs(),
        "enable_progress_bar": False,
        "enable_model_summary": False,
        "logger": pl.loggers.CSVLogger("."),
        "log_every_n_steps": 1,
        "enable_checkpointing": False,
        "callbacks": [CustomProgressBar()],
    }

    trainer_phase_1 = pl.Trainer(
        strategy=pl.strategies.DeepSpeedStrategy(stage=2),
        **trainer_args,
    )
    trainer_phase_2 = pl.Trainer(
        strategy=pl.strategies.DeepSpeedStrategy(stage=2),
        **trainer_args,
    )

    pl.seed_everything(123)
    # Phase 1.
    trainer_phase_1.fit(harnessed, phase_1_data_loader)
    # Phase 2.
    model.switch_on_amortization()
    model.replace_data(ddata)
    trainer_phase_2.fit(harnessed, phase_2_data_loader)
    trainer_phase_2.test(harnessed, test_data_loader)

    # Save output to file.
    param_store = pyro.get_param_store().get_state()
    for key, value in param_store["params"].items():
        param_store["params"][key] = value.clone().cpu()
    # Store the cell type map.
    param_store["ctmap"] = ctmap
    torch.save(param_store, out_path)


if __name__ == "__main__":
    main()
