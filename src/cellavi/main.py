import argparse
import sys

import cellavi
import lightning.pytorch as pl
import pyro
import torch
from cellavi import Cellavi, plTrainHarness
from misc_cellavi import load_parameters, read_dense_matrix, read_h5ad, read_meta_from_file, read_mtx, update_ctmap


def validate(data):
    # Validate data (make sure that all tensors have the same length).
    ctype: torch.tensor
    batch: torch.tensor
    group: torch.tensor
    modul: torch.tensor
    X: torch.tensor
    cmask: torch.tensor
    smask: torch.tensor

    (ctype, batch, group, modul, X, (cmask, smask)) = data
    ncells = X.shape[0]

    assert len(ctype) == ncells
    assert len(batch) == ncells
    assert len(group) == ncells
    assert len(modul) == ncells
    assert len(cmask) == ncells
    assert len(smask) == ncells


def main():
    parser = argparse.ArgumentParser(description="Cellavi")
    parser.add_argument("-K", type=int, default=1, help="number of moduls (default: 1)")
    parser.add_argument("-C", type=int, default=0, help="number of cell types (default: auto)")
    parser.add_argument("--data_path", type=str, required=True, help="path to data file")
    parser.add_argument("--meta_path", type=str, help="path to metadata file")
    parser.add_argument("--out_path", type=str, required=True, help="path to output file")
    parser.add_argument("--device", type=str, default="cuda", help="'cpu', 'cuda', 'cuda:0', ... (default: 'cuda')")
    parser.add_argument("--parameters", type=str, help="path to file with parameters (optional)")
    parser.add_argument(
        "--mode", type=str, default="train", help="one of 'train', 'sample', 'freeze' (default: 'train')"
    )

    args = parser.parse_args()

    pyro.clear_param_store()
    torch.set_float32_matmul_precision("high")

    device = args.device

    meta_path = args.meta_path
    data_path = args.data_path
    out_path = args.out_path

    #######################################################
    if data_path.endswith(".h5ad"):
        X, meta = read_h5ad(data_path)
    elif data_path.endswith(".mtx"):
        read_mtx(data_path)
    else:
        X = read_dense_matrix(data_path)

    # X is a `scipy` CSR matrix now.
    # X = X.to(device)

    # Overwrite h5ad metadata if another file is specified.
    if meta_path:
        meta = read_meta_from_file(meta_path)

    ctype = meta.ctypes_tensor.to(device)
    batch = meta.batches_tensor.to(device)
    group = meta.groups_tensor.to(device)
    modul = meta.states_tensor.to(device)
    cmask = meta.ctype_mask_tensor.to(device)
    smask = meta.state_mask_tensor.to(device)
    ctmap = meta.unique_ctypes

    # Make sure that the total number of moduls is no smaller than
    # the number of known (specified) moduls.
    if args.K < len(torch.unique(modul)):
        sys.exit("-K is less than the number of existing states")

    # Make sure the total number of cell types is no smaller than
    # the number of known (specified) cell types.
    if (args.C > 0) and (args.C < len(ctmap)):
        sys.exit("-C is less than the number of existing cell types")

    #######################################################

    if args.parameters is not None:
        loaded_ctmap = load_parameters(args.parameters, device)
        ctmap, ctype = update_ctmap(ctmap, loaded_ctmap, ctype)

    # Set the dimensions.
    cellavi.K = args.K
    cellavi.B = int(batch.max() + 1)  # Number of batches.
    cellavi.C = args.C if args.C > 0 else len(ctmap)  # Number of cell types.
    cellavi.R = int(group.max() + 1)  # Number of groups.
    cellavi.G = int(X.shape[-1])  # Number of genes.

    data = (ctype, batch, group, modul, X, (cmask, smask))
    data_idx = range(X.shape[0])
    validate(data)

    model = Cellavi(data, device=device)

    if args.mode == "sample":
        sample = model.resample().cpu()
        torch.save(sample, out_path)
        return
    elif args.mode == "freeze":
        model.freeze("global_base")
        model.freeze("moduls_KR")

    # Fitting.
    data_loader = torch.utils.data.DataLoader(
        dataset=data_idx,
        shuffle=True,
        batch_size=cellavi.SUBSMPL,
    )

    harnessed = plTrainHarness(model)

    trainer = pl.Trainer(
        default_root_dir=".",
        strategy=pl.strategies.DeepSpeedStrategy(stage=2),
        accelerator="gpu" if "cuda" in device else None,
        gradient_clip_val=1.0,
        max_epochs=harnessed.compute_num_training_epochs(),
        enable_progress_bar=True,
        enable_model_summary=True,
        logger=pl.loggers.CSVLogger("."),
        enable_checkpointing=False,
    )

    pl.seed_everything(123)
    trainer.fit(harnessed, data_loader)

    # Save output to file.
    param_store = pyro.get_param_store().get_state()
    for key, value in param_store["params"].items():
        param_store["params"][key] = value.clone().cpu()
    # Store the cell type map.
    param_store["ctmap"] = ctmap
    torch.save(param_store, out_path)


if __name__ == "__main__":
    main()
