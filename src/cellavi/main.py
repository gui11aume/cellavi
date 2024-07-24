import argparse
import sys

import cellavi
import lightning.pytorch as pl
import pyro
import torch
from cellavi import Cellavi, plTrainHarness
from cellavi_data import CellaviCollator, CellaviData
from misc_cellavi import load_parameters, read_h5ad, read_meta_from_file, read_mtx, read_text_matrix, update_ctmap

SUBSMPL = 512


def main():
    parser = argparse.ArgumentParser(description="Cellavi")
    parser.add_argument("-K", type=int, default=1, help="number of topics (default: 1)")
    parser.add_argument("-C", type=int, default=0, help="number of cell types (default: auto)")
    parser.add_argument("--data_path", type=str, required=True, help="path to data file")
    parser.add_argument("--meta_path", type=str, help="path to metadata file")
    parser.add_argument("--out_path", type=str, required=True, help="path to output file")
    parser.add_argument("--device", type=str, default="cuda:0", help="'cpu', 'cuda', 'cuda:0', ... (default: 'cuda:0')")
    parser.add_argument("--parameters", type=str, help="path to file with parameters (optional)")
    parser.add_argument(
        "--mode", type=str, default="train", help="one of 'train', 'sample', 'freeze' (default: 'train')"
    )

    args = parser.parse_args()

    pyro.clear_param_store()
    torch.set_default_dtype(torch.float32)
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
        X = read_text_matrix(data_path)

    # Overwrite h5ad metadata if another file is specified.
    if meta_path:
        meta = read_meta_from_file(meta_path)

    ctype = meta.ctypes_tensor.to(device)
    batch = meta.batches_tensor.to(device)
    group = meta.groups_tensor.to(device)
    topic = meta.topics_tensor.to(device)
    cmask = meta.ctype_mask_tensor.to(device)
    smask = meta.topic_mask_tensor.to(device)
    ctmap = meta.unique_ctypes

    # Make sure that the total number of topics is no smaller than
    # the number of known (specified) topics.
    if args.K < len(torch.unique(topic)):
        sys.exit("-K is less than the number of existing topics")

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
    cellavi.C = args.C if args.C > 0 else len(ctmap)  # Number of cell types.
    cellavi.B = int(batch.max() + 1)  # Number of batches.
    cellavi.R = int(group.max() + 1)  # Number of groups.
    cellavi.G = int(X.shape[-1])  # Number of genes.

    data_idx = torch.arange(X.shape[0]).to(device)

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

    model = Cellavi(X=X, ddata=ddata, device=device)

    if args.mode == "sample":
        sample = model.resample().cpu()
        torch.save(sample, out_path)
        return
    elif args.mode == "freeze":
        model.freeze("global_base")
        model.freeze("topics_KR")

    # The train data loader is a dummy list of indices.
    train_data_loader = torch.utils.data.DataLoader(
        dataset=data_idx,
        shuffle=True,
        batch_size=cellavi.SUBSMPL,
        collate_fn=CellaviCollator(ddata),
    )

    # The test data loader is the same same dummy list of indices
    # but shuffling is turned off so that cells are processed in
    # the same order as in the input data. We also make the batch
    # size 64 times larger because we just call the amortizer
    # (no gradient updates are performed).
    test_data_loader = torch.utils.data.DataLoader(
        dataset=data_idx,
        shuffle=False,
        batch_size=64 * cellavi.SUBSMPL,
        collate_fn=CellaviCollator(ddata),
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
        # enable_checkpointing=True,
    )

    pl.seed_everything(123)
    trainer.fit(harnessed, train_data_loader)
    trainer.test(harnessed, test_data_loader)

    # Save output to file.
    param_store = pyro.get_param_store().get_state()
    for key, value in param_store["params"].items():
        param_store["params"][key] = value.clone().cpu()
    # Store the cell type map.
    param_store["ctmap"] = ctmap
    torch.save(param_store, out_path)


if __name__ == "__main__":
    main()
