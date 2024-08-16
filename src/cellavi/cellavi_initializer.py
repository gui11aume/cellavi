import pyro
import torch
import torch.nn.functional as F
from cellavi_data import CellaviData

NSUPP_CELLS = 65_536
N_BATCHES = 256
SUBSMPL = 512


# def n_choose_k__01(n: int, k: int):
#     rnd__01 = torch.zeros(n, dtype=torch.bool)
#     rnd__01[:k] = True
#     return rnd__01[torch.randperm(n)]


def gather_profiles(data: CellaviData):
    # Add 1/2 pseudo-count.
    profiles = 0.5 * torch.ones(data.B, data.C, data.x.shape[-1])
    for chunk_i in data.iterate_by_chunk(SUBSMPL):
        x_i = chunk_i.x
        x_i[~chunk_i.cmask] = 0.0  # Remove cells without known type.
        profiles += torch.einsum("iB,iG,iC->BCG", chunk_i.one_hot_batch, x_i, chunk_i.one_hot_ctype)
    return profiles


def initialize_parameters(data: CellaviData):
    # Collect parameters directly to Pyro's parameter store.
    param_store = pyro.get_param_store()

    profiles = gather_profiles(data)
    baseline = profiles.sum(dim=tuple(range(profiles.dim() - 1)))

    global_base = torch.log(F.normalize(baseline, p=1, dim=-1))
    global_base -= global_base.mean()
    param_store["autonormal.locs.global_base"] = global_base

    # Use the pseudo-inverse to estimate the batch and cell type effects.
    # This is a way to break down the contributions of each term to
    # the average of the observed profiles.
    log_prof = torch.log(F.normalize(profiles, p=1, dim=-1))
    design_matrix = torch.zeros(data.B * data.C, data.B + data.C)
    for i in range(data.B):
        for j in range(data.C):
            design_matrix[i * data.C + j, i] = 1
            design_matrix[i * data.C + j, data.B + j] = 1
    pseudo_inv = torch.linalg.pinv(design_matrix)
    log_prof_ = log_prof.view(data.B * data.C, -1)
    concat = torch.einsum("ij,jG->iG", pseudo_inv, log_prof_)

    batch_fx = concat[: data.B] - concat[: data.B].mean(dim=0, keepdim=True)
    ctype_fx = concat[data.B :] - concat[data.B :].mean(dim=0, keepdim=True)

    param_store["autonormal.locs.batch_fx"] = batch_fx
    param_store["autonormal.locs.ctype_fx"] = ctype_fx
