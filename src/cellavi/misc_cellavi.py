import re
from typing import Any, Dict, List, Set, Tuple

import numpy as np
import pyro
import scanpy
import torch
from scipy.io import mmread
from scipy.sparse import csr_matrix

LOCAL_PARAMS = ["z_i_loc", "z_i_scale", "c_indx_probs", "log_theta_i_loc", "log_theta_i_scale"]


class CellaviMetadata:
    """
    Metadata for single-cell transcriptome:
       1. tensor of cell types as integers,
       2. tensor of batches as integers,
       3. tensor of groups as integers,
       4. tensor of states as integers,
       5. tensor of cell-type masks as boolean.
       6. tensor of state masks as boolean,
       7. sorted list of unique cell types.
    """

    def __init__(
        self,
        specified_fields: Set[str],
        list_of_ctypes: List[float],
        list_of_batches: List[float],
        list_of_groups: List[float],
        list_of_states: List[float],
    ):
        if "ctype" not in specified_fields:
            # Cell types are not provided. Set the cell type of every
            # cell to 0 and mark it as masked.
            ctypes_tensor = torch.tensor(list_of_ctypes)
            ctype_mask_tensor = torch.zeros_like(ctypes_tensor).bool()
            unique_ctypes = [None]
        else:
            # At last some cell types are provided.
            unique_ctypes = sorted(list(set(list_of_ctypes)))
            if "?" in unique_ctypes:
                unique_ctypes.remove("?")
                ctype_mask = [ctype != "?" for ctype in list_of_ctypes]
            else:
                ctype_mask = [True] * len(list_of_ctypes)
            ctype_mask_tensor = torch.tensor(ctype_mask, dtype=torch.bool)
            list_of_ctype_ids = [unique_ctypes.index(x) if x in unique_ctypes else 0 for x in list_of_ctypes]
            ctypes_tensor = torch.tensor(list_of_ctype_ids)
            ctypes_tensor[~ctype_mask_tensor] = 0

        if "state" not in specified_fields:
            states_tensor = torch.tensor(list_of_states)
            state_mask_tensor = torch.zeros_like(states_tensor).bool()
        else:
            unique_states = sorted(list(set(list_of_states)))
            if "?" in unique_states:
                unique_states.remove("?")
                state_mask = [state != "?" for state in list_of_states]
            else:
                state_mask = [True] * len(list_of_states)
            state_mask_tensor = torch.tensor(state_mask, dtype=torch.bool)
            list_of_state_ids = [unique_states.index(x) if x in unique_states else 0 for x in list_of_states]
            states_tensor = torch.tensor(list_of_state_ids)
            states_tensor[~state_mask_tensor] = 0

        unique_batches = sorted(list(set(list_of_batches)))
        list_of_batch_ids = [unique_batches.index(x) for x in list_of_batches]
        batches_tensor = torch.tensor(list_of_batch_ids)

        unique_groups = sorted(list(set(list_of_groups)))
        list_of_group_ids = [unique_groups.index(x) for x in list_of_groups]
        groups_tensor = torch.tensor(list_of_group_ids)

        self.ctypes_tensor = ctypes_tensor
        self.batches_tensor = batches_tensor
        self.groups_tensor = groups_tensor
        self.states_tensor = states_tensor
        self.ctype_mask_tensor = ctype_mask_tensor
        self.state_mask_tensor = state_mask_tensor
        self.unique_ctypes = unique_ctypes


def read_meta_from_file(path):
    list_of_ctypes: List[str] = list()
    list_of_batches: List[str] = list()
    list_of_groups: List[str] = list()
    list_of_states: List[str] = list()

    # Read in data from file.
    ncells = 0
    with open(path) as f:
        header = next(f)
        indices = get_field_indices(header.split())
        for line in f:
            ncells += 1
            info = line.split() + [0]  # Add default value at position -1.
            list_of_ctypes.append(info[indices.get("ctype", -1)])
            list_of_batches.append(info[indices.get("batch", -1)])
            list_of_groups.append(info[indices.get("group", -1)])
            list_of_states.append(info[indices.get("state", -1)])

    return CellaviMetadata(indices.keys(), list_of_ctypes, list_of_batches, list_of_groups, list_of_states)


def read_h5ad(filename: str, metadata_file: str = None):
    adata = scanpy.read_h5ad(filename)
    X = adata.X.astype(np.float32)

    adata.obs["zero"] = 0  # Add default value at position -1.
    indices = get_field_indices(adata.obs.columns.tolist())
    list_of_ctypes = adata.obs.iloc[:, indices.get("ctype", -1)].tolist()
    list_of_batches = adata.obs.iloc[:, indices.get("batch", -1)].tolist()
    list_of_groups = adata.obs.iloc[:, indices.get("group", -1)].tolist()
    list_of_states = adata.obs.iloc[:, indices.get("state", -1)].tolist()

    metadata = CellaviMetadata(indices.keys(), list_of_ctypes, list_of_batches, list_of_groups, list_of_states)

    return X, metadata


def load_parameters(path: str, device: str, remove_locals: bool = True) -> List[str]:
    loaded_store: Dict[Any] = torch.load(path)
    ctmap: List[str] = loaded_store.pop("ctmap")
    for key in loaded_store["params"]:
        loaded_store["params"][key] = loaded_store["params"][key].to(device)
    pyro.get_param_store().set_state(loaded_store)
    if remove_locals:
        store = pyro.get_param_store()
        for key in LOCAL_PARAMS:
            if key in store:
                del store[key]
    return ctmap


def update_ctmap(ctmap: List[str], loaded_ctmap: List[str], ctype: torch.tensor) -> Tuple[List[str], torch.tensor]:
    # Check if there is anything to do.
    if len(ctmap) == len(loaded_ctmap) and all([a == b for a, b in zip(ctmap, loaded_ctmap)]):
        # Same cell types, indexing is up to date.
        return ctmap, ctype
    all_types: List[str] = sorted(set(ctmap) | set(loaded_ctmap))
    idxmap_new: List[int] = [all_types.index(x) for x in ctmap]
    # Update `ctype`.
    for i in range(len(ctmap)):
        ctype[ctype == i] = idxmap_new[i]
    # Update parameter "base_0" if present.
    if "base_0" in pyro.get_param_store():
        old_base_0 = pyro.param("base_0")
        C = len(all_types)
        G = old_base_0.shape[-1]
        new_base_0 = torch.zeros(C, G).to(old_base_0.device)
        idxmap_loaded: List[int] = [all_types.index(x) for x in loaded_ctmap]
        for i in range(len(loaded_ctmap)):
            new_base_0[idxmap_loaded[i]] = old_base_0[i]
        pyro.get_param_store()["base_0"] = new_base_0
    return all_types, ctype


# Helper function.
def get_field_indices(items):
    indices = {}
    # Search cell type.
    matches = [re.search(r"^[Cc]ell_types?$", x) and True for x in items]
    if any(matches):
        indices["ctype"] = matches.index(True)
    # Search batch.
    matches = [re.search(r"^[Bb]atch(?:es)?$", x) and True for x in items]
    if any(matches):
        indices["batch"] = matches.index(True)
    # Search group.
    matches = [re.search(r"^[Gg]roups?$", x) and True for x in items]
    if any(matches):
        indices["group"] = matches.index(True)
    # Search states.
    matches = [re.search(r"^[Ss]tates?$", x) and True for x in items]
    if any(matches):
        indices["state"] = matches.index(True)
    return indices


def read_text_matrix(path, header_is_present=True):
    with open(path) as f:
        if header_is_present:
            _ = next(f)
        arrays = list()
        for line in f:
            arrays.append(np.array([float(x) for x in line.split()], dtype=np.float32))
    return csr_matrix(np.vstack(arrays))


def read_mtx(path):
    return mmread(path).tocsr()


#    list_of_sparse_tensors = list()
#    for path in paths:
#        sys.stderr.write(f"{path}\n")
#        with open(path) as f:
#            _ = next(f)
#            nrow, ncol, nnz = (int(x) for x in next(f).split())
#            row_indices = list()
#            col_indices = list()
#            values = list()
#            for line in f:
#                r, c, v = (int(x) for x in line.split())
#                row_indices.append(r - 1)
#                col_indices.append(c - 1)
#                values.append(v)
#            sparse = torch.sparse_coo_tensor(
#                indices=[row_indices, col_indices], values=values, size=(nrow, ncol), dtype=torch.int16
#            )
#            list_of_sparse_tensors.append(sparse)
#    return torch.vstack(list_of_sparse_tensors)
