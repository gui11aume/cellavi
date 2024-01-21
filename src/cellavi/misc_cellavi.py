import re
import sys

import torch


# Helper function.
def get_field_indices(header):
    indices = {}
    items = header.split()
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
    # Search modules.
    matches = [re.search(r"^[Mm]odules?$", x) and True for x in items]
    if any(matches):
        indices["module"] = matches.index(True)
    return indices


def read_sparse_matrix(paths):
    list_of_sparse_tensors = list()
    for path in paths:
        sys.stderr.write(f"{path}\n")
        with open(path) as f:
            _ = next(f)
            nrow, ncol, nnz = (int(x) for x in next(f).split())
            row_indices = list()
            col_indices = list()
            values = list()
            for line in f:
                r, c, v = (int(x) for x in line.split())
                row_indices.append(r - 1)
                col_indices.append(c - 1)
                values.append(v)
            sparse = torch.sparse_coo_tensor(
                indices=[row_indices, col_indices], values=values, size=(nrow, ncol), dtype=torch.int16
            )
            list_of_sparse_tensors.append(sparse)
    return torch.vstack(list_of_sparse_tensors)


def read_info_from_file(path):
    """
    Data for single-cell transcriptome, returns a 6-tuple with
       1. tensor of cell types as integers,
       2. tensor of batches as integers,
       3. tensor of groups as integers,
       4. tensor of modules as integers,
       5. tensor of cell-type masks as boolean.
       6. tensor of module masks as boolean.
    """

    list_of_ctypes = list()
    list_of_batches = list()
    list_of_groups = list()
    list_of_modules = list()

    # Read in data from file.
    with open(path) as f:
        header = next(f)
        indices = get_field_indices(header)
        for line in f:
            info = line.split() + [0]  # Add default value at position -1.
            list_of_ctypes.append(info[indices.get("ctype", -1)])
            list_of_batches.append(info[indices.get("batch", -1)])
            list_of_groups.append(info[indices.get("group", -1)])
            list_of_modules.append(info[indices.get("module", -1)])

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

    unique_modules = sorted(list(set(list_of_modules)))
    if "?" in unique_modules:
        unique_modules.remove("?")
        module_mask = [module != "?" for module in list_of_modules]
    else:
        module_mask = [True] * len(list_of_modules)
    module_mask_tensor = torch.tensor(module_mask, dtype=torch.bool)
    list_of_module_ids = [unique_modules.index(x) if x in unique_modules else 0 for x in list_of_modules]
    modules_tensor = torch.tensor(list_of_module_ids)
    modules_tensor[~module_mask_tensor] = 0

    unique_batches = sorted(list(set(list_of_batches)))
    list_of_batch_ids = [unique_batches.index(x) for x in list_of_batches]
    batches_tensor = torch.tensor(list_of_batch_ids)

    unique_groups = sorted(list(set(list_of_groups)))
    list_of_group_ids = [unique_groups.index(x) for x in list_of_groups]
    groups_tensor = torch.tensor(list_of_group_ids)

    return (
        ctypes_tensor,
        batches_tensor,
        groups_tensor,
        modules_tensor,
        ctype_mask_tensor,
        module_mask_tensor,
    )
