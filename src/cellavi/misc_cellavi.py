import re
import sys

import torch


def sc_data(data_path):
    """
    Data for single-cell transcriptome, returns a 7-tuple with
       1. list of cell identifiers (arbitrary),
       2. tensor of cell types as integers,
       3. tensor of batches as integers,
       4. tensor of groups as integers,
       5. tensor of labels as integers,
       6. tensor of read counts as float,
       7. tensor of label masks as boolean.
    """

    list_of_identifiers = list()
    list_of_ctypes = list()
    list_of_batches = list()
    list_of_groups = list()
    list_of_labels = list()
    list_of_exprs = list()

    # Helper functions.
    def parse_header(line):
        items = line.split()
        if not re.search(r"^[Gg]roups?", items[3]):
            return 3
        if not re.search(r"^[Ll]abels?", items[4]):
            return 4
        return 5

    def parse(n, row):
        return row[:n], [round(float(x)) for x in row[n:]]

    # Read in data from file.
    with open(data_path) as f:
        first_numeric_field = parse_header(next(f))
        for line in f:
            info, expr = parse(first_numeric_field, line.split())
            list_of_identifiers.append(info[0])
            list_of_ctypes.append(info[1])
            list_of_batches.append(info[2])
            if len(info) >= 4:
                list_of_groups.append(info[3])
            if len(info) >= 5:
                list_of_labels.append(info[4])
            list_of_exprs.append(torch.tensor(expr))

    unique_ctypes = sorted(list(set(list_of_ctypes)))
    list_of_ctype_ids = [unique_ctypes.index(x) for x in list_of_ctypes]
    ctype_tensor = torch.tensor(list_of_ctype_ids)

    unique_batches = sorted(list(set(list_of_batches)))
    list_of_batches_ids = [unique_batches.index(x) for x in list_of_batches]
    batches_tensor = torch.tensor(list_of_batches_ids)

    if list_of_groups:
        unique_groups = sorted(list(set(list_of_groups)))
        list_of_groups_ids = [unique_groups.index(x) for x in list_of_groups]
        groups_tensor = torch.tensor(list_of_groups_ids)
    else:
        groups_tensor = torch.zeros(len(list_of_identifiers)).to(torch.long)

    if list_of_labels:
        unique_labels = sorted(list(set(list_of_labels)))
        if "?" in unique_labels:
            unique_labels.remove("?")
            label_mask = [label != "?" for label in list_of_labels]
        else:
            label_mask = [True] * len(list_of_labels)
        label_mask_tensor = torch.tensor(label_mask, dtype=torch.bool)
        list_of_labels_ids = [unique_labels.index(x) if x in unique_labels else 0 for x in list_of_labels]
        labels_tensor = torch.tensor(list_of_labels_ids)
        labels_tensor[~label_mask_tensor] = 0
    else:
        labels_tensor = torch.zeros(len(list_of_identifiers)).to(torch.long)
        label_mask_tensor = torch.zeros(len(list_of_identifiers)).to(torch.bool)

    expr_tensor = torch.stack(list_of_exprs)

    return (
        list_of_identifiers,
        ctype_tensor,
        batches_tensor,
        groups_tensor,
        labels_tensor,
        expr_tensor,
        label_mask_tensor,
    )


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


def read_info(data_path):
    """
    Data for single-cell transcriptome, returns a 5-tuple with
       1. tensor of cell types as integers,
       2. tensor of batches as integers,
       3. tensor of groups as integers,
       4. tensor of labels as integers,
       5. tensor of cell-type masks as boolean.
    """

    list_of_ctypes = list()
    list_of_batches = list()
    list_of_groups = list()
    list_of_labels = list()

    # Read in data from file.
    with open(data_path) as f:
        for line in f:
            info = line.split()
            list_of_ctypes.append(info[0])
            list_of_batches.append(info[1])
            list_of_groups.append(info[2])
            list_of_labels.append(info[3])

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

    unique_batches = sorted(list(set(list_of_batches)))
    list_of_batches_ids = [unique_batches.index(x) for x in list_of_batches]
    batches_tensor = torch.tensor(list_of_batches_ids)

    unique_groups = sorted(list(set(list_of_groups)))
    list_of_groups_ids = [unique_groups.index(x) for x in list_of_groups]
    groups_tensor = torch.tensor(list_of_groups_ids)

    unique_labels = sorted(list(set(list_of_labels)))
    list_of_label_ids = [unique_labels.index(x) for x in list_of_labels]
    labels_tensor = torch.tensor(list_of_label_ids)

    return (
        ctypes_tensor,
        batches_tensor,
        groups_tensor,
        labels_tensor,
        ctype_mask_tensor,
    )
