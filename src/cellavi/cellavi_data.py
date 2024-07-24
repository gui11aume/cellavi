import math
from types import SimpleNamespace
from typing import Optional

import torch
import torch.nn.functional as F
from scipy.sparse import csr_matrix

SUBSMPL = 512

# TODO DEFINE K, B, C AND R.


def denslice(array: Optional[csr_matrix], idx: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
    if idx is None:
        # Without index, return the whole array.
        return torch.tensor(array.todense())
    if array is None:
        return None
    dense_array = array[idx.cpu(), :].todense()
    return torch.tensor(dense_array)


class CellaviData:
    def __init__(
        self,
        x: csr_matrix,
        ctype: torch.Tensor,
        batch: torch.Tensor,
        group: torch.Tensor,
        topic: torch.Tensor,
        cmask: torch.Tensor,
        smask: torch.Tensor,
        chunk_size: int = SUBSMPL,
        K: int = 1,
        C: int = 1,
        B: int = 1,
        R: int = 1,
    ):
        self.x: csr_matrix = x
        self.ncells: int = x.shape[0]
        self.ctype: torch.Tensor = ctype.long()
        self.batch: torch.Tensor = batch.long()
        self.group: torch.Tensor = group.long()
        self.topic: torch.Tensor = topic.long()
        self.cmask: torch.Tensor = cmask.bool()
        self.smask: torch.Tensor = smask.bool()
        self.chunk_size: int = chunk_size
        self.C: int = C
        self.B: int = B
        self.R: int = R
        self.K: int = K

        self.validate()

        self.one_hot_ctype = F.one_hot(self.ctype, num_classes=self.C).float()
        self.one_hot_batch = F.one_hot(self.batch, num_classes=self.B).float()
        self.one_hot_group = F.one_hot(self.batch, num_classes=self.R).float()
        self.one_hot_topic = F.one_hot(self.topic, num_classes=self.K).float()
        # Format observed labels. Create one-hot encoding with label smoothing.
        # Assign values so that the probability of the provided label is 0.95.
        a = 1.472219 + 0.5 * math.log(self.K - 1)
        self.stopic = 2 * a * self.one_hot_topic - a if self.K > 1 else self.one_hot_topic

    def validate(self):
        # Make sure that all tensors have the same length.
        assert self.ctype.shape == torch.Size([self.ncells])
        assert self.batch.shape == torch.Size([self.ncells])
        assert self.group.shape == torch.Size([self.ncells])
        assert self.topic.shape == torch.Size([self.ncells])
        assert self.cmask.shape == torch.Size([self.ncells])
        assert self.smask.shape == torch.Size([self.ncells])

    def iterate_by_chunk(self, chunk_size):
        for i in range(0, self.ncells, chunk_size):
            idx_i = torch.arange(self.ncells)[i : i + chunk_size]
            # Create a placeholder object to store the data.
            data_i = SimpleNamespace()
            data_i.x = denslice(self.x, idx_i)
            data_i.ctype = self.ctype[idx_i]
            data_i.batch = self.batch[idx_i]
            data_i.group = self.group[idx_i]
            data_i.topic = self.topic[idx_i]
            data_i.cmask = self.cmask[idx_i]
            data_i.smask = self.smask[idx_i]
            data_i.stopic = self.stopic[idx_i]
            data_i.one_hot_ctype = self.one_hot_ctype[idx_i, :]
            data_i.one_hot_batch = self.one_hot_batch[idx_i, :]
            data_i.one_hot_group = self.one_hot_group[idx_i, :]
            data_i.one_hot_topic = self.one_hot_topic[idx_i, :]

            yield data_i

    def __getitem__(self, key):
        if isinstance(key, int):
            if key < 0:  # Support negative indexing
                key += self.ncells
            idx_i = torch.tensor([key])
        elif isinstance(key, slice):
            start, stop, step = key.indices(self.ncells)
            idx_i = torch.arange(start, stop, step)
        elif isinstance(key, list):
            idx_i = torch.tensor(key)
        elif isinstance(key, torch.Tensor):
            idx_i = key
        else:
            raise TypeError("Invalid argument type.")

        # Create a placeholder object to store the data.
        data_i = SimpleNamespace()
        data_i.idx_i = idx_i
        data_i.x = denslice(self.x, idx_i)
        data_i.ctype = self.ctype[idx_i]
        data_i.batch = self.batch[idx_i]
        data_i.group = self.group[idx_i]
        data_i.topic = self.topic[idx_i]
        data_i.cmask = self.cmask[idx_i]
        data_i.smask = self.smask[idx_i]
        data_i.stopic = self.stopic[idx_i]
        data_i.one_hot_ctype = self.one_hot_ctype[idx_i, :]
        data_i.one_hot_batch = self.one_hot_batch[idx_i, :]
        data_i.one_hot_group = self.one_hot_group[idx_i, :]
        data_i.one_hot_topic = self.one_hot_topic[idx_i, :]

        return data_i


class CellaviExpressionData:
    def __init__(self, x: csr_matrix, chunk_size: int = SUBSMPL):
        self.x: csr_matrix = x
        self.ncells: int = x.shape[0]
        self.chunk_size: int = chunk_size

    def __iter__(self):
        self.current = 0
        return self

    def __next__(self):
        if self.current < self.ncells:
            idx_i = torch.arange(self.ncells)[self.current : self.current + self.chunk_size]
            self.current += self.chunk_size
            return denslice(self.x, idx_i)
        else:
            raise StopIteration

    def __getitem__(self, key):
        if isinstance(key, int):
            if key < 0:  # Support negative indexing
                key += self.ncells
            idx_i = torch.tensor([key])
        elif isinstance(key, slice):
            start, stop, step = key.indices(self.ncells)
            idx_i = torch.arange(start, stop, step)
        elif isinstance(key, list):
            idx_i = torch.tensor(key)
        elif isinstance(key, torch.Tensor):
            idx_i = key
        else:
            raise TypeError("Invalid argument type.")
        return denslice(self.x, idx_i)


class CellaviCollator:
    def __init__(self, data: CellaviData):
        self.data = data

    def __call__(self, examples):
        # Here, `examples` is a list of singletons. The subset
        # of indices must be kept in sorted order, otherwise
        # Pyro loses track of the associated parameters.
        unsorted_idx = torch.stack(examples)
        sorted_idx = unsorted_idx.sort().values
        return self.data[sorted_idx]
