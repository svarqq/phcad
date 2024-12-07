from collections.abc import Collection
from numbers import Number
from typing import Callable
from functools import partial
import random

import torch
import numpy as np
from torch.utils.data import Dataset


class AugmentableDataset(Dataset):
    """Extension wrapper around Dataset to support online preprocessing of base data
    for model input, including data augmentation"""

    def __init__(
        self,
        labels: str | Collection[Number],
        base_data: torch.Tensor,
        n_samples: int,
        transform: Callable[[torch.Tensor, bool], torch.Tensor],
    ):
        if isinstance(labels, str):
            self._labels = (labels,)
        else:
            self._labels = labels
        self._base_data = base_data
        self._n_samps = n_samples
        self._transform = transform

        self.__trail_idcs = {}
        self.__label_probs = []

    def __len__(self):
        return self._n_samps

    def __getitem__(self, idx):
        # Get transform and base_idx
        if idx >= self._n_samps:
            raise IndexError(
                f"Index must be less than instantiated sample size {self._n_samps}"
            )
        blen = len(self._base_data)  # base length
        if idx < blen:
            transform = self._transform(base=True)
            base_idx = idx
        elif blen <= idx < blen * (self._n_samps // blen):
            transform = self._transform(base=False)
            base_idx = idx % blen
        else:
            transform = self._transform(base=False)
            # Randomly pick base_idx w/o replacement
            if not self.__trail_idcs:
                self.__trail_idcs = random.sample(
                    range(blen), self._n_samps - self._n_samps // blen * blen
                )
            base_idx = self.__trail_idcs[idx % blen]

        if len(self._labels) == 1:
            label = self._labels[0]
        else:
            label = self._labels[base_idx]
        augmented_img = transform(self._base_data[base_idx])
        return augmented_img, torch.ones(augmented_img.shape[-2:])
