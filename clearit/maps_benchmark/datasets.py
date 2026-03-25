from __future__ import annotations

"""Minimal MAPS dataset wrapper vendored into CLEAR-IT for public benchmarking.

This file is adapted from the MAPS project:
https://github.com/mahmoodlab/MAPS

Original MAPS paper:
Shaban, M., Bai, Y., Qiu, H. et al. MAPS: pathologist-level cell type
annotation from tissue images through machine learning. Nat Commun 15, 28
(2024). https://doi.org/10.1038/s41467-023-44188-w

The MAPS repository is distributed under Apache 2.0 with Commons Clause for
non-commercial academic use. See ``clearit/maps_benchmark/THIRD_PARTY_NOTICES.md``
for attribution and license details for the vendored MAPS-derived components.
"""

import numpy as np
from torch.utils.data import DataLoader, Dataset, SequentialSampler, WeightedRandomSampler


class CellExpressionDataset(Dataset):
    """Numpy-backed dataset with optional per-feature normalization."""

    def __init__(
        self,
        expressions: np.ndarray,
        labels: np.ndarray | None = None,
        *,
        is_train: bool = True,
        mean: np.ndarray | None = None,
        std: np.ndarray | None = None,
        norm: bool = True,
    ) -> None:
        self.x = expressions
        self.y = labels
        self.is_train = is_train
        self.mean = mean
        self.std = std
        self.norm = norm

        if norm:
            if self.is_train:
                if self.mean is None or self.std is None:
                    self.mean = np.mean(self.x, axis=0)
                    self.std = np.std(self.x, axis=0) + 1e-8
                self.x = (self.x - self.mean) / self.std
            elif self.mean is not None and self.std is not None:
                self.x = (self.x - self.mean) / self.std

    def __len__(self) -> int:
        return int(self.x.shape[0])

    def __getitem__(self, idx: int) -> tuple[np.ndarray, int]:
        feature = self.x[idx]
        gt = int(self.y[idx]) if self.y is not None else -1
        return feature, gt

    @staticmethod
    def get_data_loader(
        dataset: "CellExpressionDataset",
        *,
        batch_size: int = 4,
        is_train: bool = False,
        num_workers: int = 4,
    ) -> DataLoader:
        if is_train:
            labels = dataset.y.tolist()
            n = float(len(dataset))
            weight = [0.0] * len(dataset)
            unique_labels = np.unique(labels).tolist()
            weight_per_class = [n / labels.count(c) for c in unique_labels]
            for idx, label in enumerate(labels):
                weight[idx] = weight_per_class[unique_labels.index(label)]
            sampler = WeightedRandomSampler(weight, len(weight))
        else:
            sampler = SequentialSampler(dataset)

        return DataLoader(
            dataset,
            batch_size=batch_size,
            sampler=sampler,
            drop_last=False,
            num_workers=num_workers,
        )
