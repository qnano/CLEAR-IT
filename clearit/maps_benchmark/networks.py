from __future__ import annotations

"""Minimal MAPS MLP network vendored into CLEAR-IT for public benchmarking.

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

import torch
import torch.nn as nn


class MLP(nn.Module):
    """Feed-forward classifier used by the MAPS cell phenotyping benchmark."""

    def __init__(
        self,
        input_dim: int = 47,
        hidden_dim: int = 512,
        num_classes: int = 12,
        dropout: float = 0.10,
        network_depth: int = 4,
    ) -> None:
        super().__init__()
        layers = [
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=dropout),
        ]
        for _ in range(network_depth - 1):
            layers.extend(
                [
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(p=dropout),
                ]
            )
        self.fc = nn.Sequential(*layers)
        self.classifier = nn.Linear(hidden_dim, num_classes)

    def forward(self, batch: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.fc(batch)
        logits = self.classifier(features)
        probs = torch.softmax(logits, dim=-1)
        return logits, probs
