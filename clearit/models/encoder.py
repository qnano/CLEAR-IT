# clearit/models/encoder.py
from abc import ABC, abstractmethod
import torch.nn as nn

class BaseEncoder(ABC, nn.Module):
    """Abstract base for any encoder you can pretrain."""

    def __init__(self):
        super().__init__()

    @abstractmethod
    def forward(self, x):
        """Given a batch of images, return feature embeddings."""
        pass

    @abstractmethod
    def get_feature_size(self) -> int:
        """Return the dimension of the output embeddings."""
        pass
