# clearit/models/head.py
import torch.nn as nn
from abc import ABC, abstractmethod

class BaseHead(ABC, nn.Module):
    """Abstract base for any classification or projection head."""

    def __init__(self):
        super().__init__()

    @abstractmethod
    def forward(self, x):
        """Given feature vectors, return logits or embeddings."""
        pass

    @abstractmethod
    def get_output_size(self) -> int:
        """Return the dimension of the head’s output."""
        pass


class MLPHead(BaseHead):
    def __init__(self, input_size, num_classes, dropout=0.3, head_layers=None):
        super().__init__()
        head_layers = head_layers or []
        layers = [nn.Dropout(dropout),
                  nn.Linear(input_size, head_layers[0] if head_layers else num_classes)]
        # hidden layers
        for i in range(len(head_layers)-1):
            layers += [nn.ReLU(), nn.Dropout(dropout),
                       nn.Linear(head_layers[i], head_layers[i+1])]
        # final classifier
        if head_layers:
            layers += [nn.ReLU(), nn.Dropout(dropout),
                       nn.Linear(head_layers[-1], num_classes)]
        self.head = nn.Sequential(*layers)

    def forward(self, x):
        return self.head(x)

    def load_state_dict(self, state_dict, strict: bool = True):
        # detect old‐format keys (no "head." prefix)
        if not any(k.startswith('head.') for k in state_dict):
            state_dict = {f"head.{k}": v for k, v in state_dict.items()}
        super().load_state_dict(state_dict, strict=strict)

    def get_output_size(self):
        # final Linear’s out_features
        final_lin = [m for m in self.head if isinstance(m, nn.Linear)][-1]
        return final_lin.out_features
