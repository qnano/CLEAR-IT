# clearit/models/composite.py
import torch
import torch.nn as nn

class EncoderClassifier(nn.Module):
    """
    A thin wrapper that takes:
      - an encoder (e.g. ResNetEncoder)
      - a classification head (e.g. MLPHead)
    and defines the CLEAR-IT forward-pass that:
      • treats each input’s C channels as C separate 1-channel images
      • replicates each to 3 channels, runs them all through encoder
      • reshapes and concatenates their embeddings
      • finally applies the classification head
    """

    def __init__(
        self,
        encoder: nn.Module,
        classification_head: nn.Module,
    ):
        super().__init__()
        self.encoder = encoder
        self.classification_head = classification_head

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, C, H, W]
        Returns logits: [B, num_classes]
        """
        B, C, H, W = x.shape

        # 1) unroll channels into batch
        x = x.view(B * C, 1, H, W)

        # 2) fake-RGB triplication so the ResNet’s conv1 (3-in) will accept it
        x = x.repeat(1, 3, 1, 1)               # now [B*C, 3, H, W]

        # 3) run through encoder
        feats = self.encoder(x)                # [B*C, F]

        # 4) reshape back to (B, C, F)
        feats = feats.view(B, C, -1)

        # 5) concatenate along feature dimension -> (B, C*F)
        feats = feats.reshape(B, -1)

        # 6) classification head
        return self.classification_head(feats)
