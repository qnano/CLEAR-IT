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

        # Treat each channel as its own 1-channel image.
        x = x.view(B * C, 1, H, W)

        # Replicate to RGB so the encoder input layer accepts the tensor.
        x = x.repeat(1, 3, 1, 1)               # now [B*C, 3, H, W]

        # Encode each channel view.
        feats = self.encoder(x)                # [B*C, F]

        # Restore batch and channel dimensions.
        feats = feats.view(B, C, -1)

        # Concatenate channel embeddings.
        feats = feats.reshape(B, -1)

        # Run the classification head.
        return self.classification_head(feats)
