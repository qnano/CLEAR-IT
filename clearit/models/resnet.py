from typing import List

import torch.nn as nn
from torchvision import models

from .encoder import BaseEncoder


class ResNetEncoder(BaseEncoder):
    def __init__(
        self,
        encoder_name: str = "resnet18",
        encoder_features: int = 512,
        mlp_layers: List[int] = None,
        mlp_features: int = 128,
        in_channels: int = 3,
    ):
        super().__init__()
        self.encoder_name = encoder_name
        self.encoder_features = encoder_features
        self.mlp_layers = mlp_layers or []
        self.mlp_features = mlp_features

        # Load the backbone and optionally adapt conv1 for single-channel input.
        backbone = self._select_resnet(encoder_name, pretrained=False)
        if in_channels != 3:
            orig = backbone.conv1
            backbone.conv1 = nn.Conv2d(
                in_channels,
                orig.out_channels,
                kernel_size=orig.kernel_size,
                stride=orig.stride,
                padding=orig.padding,
                bias=(orig.bias is not None),
            )

        # Replace the final fc layer so the backbone emits encoder_features.
        num_ftrs = backbone.fc.in_features
        backbone.fc = nn.Linear(num_ftrs, encoder_features)

        self.main_backbone = backbone

        # Build the SimCLR projection MLP from the configured hidden dimensions.
        modules = []
        prev_dim = encoder_features
        for hidden_dim in self.mlp_layers:
            modules += [
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(inplace=True),
            ]
            prev_dim = hidden_dim
        if self.mlp_layers:
            modules += [
                nn.Linear(prev_dim, self.mlp_features),
                nn.BatchNorm1d(self.mlp_features),
            ]
        self.mlp = nn.Sequential(*modules) if modules else nn.Identity()

    def _select_resnet(self, encoder_name, pretrained=True):
        model_funcs = {
            "resnet18": models.resnet18,
            "resnet34": models.resnet34,
            "resnet50": models.resnet50,
            "resnet101": models.resnet101,
            "resnet152": models.resnet152,
        }
        return model_funcs.get(encoder_name, models.resnet18)(pretrained=pretrained)

    def forward(self, x):
        # x: [B*C, 3, H, W] after channel expansion in EncoderClassifier.
        x = self.main_backbone(x)
        x = self.mlp(x)
        return x

    def get_feature_size(self, proj_layers: int) -> int:
        """
        Return the encoder feature dimension for a given number of projection layers.

        - `proj_layers == 0`: backbone output.
        - `0 < proj_layers < len(self.mlp_layers)`: hidden MLP path still returns
          `encoder_features`.
        - `proj_layers == len(self.mlp_layers) > 0`: final projection output.
        """
        if proj_layers == 0:
            return self.encoder_features

        if proj_layers < len(self.mlp_layers):
            return self.encoder_features

        if proj_layers == len(self.mlp_layers):
            return self.mlp_features

        raise ValueError(
            f"proj_layers ({proj_layers}) exceeds configured mlp_layers ({len(self.mlp_layers)})."
        )
