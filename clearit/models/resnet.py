# clearit/models/resnet.py
from .encoder import BaseEncoder
from torchvision import models
import torch.nn as nn
from typing import List

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
        self.encoder_name     = encoder_name
        self.encoder_features = encoder_features
        self.mlp_layers       = mlp_layers or []    # default to empty list
        self.mlp_features     = mlp_features

        # 1) Load backbone with its original 3-channel conv1
        backbone = self._select_resnet(encoder_name, pretrained=False)

        # 2) Replace final fc to output `encoder_features`
        num_ftrs = backbone.fc.in_features
        backbone.fc = nn.Linear(num_ftrs, encoder_features)

        self.main_backbone = backbone

        # 3) Build SimCLR projection MLP from the list of hidden dims
        modules = []
        prev_dim = encoder_features
        # for each hidden layer size in mlp_layers
        for hidden_dim in self.mlp_layers:
            modules += [
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(inplace=True),
            ]
            prev_dim = hidden_dim
        # final projection to mlp_features
        if self.mlp_layers:
            modules += [
                nn.Linear(prev_dim, self.mlp_features),
                nn.BatchNorm1d(self.mlp_features),
            ]
        self.mlp = nn.Sequential(*modules) if modules else nn.Identity()

    def _select_resnet(self, encoder_name, pretrained=True):
        model_funcs = {
            "resnet18":  models.resnet18,
            "resnet34":  models.resnet34,
            "resnet50":  models.resnet50,
            "resnet101": models.resnet101,
            "resnet152": models.resnet152,
        }
        return model_funcs.get(encoder_name, models.resnet18)(pretrained=pretrained)

    def forward(self, x):
        # x: [B*C, 3, H, W] in your composite
        x = self.main_backbone(x)
        x = self.mlp(x)
        return x

    def get_feature_size(self, proj_layers: int) -> int:
        """
        Return the number of features coming out of the encoder,
        honoring proj_layers = how many SimCLR-MLP layers to include:
          - If proj_layers == 0 ➞ no MLP at all ⇒ backbone output dim
          - If 0 < proj_layers < len(self.mlp_layers) ➞ backbone+MLP output
          - If proj_layers == len(self.mlp_layers) > 0 ➞ projection output dim
        """
        # 1) never use the projection head when proj_layers == 0
        if proj_layers == 0:
            return self.encoder_features

        # 2) fewer than all hidden layers ⇒ still un-projected features
        if proj_layers < len(self.mlp_layers):
            return self.encoder_features

        # 3) exactly all hidden layers ⇒ take the projection output
        if proj_layers == len(self.mlp_layers):
            return self.mlp_features

        # 4) anything else is invalid
        raise ValueError(
            f"proj_layers ({proj_layers}) exceeds configured mlp_layers ({len(self.mlp_layers)})."
        )


# clearit/models/resnet.py

# from typing import List, Optional
# from .encoder import BaseEncoder
# from torchvision import models
# import torch.nn as nn

# class ResNetEncoder(BaseEncoder):
#     def __init__(
#         self,
#         encoder_name: str = "resnet18",
#         encoder_features: int = 512,
#         mlp_layers: Optional[List[int]] = None,
#         mlp_features: int = 128,
#     ):
#         super().__init__()
#         self.encoder_name     = encoder_name
#         self.encoder_features = encoder_features
#         self.mlp_layers       = mlp_layers or []   # ensure it’s a list
#         self.mlp_features     = mlp_features

#         # 1) load the backbone
#         backbone = self._select_resnet(encoder_name, pretrained=False)

#         # 2) replace its first conv to match in_channels
#         orig = backbone.conv1
#         backbone.conv1 = nn.Conv2d(
#             in_channels,
#             orig.out_channels,
#             kernel_size=orig.kernel_size,
#             stride=orig.stride,
#             padding=orig.padding,
#             bias=(orig.bias is not None),
#         )

#         # 3) replace the final fc layer so it outputs encoder_features dims
#         num_ftrs = backbone.fc.in_features
#         backbone.fc = nn.Linear(num_ftrs, encoder_features)

#         self.main_backbone = backbone

#         # 4) build SimCLR projection head from the list of hidden dims
#         modules = []
#         in_dim = self.encoder_features
#         for out_dim in self.mlp_layers:
#             modules.append(nn.Linear(in_dim, out_dim))
#             modules.append(nn.BatchNorm1d(out_dim))
#             modules.append(nn.ReLU(inplace=True))
#             in_dim = out_dim
#         self.mlp = nn.Sequential(*modules) if modules else nn.Identity()

#     def _select_resnet(self, encoder_name, pretrained=True):
#         model_funcs = {
#             "resnet18":  models.resnet18,
#             "resnet34":  models.resnet34,
#             "resnet50":  models.resnet50,
#             "resnet101": models.resnet101,
#             "resnet152": models.resnet152,
#         }
#         return model_funcs.get(encoder_name, models.resnet18)(pretrained=pretrained)

#     def forward(self, x):
#         x = self.main_backbone(x)
#         x = self.mlp(x)
#         return x

#     def get_feature_size(self, proj_layers: int) -> int:
#         """
#         Return the dimension of the encoder’s output before the supervised head.
#         proj_layers is how many of the projection layers to use.
#         """
#         if proj_layers == 0:
#             return self.encoder_features
#         if 1 <= proj_layers <= len(self.mlp_layers):
#             # the proj_layers-th layer outputs mlp_layers[proj_layers-1]
#             return self.mlp_layers[proj_layers-1]
#         raise ValueError(f"proj_layers ({proj_layers}) exceeds mlp_layers ({len(self.mlp_layers)}).")
