# clearit/shap/explainer.py
from typing import Optional, Tuple
import torch
import torch.nn as nn
import shap
import torchvision.models.resnet as tv_resnet
import torch.nn.functional as F

def _set_all_relu_non_inplace(module: nn.Module) -> None:
    for m in module.modules():
        if isinstance(m, nn.ReLU):
            m.inplace = False

def _patch_torchvision_resnet_relu_noninplace() -> None:
    if getattr(tv_resnet, "_clearit_patched_relu", False):
        return
    if hasattr(tv_resnet, "BasicBlock"):
        def _bb_forward(self, x):
            identity = x
            out = self.conv1(x); out = self.bn1(out); out = F.relu(out, inplace=False)
            out = self.conv2(out); out = self.bn2(out)
            if self.downsample is not None: identity = self.downsample(x)
            out = out + identity; out = F.relu(out, inplace=False)
            return out
        tv_resnet.BasicBlock.forward = _bb_forward
    if hasattr(tv_resnet, "Bottleneck"):
        def _bn_forward(self, x):
            identity = x
            out = self.conv1(x); out = self.bn1(out); out = F.relu(out, inplace=False)
            out = self.conv2(x); out = self.bn2(out); out = F.relu(out, inplace=False)
            out = self.conv3(out); out = self.bn3(out)
            if self.downsample is not None: identity = self.downsample(x)
            out = out + identity; out = F.relu(out, inplace=False)
            return out
        tv_resnet.Bottleneck.forward = _bn_forward
    tv_resnet._clearit_patched_relu = True

class _ModelWrapper(nn.Module):
    def __init__(self, model: nn.Module): super().__init__(); self.model = model
    def forward(self, x: torch.Tensor) -> torch.Tensor: return self.model(x)

def prepare_shap_explainer(
    model: nn.Module,
    background_loader,
    device: Optional[torch.device] = None,
    background_strategy: str = "zeros",   # "zeros" or "data"
    background_max_batches: int = 1,
) -> Tuple[shap.DeepExplainer, torch.Tensor, torch.Tensor]:
    device = device or (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))

    sample_img, *_ = next(iter(background_loader))
    sample_in = sample_img.to(device).float()

    if background_strategy == "zeros":
        background = torch.zeros_like(sample_in[:1])
    else:
        imgs = []
        with torch.no_grad():
            for i, (x, *_) in enumerate(background_loader):
                imgs.append(x.float())
                if i + 1 >= max(1, int(background_max_batches)): break
        background = torch.cat(imgs, dim=0).to(device)

    _set_all_relu_non_inplace(model)
    _patch_torchvision_resnet_relu_noninplace()

    wrapped = _ModelWrapper(model).to(device).eval()
    explainer = shap.DeepExplainer(wrapped, background)
    return explainer, background, sample_in
