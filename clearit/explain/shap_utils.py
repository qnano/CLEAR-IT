# clearit/explain/shap_utils.py
from typing import List, Optional, Sequence, Tuple
import math
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.data._utils.collate import default_collate

# SHAP (install: pip install shap>=0.42.0)
import shap

# Optional smoothing backend(s)
try:
    from scipy.ndimage import gaussian_filter as _gaussian_filter
    _HAS_SCIPY = True
except Exception:
    _HAS_SCIPY = False

import matplotlib.pyplot as plt
from matplotlib import gridspec

from clearit.inference.pipeline import load_encoder_head
from clearit.data.classification.dataset import ClassificationCropDataset

from pathlib import Path
import yaml
from clearit.config import MODELS_DIR, OUTPUTS_DIR


# ---------- Utilities ----------

def _collate_skip_none(batch):
    """Collate that drops None samples rather than crashing."""
    batch = [b for b in batch if b is not None]
    if not batch:
        raise RuntimeError("All samples in batch were None/invalid.")
    return default_collate(batch)


def _set_all_relu_non_inplace(module: nn.Module) -> None:
    """
    Ensure every nn.ReLU in a module tree is non-inplace.
    This prevents inplace ops from interfering with SHAP's gradient hooks.
    """
    for m in module.modules():
        if isinstance(m, nn.ReLU):
            m.inplace = False


def _patch_torchvision_resnet_relu_noninplace() -> None:
    """
    Torchvision BasicBlock/Bottleneck use F.relu(..., inplace=True) in forward.
    Patch them to use inplace=False to keep SHAP stable.
    Safe to call multiple times.
    """
    import torchvision.models.resnet as tv_resnet
    import torch.nn.functional as F

    if hasattr(tv_resnet, "_clearit_patched_relu"):
        return  # idempotent

    # BasicBlock
    if hasattr(tv_resnet, "BasicBlock"):
        _orig_bb = tv_resnet.BasicBlock.forward

        def _bb_forward(self, x):
            identity = x
            out = self.conv1(x)
            out = self.bn1(out)
            out = F.relu(out, inplace=False)
            out = self.conv2(out)
            out = self.bn2(out)
            if self.downsample is not None:
                identity = self.downsample(x)
            out = out + identity
            out = F.relu(out, inplace=False)
            return out

        tv_resnet.BasicBlock.forward = _bb_forward

    # Bottleneck
    if hasattr(tv_resnet, "Bottleneck"):
        _orig_bn = tv_resnet.Bottleneck.forward

        def _bn_forward(self, x):
            identity = x
            out = self.conv1(x); out = self.bn1(out); out = F.relu(out, inplace=False)
            out = self.conv2(out); out = self.bn2(out); out = F.relu(out, inplace=False)
            out = self.conv3(out); out = self.bn3(out)
            if self.downsample is not None:
                identity = self.downsample(x)
            out = out + identity
            out = F.relu(out, inplace=False)
            return out

        tv_resnet.Bottleneck.forward = _bn_forward

    tv_resnet._clearit_patched_relu = True


# ---------- Model loading ----------

def load_model(
    encoder_id: str,
    head_id: str,
    device: Optional[torch.device] = None,
) -> nn.Module:
    """
    Load the composite Encoder+Head model in eval() mode.

    Input  : [B, C, H, W]
    Output : logits [B, K]
    """
    device = device or (torch.device("cuda") if torch.cuda.is_available()
                        else torch.device("cpu"))
    model = load_encoder_head(encoder_id, head_id, device=device)
    model.eval()
    return model

def load_model_from_test(test_id: str, device: torch.device = None):
    """
    Resolve head_id and encoder_id via tests/<test_id>/conf_test.yaml and
    return (model, test_cfg, head_cfg). Model is eval() and on `device`.
    """
    device = device or (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))

    test_dir = Path(OUTPUTS_DIR) / "tests" / test_id
    test_cfg_path = test_dir / "conf_test.yaml"
    if not test_cfg_path.exists():
        raise FileNotFoundError(f"conf_test.yaml not found for test '{test_id}' at {test_cfg_path}")

    test_cfg = yaml.safe_load(test_cfg_path.read_text())
    head_id = test_cfg["head_id"]
    encoder_id = test_cfg.get("encoder_id")

    # If encoder_id is omitted in conf_test.yaml, derive it from the head config.
    head_cfg_path = Path(MODELS_DIR) / "heads" / head_id / "conf_head.yaml"
    if not head_cfg_path.exists():
        raise FileNotFoundError(f"conf_head.yaml not found for head '{head_id}' at {head_cfg_path}")
    head_cfg = yaml.safe_load(head_cfg_path.read_text())

    if not encoder_id:
        encoder_id = head_cfg.get("base_encoder") or head_cfg.get("encoder_id")
        if not encoder_id:
            raise KeyError(f"encoder_id not found in test or head configs for test '{test_id}'")

    model = load_encoder_head(encoder_id, head_id, device=device)
    model.eval()
    return model, test_cfg, head_cfg


# ---------- DataLoader for SHAP ----------

def get_dataloader_for_shap(
    df_samples,
    *,
    dataset_name: str,
    patch_size: int = 64,
    label_mode: str = "multilabel",
    num_classes: int = 1,
    batch_size: int = 64,
    num_workers: int = 0,
    persistent_workers: bool = False,
) -> Tuple[DataLoader, torch.utils.data.Dataset]:
    """
    Deterministic, non-shuffling DataLoader over df_samples using lazy crops.

    Returns (loader, dataset). Each batch yields:
      img   : FloatTensor [B, C, H, W]
      label : Tensor      [B, K] (multilabel float) or [B] (multiclass long)
      locs  : tuple(int,int) per sample
      fname : str per sample
    """
    ds = ClassificationCropDataset(
        df_samples=df_samples,
        dataset_name=dataset_name,
        crop_size=patch_size,
        label_mode=label_mode,
        num_classes=num_classes,
        scale_col="scale",
        transform=None,
    )

    loader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=(num_workers > 0 and persistent_workers),
        collate_fn=_collate_skip_none,
    )
    return loader, ds


# ---------- SHAP explainer prep ----------

class _ModelWrapper(nn.Module):
    """
    Simple wrapper that exposes a plain forward(x)->logits interface.
    Kept for clarity; EncoderClassifier already behaves this way.
    """
    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)  # logits [B,K]


def prepare_shap_explainer(
    model: nn.Module,
    background_loader: DataLoader,
    device: Optional[torch.device] = None,
    background_strategy: str = "zeros",  # "zeros" or "data"
    background_max_batches: int = 1,
) -> Tuple[shap.DeepExplainer, torch.Tensor, torch.Tensor]:
    """
    Prepare a SHAP DeepExplainer for the composite model.

    Returns:
      explainer : shap.DeepExplainer
      background: tensor used as background [B0, C, H, W]
      sample_in : a sample batch from the loader [B, C, H, W]
    """
    device = device or (torch.device("cuda") if torch.cuda.is_available()
                        else torch.device("cpu"))

    # Take one sample batch to infer shape
    sample_img, *_ = next(iter(background_loader))
    sample_in = sample_img.to(device).float()  # [B, C, H, W]

    # Build a background
    if background_strategy == "zeros":
        background = torch.zeros_like(sample_in[:1])  # use a single zero baseline
    else:
        # Average a few real batches as background
        imgs_accum = []
        with torch.no_grad():
            for bidx, (imgs, *_) in enumerate(background_loader):
                imgs_accum.append(imgs.float())
                if bidx + 1 >= max(1, int(background_max_batches)):
                    break
        background = torch.cat(imgs_accum, dim=0).to(device)
        # Optionally reduce (e.g., take a few exemplars). Here we keep as-is.

    # Make all ReLUs non-inplace for stability
    _set_all_relu_non_inplace(model)
    _patch_torchvision_resnet_relu_noninplace()

    wrapped = _ModelWrapper(model).to(device).eval()

    # DeepExplainer expects background on device
    explainer = shap.DeepExplainer(wrapped, background)
    return explainer, background, sample_in


# ---------- SHAP computation ----------

def compute_shap_values_batch(
    explainer: shap.DeepExplainer,
    dataloader: DataLoader,
    device: Optional[torch.device] = None,
    max_batches: Optional[int] = None,
    check_additivity: bool = False,
) -> Tuple[np.ndarray, List[int]]:
    """
    Compute SHAP values for every sample in dataloader.

    Output shape: (N, C, H, W, K)  — per-class attributions on the original inputs.
    (If your old code expected an extra simulated-RGB axis, average it away — we
     already attribute w.r.t. the [C,H,W] inputs before the internal 3x replication.)
    """
    device = device or (torch.device("cuda") if torch.cuda.is_available()
                        else torch.device("cpu"))

    all_batches: List[np.ndarray] = []
    all_indices: List[int] = []

    n_seen = 0
    for bidx, (imgs, labels, locs, fnames) in enumerate(dataloader):
        x = imgs.to(device).float()  # [B,C,H,W]

        # SHAP returns a list (length K) of arrays with the same shape as x
        sv_list = explainer.shap_values(x, check_additivity=check_additivity)

        if not isinstance(sv_list, (list, tuple)):
            # Single-output edge case: make it a list of length 1
            sv_list = [sv_list]

        # Convert to numpy and stack along the last axis => (B, C, H, W, K)
        sv_list_np = [s if isinstance(s, np.ndarray) else s.detach().cpu().numpy() for s in sv_list]
        sv_batch = np.stack(sv_list_np, axis=-1)
        all_batches.append(sv_batch)

        # Keep track of running indices (0..N-1)
        bsz = x.size(0)
        all_indices.extend(range(n_seen, n_seen + bsz))
        n_seen += bsz

        if max_batches is not None and (bidx + 1) >= max_batches:
            break

    shap_values = np.concatenate(all_batches, axis=0) if all_batches else np.empty((0,))
    return shap_values, all_indices


# ---------- Post-processing & plotting ----------

def smooth_shap_maps(
    shap_values: np.ndarray,
    sigma: float = 1.0,
) -> np.ndarray:
    """
    Apply Gaussian smoothing over the spatial dimensions (H,W).

    Supports:
      (N, C, H, W, K)        — current default
      (N, C, 3, H, W, K)     — legacy format (simulated RGB per channel)

    Returns an array with identical shape to the input.
    """
    arr = np.asarray(shap_values)
    if arr.ndim not in (5, 6):
        raise ValueError(f"Unsupported shape {arr.shape}; expected 5D or 6D.")

    if not _HAS_SCIPY:
        # Lightweight fallback: no smoothing if SciPy is unavailable
        return arr.copy()

    out = arr.copy()
    if arr.ndim == 5:
        # (N,C,H,W,K)
        N, C, H, W, K = arr.shape
        for n in range(N):
            for c in range(C):
                for k in range(K):
                    out[n, c, :, :, k] = _gaussian_filter(arr[n, c, :, :, k], sigma=sigma, mode="nearest")
    else:
        # (N,C,3,H,W,K)
        N, C, R, H, W, K = arr.shape
        for n in range(N):
            for c in range(C):
                for r in range(R):
                    for k in range(K):
                        out[n, c, r, :, :, k] = _gaussian_filter(arr[n, c, r, :, :, k], sigma=sigma, mode="nearest")
    return out


def plot_shap_heatmaps(
    shap_values,
    channel_strings=None,
    class_strings=None,
    title="SHAP Heatmaps",
    average_over_batch=True,
    figsize_multiplier=2.0,
):
    """
    Plots SHAP heatmaps with rows=classes (K) and cols=channels (C).
    Colorbar gets its own fixed column so layout shouldn't break.

    Accepts shapes:
      (N, C, H, W, K)        ← preferred (CLEAR-IT inputs)
      (N, C, 3, H, W, K)     ← legacy; averages over axis=2
    """
    arr = np.asarray(shap_values)

    # Legacy simulated-RGB → average it away
    if arr.ndim == 6:  # (N, C, 3, H, W, K)
        arr = arr.mean(axis=2)
    if arr.ndim != 5:
        raise ValueError(f"Expected (N,C,H,W,K) or (N,C,3,H,W,K); got {arr.shape}")

    # Average across batch if requested
    if average_over_batch and arr.shape[0] > 1:
        arr = arr.mean(axis=0)   # (C,H,W,K)
    else:
        arr = arr[0]             # (C,H,W,K) – first sample

    C, H, W, K = arr.shape

    # Labels
    if class_strings is None or len(class_strings) != K:
        class_strings = [f"Class {i}" for i in range(K)]
    if channel_strings is None or len(channel_strings) != C:
        channel_strings = [f"Ch {j}" for j in range(C)]

    # Zero-centered color scale
    vmax = float(np.max(np.abs(arr)))
    vmax = vmax if vmax > 0 else 1e-8
    vmin = -vmax

    # We’ll visualize as (K, C, H, W)
    vis = np.transpose(arr, (3, 0, 1, 2))

    # ---- Layout: K rows, C image columns + 1 colorbar column ----
    fig_h = max(2.0, K * figsize_multiplier)
    fig_w = max(2.0, C * figsize_multiplier) + 0.6  # extra width for cbar
    fig = plt.figure(figsize=(fig_w, fig_h))
    gs = gridspec.GridSpec(
        nrows=K,
        ncols=C + 1,
        width_ratios=[1] * C + [0.05],
        wspace=0.1,
        hspace=0.1,
    )

    # Axes grid
    axs = np.empty((K, C), dtype=object)
    im = None
    for i in range(K):
        for j in range(C):
            ax = fig.add_subplot(gs[i, j])
            axs[i, j] = ax
            im = ax.imshow(vis[i, j], cmap="RdBu_r", interpolation="nearest", vmin=vmin, vmax=vmax)
            ax.set_xticks([]); ax.set_yticks([])
            if i == 0:
                ax.set_title(channel_strings[j], fontsize=10)
            if j == 0:
                ax.set_ylabel(class_strings[i], fontsize=10, rotation=0, labelpad=30, va="center")

    # Dedicated colorbar axis (full-height, last col)
    cax = fig.add_subplot(gs[:, -1])
    cbar = fig.colorbar(im, cax=cax)
    cbar.set_label("SHAP value", rotation=270, labelpad=12)

    # Title & spacing (avoid tight_layout; we already manage geometry)
    fig.suptitle(title, fontsize=14, y=0.99)
    fig.subplots_adjust(top=0.92)

    return fig, axs


# ---------- Simple report helper ----------

def classification_report(df, class_columns: List[str]) -> "pd.DataFrame":
    """
    Build a TP/FP/TN/FN count summary per class column.
    """
    import pandas as pd

    rows = []
    for col in class_columns:
        counts = df[col].value_counts()
        tp = int(counts.get("TP", 0))
        fp = int(counts.get("FP", 0))
        tn = int(counts.get("TN", 0))
        fn = int(counts.get("FN", 0))
        rows.append({
            "Class": col,
            "TP": tp, "FP": fp, "TN": tn, "FN": fn,
            "Actual Positives": tp + fn,
        })
    return pd.DataFrame(rows)
