# clearit/shap/compute.py
from typing import List, Optional, Tuple
import numpy as np
import torch

def compute_shap_values_batch(
    explainer,
    dataloader,
    device: Optional[torch.device] = None,
    max_batches: Optional[int] = None,
    check_additivity: bool = False,
) -> Tuple[np.ndarray, List[int]]:
    """
    Returns SHAP values with shape (N, C, H, W, K) for the classifier logits.
    """
    device = device or (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
    all_batches: List[np.ndarray] = []
    all_indices: List[int] = []
    n_seen = 0

    for bidx, (imgs, *_rest) in enumerate(dataloader):
        x = imgs.to(device).float()
        sv_list = explainer.shap_values(x, check_additivity=check_additivity)
        if not isinstance(sv_list, (list, tuple)):
            sv_list = [sv_list]
        sv_np = [s if isinstance(s, np.ndarray) else s.detach().cpu().numpy() for s in sv_list]
        sv_batch = np.stack(sv_np, axis=-1)  # (B,C,H,W,K)
        all_batches.append(sv_batch)
        bsz = x.size(0)
        all_indices.extend(range(n_seen, n_seen + bsz))
        n_seen += bsz
        if max_batches is not None and (bidx + 1) >= max_batches:
            break

    return (np.concatenate(all_batches, axis=0) if all_batches else np.empty((0,))), all_indices

def smooth_shap_maps(shap_values: np.ndarray, sigma: float = 1.0) -> np.ndarray:
    """
    Gaussian smoothing over spatial dims (H,W). Supports:
      (N,C,H,W,K) and (N,C,3,H,W,K) — returns same shape as input.
    """
    try:
        from scipy.ndimage import gaussian_filter
    except Exception:
        return np.asarray(shap_values).copy()

    arr = np.asarray(shap_values)
    out = arr.copy()
    if arr.ndim == 5:
        N, C, H, W, K = arr.shape
        for n in range(N):
            for c in range(C):
                for k in range(K):
                    out[n, c, :, :, k] = gaussian_filter(arr[n, c, :, :, k], sigma=sigma, mode="nearest")
    elif arr.ndim == 6:
        N, C, R, H, W, K = arr.shape
        for n in range(N):
            for c in range(C):
                for r in range(R):
                    for k in range(K):
                        out[n, c, r, :, :, k] = gaussian_filter(arr[n, c, r, :, :, k], sigma=sigma, mode="nearest")
    else:
        raise ValueError(f"Unsupported shape {arr.shape}")
    return out
