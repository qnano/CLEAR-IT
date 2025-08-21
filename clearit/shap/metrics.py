# clearit/shap/metrics.py
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Sequence

def _norm(s: str) -> str:
    return "".join(ch for ch in str(s) if ch.isalnum()).lower()

# Treat these as "ancillary" across modalities.
# (MxIF: DAPI/AF; MIBI: dsDNA; plus BG=background for MIBI)
ANCILLARY_ALIASES = {
    "dapi", "af", "autofluorescence",
    "dsdna", "dna", "intercalator",
    "bg", "background",
}

def ancillary_indices(channel_names: Sequence[str], extra_aliases: Sequence[str] = ()) -> List[int]:
    keys = set(ANCILLARY_ALIASES) | { _norm(a) for a in extra_aliases }
    return [i for i, ch in enumerate(channel_names or []) if _norm(ch) in keys]

def _to_5d(shap_values: np.ndarray) -> np.ndarray:
    """
    Ensure shape is (N, C, H, W, K). If given (N, C, 3, H, W, K), average axis=2.
    """
    arr = np.asarray(shap_values)
    if arr.ndim == 6:  # (N, C, 3, H, W, K)
        arr = arr.mean(axis=2)
    if arr.ndim != 5:
        raise ValueError(f"Expected (N,C,H,W,K) or (N,C,3,H,W,K); got {arr.shape}")
    return arr

def importance_matrix_abs_mean(shap_values: np.ndarray) -> np.ndarray:
    """
    Mean absolute SHAP over samples and spatial dims.
    Returns I with shape (C, K).
    """
    arr = _to_5d(shap_values)          # (N,C,H,W,K)
    I = np.mean(np.abs(arr), axis=(0, 2, 3))  # (C,K)
    return I

def infer_pairs_by_name(channel_strings: List[str], class_strings: List[str]) -> List[Tuple[int,int]]:
    """
    Pair class 'X+' with channel 'X' when alphanumeric names match (case-insensitive).
    Special case: CK+ pairs to Pan-Keratin if literal 'CK' channel not present.
    """
    chan_map = {_norm(ch): i for i, ch in enumerate(channel_strings or [])}

    # Pan-Keratin aliases
    pan_keys = {"pankeratin", "pancytokeratin", "panck", "pancytokeratins"}
    pan_idx = next((chan_map[k] for k in pan_keys if k in chan_map), None)

    pairs: List[Tuple[int,int]] = []
    for k, cls in enumerate(class_strings or []):
        key = _norm(str(cls).replace("+", ""))  # 'CK+' -> 'ck'
        if key in chan_map:
            pairs.append((chan_map[key], k))
        elif key == "ck" and pan_idx is not None:
            pairs.append((pan_idx, k))
    return pairs

def diagonality_index(I: np.ndarray, pairs: List[Tuple[int,int]], eps: float = 1e-12) -> float:
    """
    Fraction of total absolute importance that lies on matched class↔channel pairs.
    Returns 0.0 if total mass is ~0 or if there are no pairs.
    """
    I = np.asarray(I, dtype=np.float64)
    if I.size == 0:
        return 0.0
    diag_mass = float(sum(I[c, k] for (c, k) in pairs)) if pairs else 0.0
    tot = float(I.sum())
    if tot <= eps:
        return 0.0
    return diag_mass / tot

def channel_specificity_entropy(I: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """
    Per-class entropy H_k over channels: H = -sum_c p_c log p_c, with p_c = I[c,k]/sum_c I[c,k].
    Returns array (K,) with natural-log entropy (convert to bits with / ln(2) if desired).
    """
    C, K = I.shape
    sums = I.sum(axis=0, keepdims=True) + eps
    P = I / sums
    H = -(P * np.log(P + eps)).sum(axis=0)
    return H  # shape (K,)

def _center_surround_masks(H: int, W: int, center_frac: float = 0.25, ring: Tuple[float,float] = (0.35, 0.65)):
    """
    Build boolean masks for a central disk (radius = center_frac * min(H,W)/2)
    and a surrounding annulus with inner/outer radii = ring[0]/ring[1] * (min(H,W)/2).
    """
    yy, xx = np.mgrid[0:H, 0:W]
    cy, cx = (H - 1) / 2.0, (W - 1) / 2.0
    rr = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
    rmax = min(H, W) / 2.0

    r_center = center_frac * rmax
    r_in, r_out = ring[0] * rmax, ring[1] * rmax

    center_mask = rr <= r_center
    surround_mask = (rr >= r_in) & (rr <= r_out)
    return center_mask, surround_mask

def center_surround_metrics(
    shap_values: np.ndarray,
    center_frac: float = 0.25,
    ring: Tuple[float, float] = (0.35, 0.65),
) -> Dict[str, np.ndarray]:
    """
    Compute center-surround metrics on signed SHAP.
    Returns dict of arrays with shape (C,K) unless noted:

      - center_abs_share: fraction of total |SHAP| in center disk (mean over N).
      - CSI: (mean|center| - mean|surround|) / (mean|center| + mean|surround|)  (mean over N).
      - sign_inversion_frac: across samples, fraction with mean_center and mean_surround having opposite sign (C,K).
    """
    arr = _to_5d(shap_values)                # (N,C,H,W,K)
    N, C, H, W, K = arr.shape
    cmask, smask = _center_surround_masks(H, W, center_frac=center_frac, ring=ring)
    n_center = cmask.sum()
    n_surround = smask.sum()

    # Prepare outputs
    center_abs_share = np.zeros((C, K), dtype=float)
    CSI = np.zeros((C, K), dtype=float)
    sign_inversion_frac = np.zeros((C, K), dtype=float)

    # Flatten spatial dims for speed
    cmask_flat = cmask.reshape(-1)
    smask_flat = smask.reshape(-1)

    # Loop channels/classes (small dims; OK to loop)
    for c in range(C):
        for k in range(K):
            maps = arr[:, c, :, :, k].reshape(N, -1)  # (N, H*W)

            # per-sample means in center & surround (signed)
            mc = maps[:, cmask_flat].mean(axis=1)
            ms = maps[:, smask_flat].mean(axis=1)

            # per-sample absolute sums for share
            abs_center = np.abs(maps[:, cmask_flat]).sum(axis=1)
            abs_total  = np.abs(maps).sum(axis=1)

            # aggregate
            center_abs_share[c, k] = float((abs_center / np.maximum(abs_total, 1e-12)).mean())
            num = (np.abs(mc).mean() - np.abs(ms).mean())
            den = (np.abs(mc).mean() + np.abs(ms).mean())
            CSI[c, k] = num / den if den > 0 else np.nan
            sign_inversion_frac[c, k] = float(np.mean(np.sign(mc) * np.sign(ms) < 0.0))

    return {
        "center_abs_share": center_abs_share,     # (C,K)
        "CSI": CSI,                               # (C,K)
        "sign_inversion_frac": sign_inversion_frac,  # (C,K)
    }


def to_dataframe_matrix(M: np.ndarray, channel_strings: List[str], class_strings: List[str]) -> pd.DataFrame:
    """
    Helper: convert a (C,K) matrix to a tidy DataFrame with channels as rows and classes as columns.
    """
    C, K = M.shape
    rows = channel_strings if channel_strings and len(channel_strings) == C else [f"Ch {i}" for i in range(C)]
    cols = class_strings if class_strings and len(class_strings) == K else [f"Class {j}" for j in range(K)]
    return pd.DataFrame(M, index=rows, columns=cols)
