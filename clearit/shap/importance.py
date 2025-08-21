# clearit/shap/importance.py

from typing import Optional, List
import ast
import numpy as np
import pandas as pd
from pathlib import Path

from clearit.shap.io import load_shap_bundle
from clearit.shap.compute import smooth_shap_maps
from clearit.shap.metrics import (
    importance_matrix_abs_mean,
    channel_specificity_entropy,
    infer_pairs_by_name,
)

def load_row_normalized_importance(
    bundle_base: Path,
    *,
    apply_smoothing: bool = True,
    sigma: float = 2.0,
    canonical_order: Optional[List[str]] = None,
):
    """
    Load a SHAP bundle and return the row-normalized channel×class importance matrix.

    Parameters
    ----------
    bundle_base : Path
        Base path of the bundle (without extension), e.g. ".../SHAP_TP_highest".
    apply_smoothing : bool, default True
        If True and the bundle is not already marked as smoothed, apply Gaussian
        smoothing (sigma) to spatial SHAP maps before aggregation.
    sigma : float, default 2.0
        Gaussian smoothing sigma (pixels).
    canonical_order : list[str] or None
        If provided, subset/reorder the channel rows to this order (names that are
        not present are ignored). Labels are reordered to match.

    Returns
    -------
    I_row : np.ndarray shape (C, K)
        Row-normalized importance matrix (sum over channels = 1 for each class).
    chans : list[str] or None
        Channel names aligned to the rows of I_row (after any reordering).
    clss : list[str] or None
        Class names aligned to the columns of I_row.
    """
    shap_vals, df_cells, meta = load_shap_bundle(bundle_base)

    labels = meta.get("labels", {})
    chans  = labels.get("channel_strings")
    clss   = labels.get("class_strings")
    order  = labels.get("desired_channel_order")

    # Apply saved ordering from metadata, if present
    if order:
        shap_vals = shap_vals[:, order, :, :, :]
        if chans:
            chans = [chans[i] for i in order]

    # Optional smoothing (only if not already smoothed)
    if apply_smoothing and not meta.get("shap", {}).get("smoothed", False):
        shap_vals = smooth_shap_maps(shap_vals, sigma=sigma)

    # Absolute-mean importance, then row-normalize by class
    I = importance_matrix_abs_mean(shap_vals)                 # (C, K)
    I_row = I / (I.sum(axis=0, keepdims=True) + 1e-12)        # (C, K)

    # Optional canonical channel order
    if canonical_order and chans:
        present = {name: idx for idx, name in enumerate(chans)}
        keep = [present[name] for name in canonical_order if name in present]
        if keep:
            I_row = I_row[keep, :]
            chans = [chans[i] for i in keep]

    return I_row, chans, clss


def _norm(s: str) -> str:
    return "".join(ch for ch in str(s) if ch.isalnum()).lower()

def _find_nucleus_idx(channel_names: Optional[List[str]]) -> Optional[int]:
    """Prefer dsDNA (MIBI) if present; else DAPI (MxIF); else None."""
    if not channel_names:
        return None
    idx = { _norm(n): i for i, n in enumerate(channel_names) }
    for key in ("dsdna", "dna", "intercalator"):  # MIBI-like
        if key in idx:
            return idx[key]
    for key in ("dapi",):                          # MxIF-like
        if key in idx:
            return idx[key]
    return None

def load_and_row_normalize_TP_highest(
    bundle_base: Path,
    *,
    apply_smoothing: bool = True,
    sigma: float = 2.0,
    canonical_order: Optional[List[str]] = None,
):
    """(Unchanged behavior) Returns (I_df, Irow_df, channel_names, class_names, df_cells, meta)."""
    shap_vals, df_cells, meta = load_shap_bundle(bundle_base)

    labels = meta.get("labels", {})
    chan_names  = labels.get("channel_strings")
    class_names = labels.get("class_strings")
    order       = labels.get("desired_channel_order")

    if order is not None:
        shap_vals = shap_vals[:, order, :, :, :]
        if chan_names:
            chan_names = [chan_names[i] for i in order]

    if apply_smoothing and not meta.get("shap", {}).get("smoothed", False):
        shap_vals = smooth_shap_maps(shap_vals, sigma=sigma)

    I = importance_matrix_abs_mean(shap_vals)  # (C, K)
    denom = I.sum(axis=0, keepdims=True) + 1e-12
    I_row = I / denom

    if canonical_order and chan_names:
        present = {name: idx for idx, name in enumerate(chan_names)}
        keep = [present[n] for n in canonical_order if n in present]
        if keep:
            I = I[keep, :]
            I_row = I_row[keep, :]
            chan_names = [chan_names[i] for i in keep]

    if not chan_names:
        chan_names = [f"Ch {c}" for c in range(I.shape[0])]
    if not class_names:
        class_names = [f"Cl {k}" for k in range(I.shape[1])]

    I_df    = pd.DataFrame(I,     index=chan_names, columns=class_names)
    Irow_df = pd.DataFrame(I_row, index=chan_names, columns=class_names)
    return I_df, Irow_df, chan_names, class_names, df_cells, meta

def summarize_importance_matrices(
    I_df: pd.DataFrame,
    Irow_df: pd.DataFrame,
    chan_names: List[str],
    class_names: List[str],
    df_cells: Optional[pd.DataFrame] = None,
    *,
    dataset_label: Optional[str] = None,
):
    """(Unchanged behavior) Returns (per_class_df, overall_df)."""
    C, K = I_df.shape
    ent_vec = channel_specificity_entropy(I_df.values) / (np.log(C) + 1e-12)

    pairs = infer_pairs_by_name(chan_names or [], class_names or [])
    chan_for_class = {k: c for c, k in pairs}
    diag_share = np.full(K, np.nan, dtype=float)
    for k in range(K):
        c = chan_for_class.get(k, None)
        if c is not None:
            diag_share[k] = float(Irow_df.iloc[c, k])

    nuc_idx = _find_nucleus_idx(chan_names)
    nuc_share = np.full(K, np.nan, dtype=float)
    if nuc_idx is not None:
        nuc_share = Irow_df.iloc[nuc_idx, :].values.astype(float)

    top2_share = np.minimum(1.0, np.nan_to_num(diag_share, nan=0.0) +
                                 np.nan_to_num(nuc_share,   nan=0.0))

    support = np.full(K, np.nan, dtype=float)
    if df_cells is not None and "label" in df_cells.columns:
        arr = []
        for x in df_cells["label"].values:
            if isinstance(x, str):
                try:
                    x = ast.literal_eval(x)
                except Exception:
                    x = None
            if isinstance(x, (list, tuple)) and len(x) == K:
                arr.append(np.asarray(x, dtype=float))
        if arr:
            M = np.stack(arr, axis=0)
            support = M.sum(axis=0)

    per_class_df = pd.DataFrame({
        "class": class_names,
        "matched_channel": [chan_names[chan_for_class[k]] if k in chan_for_class else None for k in range(K)],
        "diagonal_share": diag_share,
        "nucleus_share": nuc_share,
        "top2_share": top2_share,
        "entropy_norm": ent_vec,
        "support": support,
    })

    baseline = 1.0 / C
    med_diag  = np.nanmedian(diag_share)
    med_nuc   = np.nanmedian(nuc_share)
    med_top2  = np.nanmedian(top2_share)
    med_Hnorm = np.nanmedian(ent_vec)

    overall = {
        "dataset": dataset_label,
        "C_channels": C,
        "K_classes": K,
        "median_diagonal_share": med_diag,
        "diagonal_fold_enrichment": (med_diag / baseline) if baseline > 0 else np.nan,
        "median_nucleus_share": med_nuc,
        "nucleus_fold_enrichment": (med_nuc / baseline) if baseline > 0 else np.nan,
        "median_top2_share": med_top2,
        "median_entropy_norm": med_Hnorm,
        "uniform_baseline": baseline,
    }
    overall_df = pd.DataFrame([overall])
    return per_class_df, overall_df
