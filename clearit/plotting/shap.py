# clearit/plotting/shap.py
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec
from typing import Optional, Sequence, Tuple, Union
import pandas as pd

def plot_shap_heatmaps(
    shap_values: np.ndarray,
    channel_strings: Optional[Sequence[str]] = None,
    class_strings:   Optional[Sequence[str]] = None,
    title: Optional[str] = None,
    average_over_batch: bool = True,
    figsize_multiplier: float = 2.0,
    normalize: str = "none",      # "none" | "global" | "row" | "column"
    overlay_matrix: Optional[Union[np.ndarray, pd.DataFrame]] = None,  # (C,K)
    overlay_fmt: str = "%.2g",
    overlay_loc: str = "lower-right",   # "lower-right"|"lower-left"|"upper-right"|"upper-left"
    font_scale: float = 1.0,           
    cbar_width_scale: float = 1.0,   # widen the colorbar column if text is large
    cbar_pad_scale: float = 1.0,     # add extra padding between label/ticks/bar
) -> Tuple[plt.Figure, np.ndarray]:
    """
    Plot SHAP heatmaps with classes as rows and channels as columns.

    Parameters (extra):
      overlay_matrix: optional (C,K) values to draw in each tile (e.g., importance).
      font_scale: global scaling factor for all text (1.0 = default sizes).
    """
    arr = np.asarray(shap_values)

    # Handle legacy simulated-RGB
    if arr.ndim == 6:  # (N,C,3,H,W,K)
        arr = arr.mean(axis=2)
    if arr.ndim != 5:
        raise ValueError(f"Expected (N,C,H,W,K) or (N,C,3,H,W,K); got {arr.shape}")

    if average_over_batch and arr.shape[0] > 1:
        arr = arr.mean(axis=0)  # (C,H,W,K)
    else:
        arr = arr[0]            # (C,H,W,K)

    C, H, W, K = arr.shape

    # Labels
    if class_strings is None or len(class_strings) != K:
        class_strings = [f"Class {i}" for i in range(K)]
    if channel_strings is None or len(channel_strings) != C:
        channel_strings = [f"Ch {j}" for j in range(C)]

    # ----- Normalization -----
    eps = 1e-12
    arr_norm = arr.copy()

    if normalize not in {"none", "global", "row", "column"}:
        raise ValueError(f"normalize must be one of 'none','global','row','column', got {normalize!r}")

    if normalize == "global":
        s = float(np.max(np.abs(arr_norm)))
        if s > eps:
            arr_norm /= s
        vmin, vmax = -1.0, 1.0
        cbar_label = "SHAP value (global-normalized)"
    elif normalize == "row":
        s = np.max(np.abs(arr_norm), axis=(0,1,2), keepdims=True)  # (1,1,1,K)
        s = np.maximum(s, eps)
        arr_norm = arr_norm / s
        vmin, vmax = -1.0, 1.0
        cbar_label = "SHAP value (row-normalized)"
    elif normalize == "column":
        s = np.max(np.abs(arr_norm), axis=(1,2,3), keepdims=True)  # (C,1,1,1)
        s = np.maximum(s, eps)
        arr_norm = arr_norm / s
        vmin, vmax = -1.0, 1.0
        cbar_label = "SHAP value (column-normalized)"
    else:  # "none"
        vmax = float(np.max(np.abs(arr_norm))) or 1e-8
        vmin = -vmax
        cbar_label = "SHAP value"

    # Arrange for plotting: (K, C, H, W)
    vis = np.transpose(arr_norm, (3, 0, 1, 2))

    # ----- Optional overlay (C,K) -----
    overlay = None
    if overlay_matrix is not None:
        if isinstance(overlay_matrix, pd.DataFrame):
            df = overlay_matrix.copy()
            if (df.index.tolist() != channel_strings) and set(channel_strings).issubset(set(df.index)):
                df = df.reindex(channel_strings)
            if (df.columns.tolist() != class_strings) and set(class_strings).issubset(set(df.columns)):
                df = df[class_strings]
            overlay = df.values
        else:
            overlay = np.asarray(overlay_matrix)
        if overlay.shape != (C, K):
            raise ValueError(f"overlay_matrix must have shape (C,K); got {overlay.shape}")

    # ----- Figure layout -----
    fig_h = max(2.0, K * figsize_multiplier)
    fig_w = max(2.0, C * figsize_multiplier) + 0.6

    # Scale the colorbar column with font size (and user override)
    # base width 0.05 → expand up to ~0.10 for very large fonts
    cb_col = 0.05 * max(1.0, min(2.0, font_scale)) * max(1.0, cbar_width_scale)

    fig = plt.figure(figsize=(fig_w, fig_h))
    gs = gridspec.GridSpec(
        nrows=K,
        ncols=C + 1,
        width_ratios=[1] * C + [cb_col],
        wspace=0.1,
        hspace=0.1,
    )

    # Font sizes (scaled)
    fs_coltitle = 10 * font_scale
    fs_rowlabel = 10 * font_scale
    fs_overlay  = 8  * font_scale
    fs_suptitle = 14 * font_scale
    fs_cbar_lab = 12 * font_scale
    fs_cbar_tick= 9  * font_scale

    pos_map = {
        "lower-right": (0.98, 0.02, "right", "bottom"),
        "lower-left":  (0.02, 0.02, "left",  "bottom"),
        "upper-right": (0.98, 0.98, "right", "top"),
        "upper-left":  (0.02, 0.98, "left",  "top"),
    }
    if overlay_loc not in pos_map:
        raise ValueError("overlay_loc must be one of "
                         "'lower-right','lower-left','upper-right','upper-left'")
    ox, oy, oha, ova = pos_map[overlay_loc]

    axs = np.empty((K, C), dtype=object)
    im = None
    for i in range(K):
        for j in range(C):
            ax = fig.add_subplot(gs[i, j])
            axs[i, j] = ax
            im = ax.imshow(vis[i, j], cmap="RdBu_r", interpolation="nearest", vmin=vmin, vmax=vmax)
            ax.set_xticks([]); ax.set_yticks([])

            if i == 0:
                ax.set_title(channel_strings[j], fontsize=fs_coltitle)
            if j == 0:
                ax.set_ylabel(class_strings[i], fontsize=fs_rowlabel, rotation=0, labelpad=30, va="center")

            if overlay is not None:
                val = overlay[j, i]  # (C,K) -> (row=class i, col=channel j)
                ax.text(
                    ox, oy, (overlay_fmt % val),
                    ha=oha, va=ova, transform=ax.transAxes, fontsize=fs_overlay, color="black",
                    bbox=dict(facecolor="white", alpha=0.6, edgecolor="none", pad=1.2),
                )

    # ----- Colorbar -----
    cax = fig.add_subplot(gs[:, -1])
    cbar = fig.colorbar(im, cax=cax)

    # Scaled font sizes
    fs_cbar_lab  = 12 * font_scale
    fs_cbar_tick =  9 * font_scale

    # Scaled paddings (bigger fonts → more space)
    labelpad = 12 * font_scale * max(1.0, 1.2 * cbar_pad_scale)
    tickpad  =  2 * font_scale * max(1.0, 1.2 * cbar_pad_scale)

    cbar.set_label(cbar_label, rotation=270, labelpad=labelpad, fontsize=fs_cbar_lab)
    cbar.ax.tick_params(labelsize=fs_cbar_tick, pad=tickpad)

    if title:
        fig.suptitle(title, fontsize=fs_suptitle, y=0.99)
        fig.subplots_adjust(top=0.92)

    return fig, axs
