# clearit/plotting/importance.py
from typing import Optional, Sequence, Tuple, Union
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.image import AxesImage
from matplotlib.colorbar import Colorbar
from mpl_toolkits.axes_grid1 import make_axes_locatable

ArrayLike = Union[np.ndarray, Sequence[Sequence[float]]]

def plot_importance_matrix(
    I_row: ArrayLike,
    chans: Optional[Sequence[str]] = None,
    clss: Optional[Sequence[str]] = None,
    ax: Optional[Axes] = None,
    title: Optional[str] = None,
    *,
    font_scale: float = 1.0,
    show_colorbar: bool = False,
    cbar_label: Optional[str] = None,
    cbar_width_scale: float = 1.0,   # multiplier on a 5% base width
    cbar_pad: float = 0.02,
    grid: bool = True,
    annotate: bool = False,
    annot_fmt: str = "%.2f",
    annot_font_scale: Optional[float] = None,
) -> Tuple[Axes, AxesImage, Optional[Colorbar], Optional[Figure]]:
    """
    Plot a row-normalized importance matrix (C x K) with classes as rows and channels as columns.
    Color scale: white (0) → red (1).

    Parameters
    ----------
    I_row : array-like (C, K)
        Row-normalized importance (sum over channels = 1 per class).
    chans : list[str] or None
        Channel names (columns). If None, uses 'Ch {j}'.
    clss : list[str] or None
        Class names (rows). If None, uses 'Cl {i}'.
    ax : matplotlib.axes.Axes or None
        If None, a new figure and axes are created.
    title : str or None
        Optional title.
    font_scale : float
        Global font scaling.
    show_colorbar : bool
        If True, add a colorbar to the right.
    cbar_label : str or None
        Optional colorbar label (e.g., "Row-normalized importance").
    cbar_width_scale : float
        Multiplier for colorbar width (helps avoid label/ticks clash).
    cbar_pad : float
        Padding between image and colorbar (in axes fraction).
    grid : bool
        Draw minor-grid cell boundaries.
    annotate : bool
        If True, overlay numeric values in each tile.
    annot_fmt : str
        Format string for annotations (e.g., "%.2f").
    annot_font_scale : float or None
        Font scale for annotations; defaults to 0.8 * font_scale if None.

    Returns
    -------
    ax, im, cbar, fig
        Axes, image handle, colorbar (or None), and Figure (or None if ax was provided).
    """
    I_row = np.asarray(I_row)
    if I_row.ndim != 2:
        raise ValueError(f"I_row must be 2D (C, K); got shape {I_row.shape}")

    C, K = I_row.shape
    M = I_row.T  # (K, C)

    created_fig: Optional[Figure] = None
    if ax is None:
        created_fig, ax = plt.subplots(
            figsize=(max(4, C * 0.9 * font_scale), max(3, K * 0.9 * font_scale))
        )

    im: AxesImage = ax.imshow(M, vmin=0.0, vmax=1.0, cmap="Reds", interpolation="nearest")

    xlabels = list(chans) if chans is not None else [f"Ch {j}" for j in range(C)]
    ylabels = list(clss)  if clss  is not None else [f"Cl {i}" for i in range(K)]

    ax.set_xticks(np.arange(C))
    ax.set_yticks(np.arange(K))
    ax.set_xticklabels(xlabels, fontsize=9 * font_scale, rotation=45, ha="right")
    ax.set_yticklabels(ylabels, fontsize=9 * font_scale)

    ax.set_xticks(np.arange(-0.5, C, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, K, 1), minor=True)
    if grid:
        ax.grid(which="minor", color="w", linewidth=0.6)
    ax.tick_params(which="minor", bottom=False, left=False)

    if title:
        ax.set_title(title, fontsize=11 * font_scale)

    if annotate:
        afs = annot_font_scale if annot_font_scale is not None else 0.8 * font_scale
        for i in range(K):
            for j in range(C):
                val = M[i, j]
                ax.text(
                    j, i, annot_fmt % val,
                    ha="center", va="center",
                    fontsize=8 * afs,
                    color=("white" if val >= 0.6 else "black"),
                )

    cbar: Optional[Colorbar] = None
    if show_colorbar:
        divider = make_axes_locatable(ax)
        # Use a 5% base width scaled by cbar_width_scale
        cax = divider.append_axes("right", size=f"{5 * cbar_width_scale}%", pad=cbar_pad)
        cbar = plt.colorbar(im, cax=cax)
        cbar.ax.tick_params(labelsize=8 * font_scale)
        if cbar_label:
            cbar.set_label(cbar_label, rotation=270, labelpad=10 * font_scale, fontsize=9 * font_scale)

    return ax, im, cbar, created_fig