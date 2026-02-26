# clearit/plotting/scatter.py
"""
Scatter plots for per-patient performance vs. image-quality rankings.

The main function takes a DataFrame with one performance column (e.g. MoM PR-AUC)
and one or more ranking columns (normalized to [0, 1]) and plots them.
"""

from typing import Sequence, Dict, Tuple, Optional

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import spearmanr


def scatter_performance_vs_rank(
    df: pd.DataFrame,
    perf_col: str,
    rank_cols: Sequence[str],
    *,
    figsize: Tuple[float, float] = (2.0, 2.0),
    xlabel: str = "Normalized image quality rank",
    ylabel: str = "Mean of median PR-AUC",
    title: Optional[str] = None,
    xlim: Tuple[float, float] = (0.0, 1.0),
    ylim: Tuple[float, float] = (0.0, 1.0),
    marker_size: float = 10.5,
    show_legend: bool = False,
) -> Tuple[plt.Axes, Dict[str, Tuple[float, float]], pd.DataFrame]:
    """
    Scatter plot of performance vs. one or more ranking columns.

    Returns
    -------
    ax : matplotlib.axes.Axes
        The axes containing the scatter plot.
    corr_map : dict
        Mapping rank column name -> (Spearman r, p-value).
    data_df : pandas.DataFrame
        Long-format table of the plotted points with columns:
        ['rank_col', 'rank_value', perf_col].
    """
    fig, ax = plt.subplots(figsize=figsize)

    corr_map: Dict[str, Tuple[float, float]] = {}
    records = []  # for tabular data of plotted points

    for rank_col in rank_cols:
        if rank_col not in df.columns:
            raise KeyError(f"Rank column '{rank_col}' not found in DataFrame.")

        # Raw values
        x = df[rank_col].values.astype(float)
        y = df[perf_col].values.astype(float)

        # Drop NaNs so spearmanr and scatter see the same points
        mask = np.isfinite(x) & np.isfinite(y)
        x_plot = x[mask]
        y_plot = y[mask]

        # Spearman correlation
        r, p = spearmanr(x_plot, y_plot)
        corr_map[rank_col] = (r, p)

        # Add to scatter
        label = f"{rank_col} (r={r:.2f})"
        ax.scatter(x_plot, y_plot, label=label, s=marker_size, zorder=10)

        # Add to records table
        for xv, yv in zip(x_plot, y_plot):
            records.append(
                {
                    "rank_col": rank_col,
                    "rank_value": float(xv),
                    perf_col: float(yv),
                }
            )

    # Build tabular DataFrame of points
    data_df = pd.DataFrame.from_records(records)

    # Axes formatting
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_aspect("equal", adjustable="box")

    ax.set_xlabel(xlabel, fontsize=8)
    ax.set_ylabel(ylabel, fontsize=8)

    # ----- Title logic -----
    if len(rank_cols) == 1:
        col = rank_cols[0]
        r, p = corr_map[col]

        # Base title: user-provided or column name
        base_title = title or col

        # Append Spearman text + r and p on new lines
        full_title = (
            f"{base_title}\n"
            f"Spearman correlation coeff.\n"
            f"r={r:.2f}, p={p:.4g}"
        )
        ax.set_title(full_title, fontsize=8)
    else:
        # Multiple rank columns: just use given title (no single r,p)
        if title is not None:
            ax.set_title(title, fontsize=8)

    # Ticks and ticklabels
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_xticklabels(["0", "", "0.5", "", "1.0"], fontsize=8)
    ax.set_yticklabels(["0", "", "0.5", "", "1.0"], fontsize=8)

    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.grid(linestyle="--")

    if len(rank_cols) > 1 or show_legend:
        ax.legend(loc="best", fontsize=6)

    return ax, corr_map, data_df
