# clearit/plotting/boxplot.py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from typing import Sequence, Optional, Mapping, Tuple
from .utils import get_group_color
import matplotlib.colors as mcolors


def boxplot_performance(
    df: pd.DataFrame,
    *,
    value_col: str = 'Value',
    group_col: str = 'Group',
    x_col: str = 'Configuration',
    order: Optional[Sequence[str]] = None,
    palette: Optional[Mapping[str, Tuple[float,float,float]]] = None,
    show_total: bool = True,
    show_mean_of_medians: bool = True,
    whis: Tuple[int,int] = (5,95),
    showfliers: bool = False,
    figsize: Tuple[int,int] = (10,6),
    title: str = '',
    xlabel: str = '',
    ylabel: Optional[str] = None,
    ylim: Tuple[float,float] = (0,1.05),
    yticks: Optional[Sequence[float]] = None,
    xtick_rotation: float = 0,
    font_size: int = 10,
    legend_loc: str = 'upper center',
    legend_ncol: int = 1,
    label_map: Optional[Mapping[str,str]] = None,
    show: bool = True
) -> Tuple[plt.Axes, pd.DataFrame]:
    """
    Plot boxplots of df[value_col] grouped by group_col across x_col categories.
    """
    # 1) Prepare order & palette
    if order is None:
        order = sorted(df[x_col].unique())
    groups = list(df[group_col].unique())
    if palette is None:
        palette = {g: get_group_color(g, 1) for g in groups}

    df[x_col]      = pd.Categorical(df[x_col],      categories=order,  ordered=True)
    df[group_col]  = pd.Categorical(df[group_col],  categories=groups, ordered=True)

    plt.rcParams.update({'font.size': font_size})
    fig, ax = plt.subplots(figsize=figsize)

    grey_rgb = mcolors.to_rgb('lightgray')

    # 2) Draw the overall “total” box
    if show_total:
        total_df = df[[x_col, value_col]].copy()
        sns.boxplot(
            data=total_df,
            x=x_col, y=value_col,
            order=order,
            width=0.75,
            color='lightgray',
            whis=whis,
            showfliers=showfliers,
            ax=ax,
            boxprops={'facecolor':'lightgray','edgecolor':'lightgray'}
        )

    # 3) Draw the per‐group boxes, but remove their outlines by setting linewidth=0
    sns.boxplot(
        data=df,
        x=x_col, y=value_col, hue=group_col,
        order=order,
        palette=palette,
        whis=whis,
        showfliers=showfliers,
        dodge=True,
        ax=ax,
        boxprops={'linewidth': 0},            # ← no visible box border
        medianprops={'label':'_median_'},
        whiskerprops={'label':'_whisker_'},
        capprops={'label':'_cap_'}
    )

    # 4) Re‐outline each group‐box by matching its facecolor back to the palette
    for box in ax.artists:
        fc = box.get_facecolor()[:3]
        # skip the grey “total” if present
        if np.allclose(fc, grey_rgb, atol=1e-3):
            box.set_edgecolor(grey_rgb)
            continue
        # find which group this belongs to
        for grp, col in palette.items():
            if np.allclose(fc, col, atol=1e-3):
                box.set_facecolor(col)
                box.set_edgecolor(col)
                break

    # 5) Re‐color the median, whisker and cap lines
    median_lines  = [l for l in ax.lines if l.get_label() == '_median_']
    whisker_lines = [l for l in ax.lines if l.get_label() == '_whisker_']
    cap_lines     = [l for l in ax.lines if l.get_label() == '_cap_']

    # medians
    for i, ln in enumerate(median_lines):
        grp = groups[i % len(groups)]
        ln.set_color(get_group_color(grp, 2))
        ln.set_linewidth(1.5)
    # whiskers & caps
    for i, ln in enumerate(whisker_lines):
        grp = groups[(i//2) % len(groups)]
        ln.set_color(get_group_color(grp, 0))
        ln.set_linewidth(1.5)
    for i, ln in enumerate(cap_lines):
        grp = groups[(i//2) % len(groups)]
        ln.set_color(get_group_color(grp, 0))
        ln.set_linewidth(1.5)

    # 6) Add dashed “mean of medians” lines
    if show_mean_of_medians:
        mom = (
            df.groupby([x_col, group_col])[value_col]
              .median()
              .groupby(level=0)
              .mean()
        )
        for i, xc in enumerate(order):
            ax.hlines(
                y=mom.loc[xc],
                xmin=i - 0.4, xmax=i + 0.4,
                colors='black', linestyle='dashed'
            )
        mean_line = Line2D(
            [0], [0], color='black', linestyle='dashed',
            label='mean of medians'
        )

    # 7) Build the legend: first the hue handles, then “total”, then “mean of medians”
    handles, labels = ax.get_legend_handles_labels()

    # Remap labels for display
    if label_map:
        labels = [ label_map.get(lbl, lbl) for lbl in labels ]

    if show_total:
        total_patch = Patch(facecolor=grey_rgb, edgecolor=grey_rgb, label='total')
        handles.append(total_patch)
        labels.append('total')

    if show_mean_of_medians:
        handles.append(mean_line)
        labels.append('mean of medians')

    ax.legend(
        handles=handles,
        labels=labels,
        loc=legend_loc,
        ncol=legend_ncol,
        frameon=False
    )

    # 8) Final styling
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel or value_col)
    ax.set_ylim(*ylim)
    if yticks is not None:
        ax.set_yticks(yticks)
    plt.xticks(rotation=xtick_rotation)
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()

    fig = ax.get_figure()

    if not show:
        plt.close(fig)

    # Build the summary table
    # 1) median per (Configuration, Group)
    # 1) median per (Configuration, Group)
    med = (
        df.groupby([x_col, group_col])[value_col]
          .median()
          .unstack(fill_value=np.nan)
          .rename_axis(columns=None)
    )

    pieces = [med]
    cols   = list(med.columns)

    # 2) “total” column, if requested
    if show_total:
        total_med = (
            df.groupby(x_col)[value_col]
              .median()
              .rename('total')
        )
        pieces.append(total_med)
        cols.append('total')

    # 3) “mean of medians” column, if requested
    if show_mean_of_medians:
        mom = med.mean(axis=1).rename('mean_of_medians')
        pieces.append(mom)
        cols.append('mean_of_medians')

    # 4) join them
    summary_df = pd.concat(pieces, axis=1)

    # 5) enforce column order
    summary_df = summary_df[cols]

    return ax, summary_df