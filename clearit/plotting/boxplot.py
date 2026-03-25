import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from typing import Sequence, Optional, Mapping, Tuple, Dict
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
    yaxis_right: bool = False,
    xtick_rotation: float = 0,
    ytick_rotation: float = 0,
    font_size: int = 10,
    legend_loc: str = 'upper center',
    legend_ncol: int = 1,
    label_map: Optional[Mapping[str,str]] = None,
    show: bool = True
) -> Tuple[plt.Axes, pd.DataFrame, Dict[str, pd.DataFrame]]:
    """
    Plot boxplots of df[value_col] grouped by group_col across x_col categories.

    Returns
    -------
    ax : matplotlib.axes.Axes
        The axes with the boxplot.

    summary_df : pandas.DataFrame
        Table with one row per `x_col` value and columns:
        - one column per group (median per (x_col, group_col))
        - optional 'total' column (median over all groups per x_col)
        - optional 'mean_of_medians' column (mean over group medians).

    plot_tables : dict[str, pandas.DataFrame]
        Dictionary suitable for writing to an Excel file with multiple sheets.
        Keys correspond to legend entries:

        - One key per group in `group_col`, each mapping to a DataFrame:

              index  : x_col (e.g. Configuration)
              columns: ['lower_whisker', 'lower_box', 'median',
                        'upper_box', 'upper_whisker']

        - If show_total=True, an extra key 'total' with the same structure,
          computed across all groups.

        - If show_mean_of_medians=True, an extra key 'mean_of_medians' with:

              index  : x_col
              column : ['mean_of_medians']
    """

    # Helper: compute 5-number box stats given a 1D array-like
    def _compute_box_stats(values: pd.Series, whis: Tuple[int, int]):
        arr = np.asarray(values.dropna())
        if arr.size == 0:
            return {
                'lower_whisker': np.nan,
                'lower_box':     np.nan,
                'median':        np.nan,
                'upper_box':     np.nan,
                'upper_whisker': np.nan,
            }

        low_w, high_w = whis
        q1 = np.percentile(arr, 25)
        q2 = np.percentile(arr, 50)
        q3 = np.percentile(arr, 75)
        wl = np.percentile(arr, low_w)
        wu = np.percentile(arr, high_w)

        return {
            'lower_whisker': wl,
            'lower_box':     q1,
            'median':        q2,
            'upper_box':     q3,
            'upper_whisker': wu,
        }

    # Prepare order and palette.
    if order is None:
        order = sorted(df[x_col].unique())
    groups = list(df[group_col].unique())
    if palette is None:
        palette = {g: get_group_color(g, 1) for g in groups}

    df[x_col]     = pd.Categorical(df[x_col],     categories=order,  ordered=True)
    df[group_col] = pd.Categorical(df[group_col], categories=groups, ordered=True)

    plt.rcParams.update({'font.size': font_size})
    fig, ax = plt.subplots(figsize=figsize)

    # Move y-axis to the right
    if yaxis_right:
        ax.yaxis.tick_right()
        ax.yaxis.set_label_position("right")

    grey_rgb = mcolors.to_rgb('lightgray')

    # Draw the overall "total" box.
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
            boxprops={'facecolor': 'lightgray', 'edgecolor': 'lightgray'}
        )

    # Draw the per-group boxes without visible outlines.
    sns.boxplot(
        data=df,
        x=x_col, y=value_col, hue=group_col,
        order=order,
        palette=palette,
        whis=whis,
        showfliers=showfliers,
        dodge=True,
        ax=ax,
        boxprops={'linewidth': 0},
        medianprops={'label': '_median_'},
        whiskerprops={'label': '_whisker_'},
        capprops={'label': '_cap_'}
    )

    # Reapply outlines that match each box facecolor.
    for box in ax.artists:
        fc = box.get_facecolor()[:3]
        # Skip the grey "total" box if present.
        if np.allclose(fc, grey_rgb, atol=1e-3):
            box.set_edgecolor(grey_rgb)
            continue
        # Match the facecolor back to its group.
        for grp, col in palette.items():
            if np.allclose(fc, col, atol=1e-3):
                box.set_facecolor(col)
                box.set_edgecolor(col)
                break

    # Recolor the median, whisker, and cap lines.
    median_lines  = [l for l in ax.lines if l.get_label() == '_median_']
    whisker_lines = [l for l in ax.lines if l.get_label() == '_whisker_']
    cap_lines     = [l for l in ax.lines if l.get_label() == '_cap_']

    # Medians
    for i, ln in enumerate(median_lines):
        grp = groups[i % len(groups)]
        ln.set_color(get_group_color(grp, 2))
        ln.set_linewidth(1.5)
    # Whiskers and caps
    for i, ln in enumerate(whisker_lines):
        grp = groups[(i // 2) % len(groups)]
        ln.set_color(get_group_color(grp, 0))
        ln.set_linewidth(1.5)
    for i, ln in enumerate(cap_lines):
        grp = groups[(i // 2) % len(groups)]
        ln.set_color(get_group_color(grp, 0))
        ln.set_linewidth(1.5)

    # Add dashed "mean of medians" lines.
    mom_series = None
    if show_mean_of_medians:
        mom_series = (
            df.groupby([x_col, group_col])[value_col]
              .median()
              .groupby(level=0)
              .mean()
        )
        for i, xc in enumerate(order):
            ax.hlines(
                y=mom_series.loc[xc],
                xmin=i - 0.4, xmax=i + 0.4,
                colors='black', linestyle='dashed'
            )
        mean_line = Line2D(
            [0], [0], color='black', linestyle='dashed',
            label='mean of medians'
        )

    # Build the legend from group handles, then add "total" and "mean of medians".
    handles, labels = ax.get_legend_handles_labels()

    # Remap labels for display
    if label_map:
        labels = [label_map.get(lbl, lbl) for lbl in labels]

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

    # Final styling
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel or value_col)
    ax.set_ylim(*ylim)
    if yticks is not None:
        ax.set_yticks(yticks)
    plt.xticks(rotation=xtick_rotation)
    plt.yticks(rotation=ytick_rotation)
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()

    fig = ax.get_figure()
    if not show:
        plt.close(fig)

    # Build the summary table.
    med = (
        df.groupby([x_col, group_col])[value_col]
          .median()
          .unstack(fill_value=np.nan)
          .rename_axis(columns=None)
    )

    pieces = [med]
    cols   = list(med.columns)

    if show_total:
        total_med = (
            df.groupby(x_col)[value_col]
              .median()
              .rename('total')
        )
        pieces.append(total_med)
        cols.append('total')

    if show_mean_of_medians:
        if mom_series is None:
            mom_series = (
                df.groupby([x_col, group_col])[value_col]
                  .median()
                  .groupby(level=0)
                  .mean()
            )
        mom = mom_series.rename('mean_of_medians')
        pieces.append(mom)
        cols.append('mean_of_medians')

    summary_df = pd.concat(pieces, axis=1)
    summary_df = summary_df[cols]  # enforce column order
    summary_df = summary_df.reindex(order)

    # Build the per-legend-entry tables defining boxes and lines.
    # Box stats per (x_col, group_col) for hue groups.
    records = []
    for xc in order:
        for grp in groups:
            mask = (df[x_col] == xc) & (df[group_col] == grp)
            vals = df.loc[mask, value_col]
            if vals.dropna().empty:
                continue
            stats = _compute_box_stats(vals, whis)
            stats[x_col] = xc
            stats[group_col] = grp
            records.append(stats)

    box_stats_df = (
        pd.DataFrame.from_records(records)
        .set_index([x_col, group_col])
        .sort_index()
    )

    # Create one sheet per group
    plot_tables: Dict[str, pd.DataFrame] = {}
    for grp in groups:
        if (x_col, grp) not in box_stats_df.index:
            # Skip groups that are entirely missing.
            if not (box_stats_df.index.get_level_values(group_col) == grp).any():
                continue
        grp_df = box_stats_df.xs(grp, level=group_col).copy()
        grp_df.index.name = x_col
        grp_df = grp_df.reindex(order)
        plot_tables[str(grp)] = grp_df


    # "Total" box stats, if requested.
    if show_total:
        total_records = []
        for xc in order:
            vals = df.loc[df[x_col] == xc, value_col]
            if vals.dropna().empty:
                continue
            stats = _compute_box_stats(vals, whis)
            stats[x_col] = xc
            total_records.append(stats)

        if total_records:
            total_df_stats = (
                pd.DataFrame.from_records(total_records)
                .set_index(x_col)
            )
            total_df_stats = total_df_stats.reindex(order)
            plot_tables['total'] = total_df_stats


    # "Mean of medians" line, if requested.
    if show_mean_of_medians and mom_series is not None:
        mom_df = mom_series.to_frame(name='mean_of_medians').copy()
        mom_df.index.name = x_col
        mom_df = mom_df.reindex(order)
        plot_tables['mean_of_medians'] = mom_df


    return ax, summary_df, plot_tables
