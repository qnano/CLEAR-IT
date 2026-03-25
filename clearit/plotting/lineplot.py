import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
from typing import Sequence, Optional, Mapping, Tuple, Dict
from .utils import get_group_color


def plot_region_and_lines(
    region_df,
    *line_dfs,
    config_col: str = 'Configuration',
    group_col:  str = 'Group',
    low_col:    str = 'low',
    high_col:   str = 'high',
    mid_col:    str = 'mean_of_medians',
    region_color:     str = 'lightgray',
    region_edgecolor: str = 'dimgray',
    region_alpha:     float = 0.4,
    region_linewidth: float = 1.5,
    marker_styles: Optional[Mapping[str,str]] = None,
    color_map:     Optional[Mapping[str,str]] = None,
    default_markers:  Sequence[str] = ('D','o','s','^','v','*','X'),
    categorical_x:     bool = False,
    figsize:       Tuple[int,int] = (10,6),
    title:         str = '',
    xlabel:        str = '',
    ylabel:        Optional[str] = None,
    ylim: Tuple[float,float] = (0,1.05),
    xtick_rotation: float = 0,
    font_size:     int = 10,
    legend_loc:    str = 'upper center',
    legend_ncol:   int = 1,
    label_map: Optional[Mapping[str,str]] = None,
    overlay_points_df: Optional[pd.DataFrame] = None,
    overlay_value_col: str = 'run_mom',
    overlay_marker: str = '+',
    overlay_color: str = 'dimgray',
    overlay_alpha: float = 0.65,
    overlay_size: float = 28,
    overlay_linewidths: float = 1.0,
    overlay_jitter: float = 0.08,
    overlay_label: str = 'individual runs',
    overlay_in_legend: bool = True,
    include_overlay_table: bool = False,
    overlay_table_name: Optional[str] = None,
    ax:            Optional[plt.Axes] = None
) -> Tuple[plt.Axes, Dict[str, pd.DataFrame]]:
    """
    Plot a shaded band from `region_df` (low/high) plus any number of
    line+marker series from `line_dfs` (mid_col).

    Returns
    -------
    ax : matplotlib.axes.Axes
        The axes with the plot.

    tables : dict[str, pandas.DataFrame]
        Dictionary intended for Excel export; one key per legend entry.
        Sheet names are the legend labels (after applying `label_map` if given).

        For the region entry:
            index  : config_col (x-axis order)
            columns: [low_col, high_col]

        For each line group:
            index  : config_col (x-axis order)
            columns: [mid_col]

        Optional run-level overlay points can also be exported when
        include_overlay_table=True.
    """
    plt.rcParams.update({'font.size': font_size})
    fig, ax = (plt.subplots(figsize=figsize) if ax is None else (ax.figure, ax))

    # Create x-axis positions and tick labels.
    x_vals = list(region_df[config_col].to_list())
    if categorical_x:
        x_pos      = np.arange(len(x_vals))
        xtick_lbls = x_vals
    else:
        # numeric spacing
        x_pos      = np.array(x_vals, dtype=float)
        xtick_lbls = x_vals
    x_lookup = dict(zip(x_vals, x_pos))

    # Shaded region
    low_arr  = np.array(region_df[low_col].to_list(),  dtype=float)
    high_arr = np.array(region_df[high_col].to_list(), dtype=float)
    grp0     = region_df[group_col].iat[0]

    ax.fill_between(x_pos, low_arr, high_arr,
                    color=region_color, alpha=region_alpha)
    ax.plot(x_pos, low_arr,  '--',
            color=region_edgecolor, linewidth=region_linewidth)
    ax.plot(x_pos, high_arr, '--',
            color=region_edgecolor, linewidth=region_linewidth)

    # Line and markers
    marker_styles = marker_styles or {}
    color_map     = color_map     or {}
    m_it          = iter(default_markers)

    handles = [
        Patch(facecolor=region_color,
              edgecolor=region_edgecolor,
              label=grp0)
    ]

    # For building tables later
    tables: Dict[str, pd.DataFrame] = {}

    # Region table (sheet name = legend label, after label_map)
    region_sheet_name = label_map.get(grp0, grp0) if label_map else grp0
    region_tbl = (
        region_df[[config_col, low_col, high_col]]
        .set_index(config_col)
        .loc[x_vals]  # enforce same order as plotted
        .copy()
    )
    region_tbl.index.name = config_col
    tables[str(region_sheet_name)] = region_tbl

    # Line groups
    seen_groups = set()

    for df_line in line_dfs:
        for grp in df_line[group_col].unique():
            sub = df_line[df_line[group_col] == grp]
            # map config -> mid
            mapping = dict(zip(sub[config_col].to_list(),
                               sub[mid_col].to_list()))
            # build y-array in exact x_vals order
            y_arr = np.array([mapping[c] for c in x_vals], dtype=float)

            mk  = marker_styles.get(grp, next(m_it))
            col = color_map.get(grp, get_group_color(grp, 1))

            ln, = ax.plot(x_pos, y_arr,
                          marker=mk, linestyle='-',
                          label=grp, color=col)
            handles.append(ln)

            # Build/overwrite table entry for this group
            sheet_name = label_map.get(grp, grp) if label_map else grp
            line_tbl = pd.DataFrame(
                {mid_col: y_arr},
                index=x_vals
            )
            line_tbl.index.name = config_col
            tables[str(sheet_name)] = line_tbl
            seen_groups.add(grp)

    # Optional run-level point overlay (e.g., individual runs).
    if overlay_points_df is not None and not overlay_points_df.empty:
        req_cols = {config_col, overlay_value_col}
        missing_cols = req_cols.difference(overlay_points_df.columns)
        if missing_cols:
            raise ValueError(
                f"overlay_points_df is missing required columns: {sorted(missing_cols)}"
            )

        overlay_df = overlay_points_df[
            overlay_points_df[config_col].isin(x_vals)
        ].copy()
        overlay_df['_x_base'] = overlay_df[config_col].map(x_lookup).astype(float)

        rng = np.random.default_rng(42)
        if overlay_jitter > 0:
            overlay_df['_x_plot'] = (
                overlay_df['_x_base'] +
                rng.uniform(-overlay_jitter, overlay_jitter, len(overlay_df))
            )
        else:
            overlay_df['_x_plot'] = overlay_df['_x_base']

        ax.scatter(
            overlay_df['_x_plot'].to_numpy(),
            overlay_df[overlay_value_col].to_numpy(dtype=float),
            marker=overlay_marker,
            c=overlay_color,
            alpha=overlay_alpha,
            s=overlay_size,
            linewidths=overlay_linewidths
        )

        if overlay_in_legend:
            handles.append(
                Line2D(
                    [], [],
                    linestyle='None',
                    marker=overlay_marker,
                    color=overlay_color,
                    markeredgecolor=overlay_color,
                    alpha=overlay_alpha,
                    markersize=max(4.0, np.sqrt(overlay_size)),
                    label=overlay_label
                )
            )

        if include_overlay_table:
            name = overlay_table_name or overlay_label
            cols = [config_col]
            if group_col in overlay_df.columns:
                cols.append(group_col)
            cols.append(overlay_value_col)
            tables[str(name)] = overlay_df[cols].reset_index(drop=True).copy()

    # Labels, ticks, grid, and legend
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel or mid_col)
    ax.set_ylim(*ylim)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(xtick_lbls, rotation=xtick_rotation)
    ax.grid(axis='y', linestyle='--', alpha=0.3)

    # Remap legend labels if requested
    legend_labels = [
        label_map.get(h.get_label(), h.get_label()) if label_map else h.get_label()
        for h in handles
    ]
    ax.legend(handles=handles,
              labels=legend_labels,
              loc=legend_loc,
              ncol=legend_ncol,
              frameon=False)
    plt.tight_layout()

    return ax, tables
