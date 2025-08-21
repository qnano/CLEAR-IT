# clearit/plotting/lineplot.py
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from typing import Sequence, Optional, Mapping, Tuple
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
    ax:            Optional[plt.Axes] = None
) -> plt.Axes:
    """
    Plot a shaded band from `region_df` (low/high) plus any number of
    line+marker series from `line_dfs` (mean_of_medians).  

    Parameters
    ----------
    region_df : DataFrame
        Columns: [config_col, group_col, low_col, high_col]. Assumed exactly
        one unique group in this df; used for the legend patch.
    *line_dfs : DataFrame(s)
        Each with columns [config_col, group_col, mid_col]. Each unique
        group will be drawn as a line+marker series.
    categorical_x : bool
        If True, treat the values in `config_col` as discrete categories
        spaced evenly. If False, interpret them as numeric x positions.
    marker_styles : dict
        group_name -> matplotlib marker (e.g. 'D', 'o').
    color_map : dict
        group_name -> color (any Matplotlib color spec).
    default_markers : list
        fallback markers to cycle through.
    """
    plt.rcParams.update({'font.size': font_size})
    fig, ax = (plt.subplots(figsize=figsize) if ax is None else (ax.figure, ax))

    # 1) Create our x-axis positions and tick labels
    x_vals = list(region_df[config_col].to_list())
    if categorical_x:
        x_pos      = np.arange(len(x_vals))
        xtick_lbls = x_vals
    else:
        # numeric spacing
        x_pos      = np.array(x_vals, dtype=float)
        xtick_lbls = x_vals

    # 2) Shaded region
    low_arr  = np.array(region_df[low_col].to_list(),  dtype=float)
    high_arr = np.array(region_df[high_col].to_list(), dtype=float)
    grp0     = region_df[group_col].iat[0]

    ax.fill_between(x_pos, low_arr, high_arr,
                    color=region_color, alpha=region_alpha)
    ax.plot(   x_pos, low_arr,  '--',
              color=region_edgecolor, linewidth=region_linewidth)
    ax.plot(   x_pos, high_arr, '--',
              color=region_edgecolor, linewidth=region_linewidth)

    # 3) Line + markers
    marker_styles = marker_styles or {}
    color_map     = color_map     or {}
    m_it          = iter(default_markers)

    handles = [
        Patch(facecolor=region_color,
              edgecolor=region_edgecolor,
              label=grp0)
    ]

    for df in line_dfs:
        for grp in df[group_col].unique():
            sub = df[df[group_col] == grp]
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

    # 4) Labels, ticks, grid, legend
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
    return ax
