"""
Helpers to prepare data for performance-vs-image-quality scatter plots.

We reuse `gather_pr_auc` to compute a single PR-AUC value per test_dir,
where each test_dir corresponds to a network trained on a single patient.

Typical workflow:
- Build an `entries` list where each entry has `path`, `group`, and `config`.
- Call `prepare_scatter_dataframe_single_patient_networks(...)` with that list
  and the path to `image_statistics_extended.csv`.
- Feed the resulting DataFrame into `clearit.plotting.scatter`.
"""

from pathlib import Path
from typing import Optional, Sequence, Dict, List

import numpy as np
import pandas as pd

from .gather_pr_auc import gather_pr_auc


# Image-quality ranking helpers

# Channels and settings used for the image-quality ranking.
# (channel_index, percentile, ascending_flag)
# ascending=True  -> smaller metric value gets a better rank (rank=1)
# ascending=False -> larger metric value gets a better rank (rank=1)
LEGACY_CHANNEL_CONFIG = [
    (3,   0.1,  True),   # CD3
    (5,   0.1,  True),   # CD8
    (7,   0.1,  True),   # CD20
    (6,   0.1,  True),   # CD56
    (4,   0.1,  True),   # CD68
    (8, 100.0, False),   # background
]


def _aggregate_image_stats_per_patient(df_images: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate per-image extended statistics to per-patient metrics via median.
    """
    if "patient_id" not in df_images.columns:
        raise KeyError("Expected column 'patient_id' in image statistics CSV.")

    df_numeric = df_images.select_dtypes(include=[np.number])
    df_patient = (
        df_images.groupby("patient_id")[df_numeric.columns]
        .median()
        .reset_index()
    )
    return df_patient


def _build_legacy_order_dict(statistic_suffix: str = "std") -> Dict[str, bool]:
    """
    Build a mapping {column_name: ascending_bool} for the ranking score.
    """
    order_dict: Dict[str, bool] = {}
    for channel_idx, percent, ascending in LEGACY_CHANNEL_CONFIG:
        col_name = f"channel_{channel_idx}_top_{percent}_percent_{statistic_suffix}"
        order_dict[col_name] = ascending
    return order_dict


def _compute_legacy_ranking(df_patient_stats: pd.DataFrame) -> pd.DataFrame:
    """
    Compute the aggregated ranking summary from per-patient image statistics.
    """
    order_dict = _build_legacy_order_dict("std")

    missing = [c for c in order_dict if c not in df_patient_stats.columns]
    if missing:
        raise KeyError(
            "Missing required columns for ranking:\n"
            f"{missing}"
        )

    df = df_patient_stats.copy()

    rank_cols: List[str] = []
    for col, ascending in order_dict.items():
        rank_col = col + "_rank"
        df[rank_col] = df[col].rank(method="min", ascending=ascending)
        rank_cols.append(rank_col)

    df["Total_Score_std"] = df[rank_cols].sum(axis=1)
    max_score = df["Total_Score_std"].max()
    if max_score > 0:
        df["Normalized_Score_std"] = df["Total_Score_std"] / max_score
    else:
        df["Normalized_Score_std"] = 0.0

    return df


# Performance gathering for single-patient networks

def gather_network_performance_for_scatter(
    entries: Sequence[dict],
) -> pd.DataFrame:
    """
    Use `gather_pr_auc` to compute a single PR-AUC value per test_dir/network.

    Expected entries format (per network):
        {
            "path": Path to test_dir with sigmoid/target CSVs,
            "group": dataset label (e.g. "TNBC1-MxIF8"),
            "config": patient ID (e.g. "P01")
        }

    We call `gather_pr_auc` with `chunks=1` and `total=True`, which yields
    one PR-AUC per class for each entry. We then average across classes to
    get a single scalar PR-AUC per entry.

    Returns
    -------
    df_perf : pandas.DataFrame with columns:
        - patient_id   (taken from entry['config'])
        - dataset      (taken from entry['group'])
        - pr_auc       (mean across classes)
    """
    # chunks=1, total=True → one row per (entry, class)
    df_raw = gather_pr_auc(entries, chunks=1, total=True)

    if df_raw.empty:
        raise RuntimeError("gather_pr_auc returned an empty DataFrame.")

    # Average across classes for each network (Configuration)
    df_perf = (
        df_raw.groupby(["Configuration", "Group"], as_index=False)["Value"]
        .mean()
    )

    df_perf = df_perf.rename(
        columns={
            "Configuration": "patient_id",  # config should be patient ID
            "Group": "dataset",
            "Value": "pr_auc",
        }
    )

    return df_perf


# Scatter-plot dataframe assembly

def prepare_scatter_dataframe_single_patient_networks(
    entries: Sequence[dict],
    image_stats_csv: Path,
    *,
    extra_rank_metrics: Optional[Sequence[str]] = None,
) -> pd.DataFrame:
    """
    Prepare a DataFrame for performance-vs-ranking scatter plots, where each
    point corresponds to one network trained on a single patient.

    Parameters
    ----------
    entries : sequence of dict
        List of entries, one per network, with keys:
            - "path"   : test_dir path containing sigmoid/target CSVs
            - "group"  : dataset label (e.g. "TNBC1-MxIF8")
            - "config" : patient ID (e.g. "P01")
        (This is the same structure as used by `gather_pr_auc`.)
    image_stats_csv : Path
        Path to image_statistics_extended.csv (per-image stats).
    extra_rank_metrics : sequence of str, optional
        Names of additional numeric columns in the per-patient image statistics
        to convert into normalized ranking columns. For each name 'col', a
        column 'rank_<col>' is created where higher values correspond to higher
        rank (1.0 = best).

    Returns
    -------
    df_scatter : pandas.DataFrame
        DataFrame with one row per patient/network and columns:
            - patient_id
            - dataset
            - pr_auc                (overall PR-AUC for that network)
            - Normalized_Score_std  (image-quality rank)
            - rank_<metric>         (for each entry in extra_rank_metrics)
    """
    image_stats_csv = Path(image_stats_csv)

    # Gather performance per network.
    df_perf = gather_network_performance_for_scatter(entries)

    # Aggregate image statistics per patient and compute the ranking score.
    df_images = pd.read_csv(image_stats_csv)
    df_patient_stats = _aggregate_image_stats_per_patient(df_images)
    df_ranked = _compute_legacy_ranking(df_patient_stats)

    # Merge performance with ranking on patient_id.
    df = df_perf.merge(
        df_ranked,
        on="patient_id",
        how="inner",
    )

    # Add extra ranking metrics when requested.
    if extra_rank_metrics is not None:
        for col in extra_rank_metrics:
            if col not in df.columns:
                raise KeyError(
                    f"Requested extra_rank_metrics column '{col}' not found "
                    "after merging performance and image statistics."
                )
            # Higher metric value -> better quality -> higher normalized rank
            rank = df[col].rank(method="min", ascending=False)
            df[f"rank_{col}"] = rank / rank.max()

    # Keep the primary output columns first, plus any extras.
    base_cols = ["patient_id", "dataset", "pr_auc", "Normalized_Score_std"]
    extra_cols = [c for c in df.columns if c.startswith("rank_")]
    other_cols = [c for c in df.columns if c not in base_cols + extra_cols]

    # Put important columns first
    df_scatter = df[base_cols + extra_cols + other_cols]

    return df_scatter
