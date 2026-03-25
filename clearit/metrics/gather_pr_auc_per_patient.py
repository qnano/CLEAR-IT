"""
Utilities for gathering per-patient PR-AUC scores from sigmoid prediction CSVs.

Each CSV is assumed to have a filename of the form "Pxx_ROIyy.csv" and contain
columns:

    - sigmoid_<class_name>
    - target_<class_name>

For each patient, we:
    - compute PR-AUC per class and per ROI
    - take the median PR-AUC across ROIs for each class
    - take the mean across classes, yielding a single scalar performance:
      "mean of medians" (MoM PR-AUC) per patient

This is intended for use in scatter plots comparing per-patient performance
with image-quality rankings.
"""

from pathlib import Path
from typing import Optional, Sequence, Callable, Dict, List

import glob
import os

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score


def _derive_patient_id_from_filename(filename: str) -> str:
    """
    Derive a patient identifier from a filename of the form "Pxx_ROIyy.csv".
    If the pattern does not match, return the stem as patient_id.
    """
    stem = Path(filename).stem  # e.g. "P01_ROI01"
    parts = stem.split("_")
    if len(parts) >= 1 and parts[0].startswith("P"):
        return parts[0]
    return stem


def gather_patient_pr_auc(
    test_dir: Path,
    *,
    metric_fn: Optional[Callable[[np.ndarray, np.ndarray], float]] = None,
    class_labels: Optional[Sequence[str]] = None,
) -> pd.DataFrame:
    """
    Gather per-patient PR-AUC scores from a test directory.

    Parameters
    ----------
    test_dir : Path
        Directory containing CSV files with sigmoid_* and target_* columns.
    metric_fn : callable, optional
        Function(y_true, y_pred) -> float. Defaults to average_precision_score.
    class_labels : sequence of str, optional
        Class names in the order of the sigmoid_ columns. If None, inferred from
        the suffix of the sigmoid_ column names.

    Returns
    -------
    df_patient : pandas.DataFrame
        DataFrame with one row per patient and columns:
            - patient_id
            - MoM              (mean-of-medians PR-AUC per patient)
            - n_rois           (number of ROI CSVs for that patient)
            - (optional) per-class median PR-AUC columns: "prauc_<class_name>"
    """
    if metric_fn is None:
        metric_fn = average_precision_score

    test_dir = Path(test_dir)

    # Collect all CSVs under test_dir/**
    files = glob.glob(os.path.join(str(test_dir), "**", "*.csv"), recursive=True)
    if not files:
        raise FileNotFoundError(f"No CSV prediction files found under {test_dir}")

    roi_records: List[Dict] = []

    for f in files:
        fname = os.path.basename(f)
        patient_id = _derive_patient_id_from_filename(fname)

        try:
            df = pd.read_csv(f)
        except Exception:
            continue

        pred_cols = [c for c in df.columns if c.startswith("sigmoid_")]
        if not pred_cols:
            continue

        true_cols = [c.replace("sigmoid_", "target_") for c in pred_cols]
        if not all(c in df.columns for c in true_cols):
            continue

        # Determine class labels
        if class_labels is None:
            labels = [c.split("sigmoid_")[1] for c in pred_cols]
        else:
            if len(class_labels) != len(pred_cols):
                raise ValueError(
                    "Provided class_labels length does not match number of sigmoid_ columns"
                )
            labels = list(class_labels)

        y_pred = df[pred_cols].values
        y_true = df[true_cols].values

        # For each class, compute PR-AUC for this ROI
        for j, label in enumerate(labels):
            try:
                val = metric_fn(y_true[:, j], y_pred[:, j])
            except Exception:
                # In case of degenerate labels, fall back to NaN
                val = np.nan

            roi_records.append(
                {
                    "patient_id": patient_id,
                    "roi_id": Path(fname).stem,
                    "class": label,
                    "prauc": val,
                }
            )

    if not roi_records:
        raise RuntimeError(f"No valid sigmoid_/target_ columns found under {test_dir}")

    df_roi = pd.DataFrame(roi_records)

    # Drop rows with NaN PR-AUC
    df_roi = df_roi.dropna(subset=["prauc"])

    # Median PR-AUC per patient × class across ROIs
    df_patient_class = (
        df_roi.groupby(["patient_id", "class"])["prauc"]
        .median()
        .reset_index(name="prauc_median")
    )

    # Mean of medians across classes (MoM)
    df_patient = (
        df_patient_class.groupby("patient_id")["prauc_median"]
        .mean()
        .reset_index(name="MoM")
    )

    # Count ROIs per patient
    counts = df_roi.groupby("patient_id")["roi_id"].nunique().rename("n_rois")
    df_patient = df_patient.merge(counts, on="patient_id", how="left")

    # Optionally, add one column per class with median PR-AUC
    pivot = df_patient_class.pivot(
        index="patient_id", columns="class", values="prauc_median"
    )
    if pivot is not None:
        pivot = pivot.add_prefix("prauc_")
        df_patient = df_patient.merge(
            pivot.reset_index(), on="patient_id", how="left"
        )

    return df_patient

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


# Ranking configuration reused here
LEGACY_CHANNEL_CONFIG = [
    (3,   0.1,  True),   # CD3
    (5,   0.1,  True),   # CD8
    (7,   0.1,  True),   # CD20
    (6,   0.1,  True),   # CD56
    (4,   0.1,  True),   # CD68
    (8, 100.0, False),   # background
]


def _build_legacy_order_dict(statistic_suffix: str = "std") -> Dict[str, bool]:
    """
    Build a mapping {column_name: ascending_bool} for the ranking score.
    """
    order_dict: Dict[str, bool] = {}
    for channel_idx, percent, ascending in LEGACY_CHANNEL_CONFIG:
        col_name = f"channel_{channel_idx}_percentile_{percent}_{statistic_suffix}"
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


def prepare_scatter_dataframe(
    test_dir: Path,
    image_stats_csv: Path,
    *,
    extra_rank_metrics: Optional[Sequence[str]] = None,
) -> pd.DataFrame:
    """
    Prepare a DataFrame for performance-vs-ranking scatter plots.

    Parameters
    ----------
    test_dir : Path
        Directory containing sigmoid prediction CSVs (for performance).
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
        DataFrame with one row per patient and columns:
            - patient_id
            - MoM                     (performance)
            - Normalized_Score_std    (normalized rank)
            - rank_<metric>           (for each extra_rank_metrics entry)
    """
    test_dir = Path(test_dir)
    image_stats_csv = Path(image_stats_csv)

    # Gather per-patient performance.
    df_perf = gather_patient_pr_auc(test_dir)

    # Compute per-patient image statistics and the ranking score.
    df_images = pd.read_csv(image_stats_csv)
    df_patient_stats = _aggregate_image_stats_per_patient(df_images)
    df_ranked = _compute_legacy_ranking(df_patient_stats)

    # Base merge on patient_id
    df = df_perf.merge(
        df_ranked[["patient_id", "Normalized_Score_std"] + 
                  [c for c in df_ranked.columns if c not in ("patient_id", "Normalized_Score_std")]],
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

    # For typical usage, the primary ranking column is "Normalized_Score_std".
    return df
