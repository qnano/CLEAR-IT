#!/usr/bin/env python3
"""
Compute per-patient image quality rankings from extended image statistics.

Input:
    - A CSV produced by compute_image_statistics_extended.py, containing
      per-image metrics (one row per TIFF, with 'patient_id' column).

Workflow:
    - Aggregate per-image metrics to per-patient metrics via median.
    - Compute the image-quality ranking based on selected channels and the
      'percentile_0.1_std' (and 'percentile_100.0_std' for background).
    - Save a CSV with per-patient ranking information.

Output:
    - patient_rankings.csv in the specified output directory, with columns:
        * patient_id
        * n_images
        * Total_Score_std   (image quality score)
        * Normalized_Score_std
        * Rank_desc         (1 = best image quality)
"""

import argparse
from pathlib import Path
from typing import Optional, Dict, List

import numpy as np
import pandas as pd


# Ranking configuration

# Channels and settings used for the notebook-based ranking.
# (channel_index, percentile, ascending_flag)
# ascending=True  -> smaller metric value gets a better rank (rank=1)
# ascending=False -> larger metric value gets a better rank (rank=1)
LEGACY_CHANNEL_CONFIG = [
    (3,   0.1,  True),   # CD3
    (5,   0.1,  True),   # CD8
    (7,   0.1,  True),   # CD20
    (6,   0.1,  True),   # CD56
    (4,   0.1,  True),   # CD68
    (8, 100.0, False),   # background (effectively constant in practice)
]


def build_legacy_order_dict(statistic_suffix: str = "std") -> Dict[str, bool]:
    """
    Build a mapping {column_name: ascending_bool} for the ranking score.
    """
    order_dict: Dict[str, bool] = {}
    for channel_idx, percent, ascending in LEGACY_CHANNEL_CONFIG:
        col_name = f"channel_{channel_idx}_top_{percent}_percent_{statistic_suffix}"
        order_dict[col_name] = ascending
    return order_dict


# Core routines

def aggregate_per_patient(df_images: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate per-image extended statistics to per-patient metrics via median.

    Returns a DataFrame with:
        - patient_id
        - n_images  (number of images/ROIs for that patient)
        - median of all numeric metrics per patient
    """
    if "patient_id" not in df_images.columns:
        raise KeyError("Expected column 'patient_id' in extended stats CSV.")

    # Count images per patient
    counts = (
        df_images.groupby("patient_id")["filename"]
        .count()
        .rename("n_images")
    )

    # Aggregate all numeric columns by median
    df_numeric = df_images.select_dtypes(include=[np.number])
    df_patient_median = (
        df_images.groupby("patient_id")[df_numeric.columns]
        .median()
    )

    # Combine counts and medians
    df_patient = pd.concat([df_patient_median, counts], axis=1).reset_index()

    # Move 'n_images' next to patient_id
    cols = ["patient_id", "n_images"] + [c for c in df_patient.columns if c not in ("patient_id", "n_images")]
    df_patient = df_patient[cols]

    return df_patient


def compute_legacy_ranking(df_patient: pd.DataFrame) -> pd.DataFrame:
    """
    Given a per-patient metrics DataFrame, compute the ranking
    and append these columns:
        - Total_Score_std
        - Normalized_Score_std
        - Rank_desc  (1 = best image quality)
    """
    order_dict = build_legacy_order_dict(statistic_suffix="std")

    # Check that all required columns exist
    missing = [col for col in order_dict if col not in df_patient.columns]
    if missing:
        raise KeyError(
            "The following required columns for ranking are missing "
            f"from the per-patient dataframe:\n{missing}"
        )

    df = df_patient.copy()

    # Rank each metric according to its ascending flag
    rank_cols: List[str] = []
    for col, ascending in order_dict.items():
        rank_col = col + "_rank"
        df[rank_col] = df[col].rank(method="min", ascending=ascending)
        rank_cols.append(rank_col)

    # Sum the ranks into the image-quality score.
    df["Total_Score_std"] = df[rank_cols].sum(axis=1)

    # Normalize to [0, 1] (higher = better)
    max_score = df["Total_Score_std"].max()
    if max_score > 0:
        df["Normalized_Score_std"] = df["Total_Score_std"] / max_score
    else:
        df["Normalized_Score_std"] = 0.0

    # Rank scores descending (1 = best)
    df["Rank_desc"] = df["Total_Score_std"].rank(method="min", ascending=False)

    return df


# Script entry point

def main(stats_csv_path: str, output_dir: str) -> None:
    """
    Compute per-patient rankings from an extended stats CSV.

    Parameters
    ----------
    stats_csv_path : str
        Path to image_statistics_extended.csv (per-image stats).
    output_dir : str
        Directory where patient_rankings.csv will be saved.
    """
    stats_path = Path(stats_csv_path)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not stats_path.exists():
        raise FileNotFoundError(f"Extended stats CSV not found: {stats_path}")

    print(f"Loading extended stats from: {stats_path}")
    df_images = pd.read_csv(stats_path)

    print("Aggregating per-patient metrics (median over images)...")
    df_patient = aggregate_per_patient(df_images)

    print("Computing image-quality ranking...")
    df_ranked = compute_legacy_ranking(df_patient)

    # Sort by quality (best first)
    df_ranked_sorted = df_ranked.sort_values(by="Total_Score_std", ascending=False)

    out_csv = out_dir / "patient_rankings.csv"
    df_ranked_sorted.to_csv(out_csv, index=False)
    print(f"Saved patient rankings to: {out_csv}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compute per-patient image quality rankings from extended image statistics."
    )
    parser.add_argument(
        "stats_csv",
        type=str,
        help="Path to image_statistics_extended.csv (per-image stats).",
    )
    parser.add_argument(
        "output_dir",
        type=str,
        help="Directory where patient_rankings.csv will be saved.",
    )

    args = parser.parse_args()
    main(args.stats_csv, args.output_dir)
