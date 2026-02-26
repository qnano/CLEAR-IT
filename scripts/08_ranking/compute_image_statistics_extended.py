#!/usr/bin/env python3
"""
Compute extended image quality statistics for multiplex TIFF images.

For each image:
- Derive patient_id and roi_id from filename "Pxx_ROIyy.tiff".
- Treat the image as (C, H, W); if 2D, treat it as a single-channel image.
- Compute per-channel and "total" (all channels flattened) metrics:
  * Legacy trimmed-percentile metrics for p in {0.1, 100}.
  * Basic intensity statistics (mean, median, std, MAD, mean/median).
  * Clipped statistics using upper clip at the 99.5th percentile.
  * Extreme bright tail statistics using the 99.9th percentile.
  * Histogram entropy (after normalization by 99.9th percentile).
  * Simple focus metric based on gradient energy.

Output:
- A CSV file named "image_statistics_extended.csv" in the specified output directory.
"""

import os
import argparse
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import tifffile

from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm
from typing import Optional


# ----------------------------- utility functions ----------------------------- #

def derive_ids_from_filename(filename: str) -> Dict[str, str]:
    """
    Derive patient_id and roi_id from a filename of the form "Pxx_ROIyy.ext".
    If the pattern does not match, use the stem as patient_id and set roi_id to "".
    """
    stem = Path(filename).stem  # e.g. "P01_ROI01"
    parts = stem.split("_")
    patient_id = stem
    roi_id = ""

    if len(parts) >= 2 and parts[0].startswith("P") and parts[1].startswith("ROI"):
        patient_id = parts[0]        # "P01"
        roi_id = parts[1]            # "ROI01"

    return {"patient_id": patient_id, "roi_id": roi_id}


def median_absolute_deviation(x: np.ndarray) -> float:
    """
    Compute the median absolute deviation (MAD) of a 1D array.
    """
    if x.size == 0:
        return 0.0
    med = np.median(x)
    return float(np.median(np.abs(x - med)))


def compute_entropy(values: np.ndarray, n_bins: int = 64) -> float:
    """
    Compute Shannon entropy of intensity values after normalizing by the
    99.9th percentile and clipping to [0, 1].
    """
    if values.size == 0:
        return 0.0

    # Use 99.9th percentile to normalize; if zero, entropy is zero.
    q99_9 = np.percentile(values, 99.9)
    if q99_9 <= 0:
        return 0.0

    norm = np.clip(values / q99_9, 0.0, 1.0)
    hist, _ = np.histogram(norm, bins=n_bins, range=(0.0, 1.0), density=False)
    total = hist.sum()
    if total == 0:
        return 0.0

    p = hist.astype(np.float64) / float(total)
    p = p[p > 0]
    entropy = -np.sum(p * np.log(p))
    return float(entropy)


def compute_focus_gradient_energy(channel_2d: np.ndarray) -> float:
    """
    Compute a simple focus/sharpness metric based on gradient energy.
    Uses finite differences in x and y directions.
    """
    if channel_2d.ndim != 2:
        raise ValueError("Focus metric expects a 2D array.")

    if channel_2d.size == 0:
        return 0.0

    # Horizontal and vertical differences
    dx = channel_2d[:, 1:] - channel_2d[:, :-1]
    dy = channel_2d[1:, :] - channel_2d[:-1, :]
    grad_energy = np.mean(dx ** 2) + np.mean(dy ** 2)
    return float(grad_energy)


def compute_legacy_trimmed_stats(values: np.ndarray, percent: float) -> Dict[str, float]:
    """
    Compute the legacy trimmed-percentile statistics:
    - lower = percentile(percent)
    - upper = percentile(100 - percent)
    - within_bounds = values between [lower, upper]
    - mean and std of within_bounds.

    This matches the original behavior, even if the naming elsewhere
    referred to "top p%".
    """
    if values.size == 0:
        return {
            f"percentile_{percent}_mean": 0.0,
            f"percentile_{percent}_std": 0.0,
        }

    lower = np.percentile(values, percent)
    upper = np.percentile(values, 100.0 - percent)
    mask = (values >= lower) & (values <= upper)
    trimmed = values[mask]

    if trimmed.size == 0:
        trimmed_mean = 0.0
        trimmed_std = 0.0
    else:
        trimmed_mean = float(trimmed.mean())
        trimmed_std = float(trimmed.std())

    return {
        f"percentile_{percent}_mean": trimmed_mean,
        f"percentile_{percent}_std": trimmed_std,
    }


def compute_channel_stats(values: np.ndarray) -> Dict[str, float]:
    """
    Compute basic and extended statistics for a 1D array of intensities.

    Includes:
      - mean, median, std, MAD, mean/median
      - clipped metrics at 99.5th percentile
      - extreme bright tail metrics at 99.9th percentile
      - saturation fraction (exact max)
      - histogram entropy
      - legacy "top p% brightest pixels" metrics for p in {0.1, 100},
        matching the original new_image_statistics.py implementation.
    """
    stats: Dict[str, float] = {}

    if values.size == 0:
        # Initialize all metrics to zero for empty input
        for key in [
            "mean", "median", "std", "mad",
            "mean_to_median",
            "mean_clipped_99_5", "std_clipped_99_5", "mean_to_median_clipped_99_5",
            "frac_extreme_99_9", "mean_extreme_99_9", "std_extreme_99_9",
            "frac_at_max",
            "entropy",
        ]:
            stats[key] = 0.0

        # Legacy top-p metrics
        for p in (0.1, 100.0):
            stats[f"top_{p}_percent_mean"] = 0.0
            stats[f"top_{p}_percent_std"] = 0.0

        return stats

    # ---------------- basic stats ----------------
    mean_val = float(values.mean())
    median_val = float(np.median(values))
    std_val = float(values.std())
    mad_val = median_absolute_deviation(values)
    eps = 1e-6
    mean_to_median = mean_val / (median_val + eps)

    stats["mean"] = mean_val
    stats["median"] = median_val
    stats["std"] = std_val
    stats["mad"] = mad_val
    stats["mean_to_median"] = mean_to_median

    # ------------- clipped stats at 99.5% -------------
    q99_5 = np.percentile(values, 99.5)
    clipped = np.clip(values, 0.0, q99_5)
    mean_clipped = float(clipped.mean())
    median_clipped = float(np.median(clipped))
    std_clipped = float(clipped.std())
    mean_to_median_clipped = mean_clipped / (median_clipped + eps)

    stats["mean_clipped_99_5"] = mean_clipped
    stats["std_clipped_99_5"] = std_clipped
    stats["mean_to_median_clipped_99_5"] = mean_to_median_clipped

    # ------------- extreme bright tail at 99.9% -------------
    q99_9 = np.percentile(values, 99.9)
    extreme_mask = values > q99_9
    n_extreme = int(extreme_mask.sum())
    n_total = int(values.size)
    frac_extreme = float(n_extreme) / float(n_total) if n_total > 0 else 0.0

    if n_extreme > 0:
        extreme_vals = values[extreme_mask]
        mean_extreme = float(extreme_vals.mean())
        std_extreme = float(extreme_vals.std())
    else:
        mean_extreme = 0.0
        std_extreme = 0.0

    stats["frac_extreme_99_9"] = frac_extreme
    stats["mean_extreme_99_9"] = mean_extreme
    stats["std_extreme_99_9"] = std_extreme

    # ------------- fraction at max (pseudo-saturation) -------------
    max_val = float(values.max())
    frac_at_max = float(np.sum(values == max_val)) / float(n_total)
    stats["frac_at_max"] = frac_at_max

    # ------------- entropy -------------
    stats["entropy"] = compute_entropy(values)

    # ------------- legacy "top p% brightest pixels" metrics -------------
    # Match the original code:
    #   sorted_data = np.sort(data)
    #   index = int(np.floor(p / 100.0 * n))
    #   top_percent = sorted_data[-index:]
    #
    # and then mean/std on that subset.
    sorted_vals = np.sort(values)
    n = sorted_vals.size

    for p in (0.1, 100.0):
        index = int(np.floor(p / 100.0 * n))
        if index <= 0:
            # original code would give empty slice; we define mean/std as 0.0
            top_vals = np.array([], dtype=sorted_vals.dtype)
        else:
            top_vals = sorted_vals[-index:]

        if top_vals.size == 0:
            top_mean = 0.0
            top_std = 0.0
        else:
            top_mean = float(top_vals.mean())
            top_std = float(top_vals.std())

        stats[f"top_{p}_percent_mean"] = top_mean
        stats[f"top_{p}_percent_std"] = top_std

    return stats

# ----------------------------- core processing ----------------------------- #

def process_image(image_path: Path) -> Dict[str, float]:
    """
    Process a single TIFF image and return a flat dictionary of statistics.
    """
    img = tifffile.imread(str(image_path))

    # Convert to float32 for all computations
    img = np.asarray(img, dtype=np.float32)

    stats: Dict[str, float] = {}

    # Basic info
    stats["filename"] = image_path.name
    ids = derive_ids_from_filename(image_path.name)
    stats["patient_id"] = ids["patient_id"]
    stats["roi_id"] = ids["roi_id"]

    # Interpret shape
    if img.ndim == 2:
        # Single-channel image: treat as (1, H, W)
        img = img[None, ...]
    elif img.ndim != 3:
        raise ValueError(f"Unsupported image shape {img.shape} for file {image_path}")

    n_channels, height, width = img.shape
    stats["n_channels"] = int(n_channels)
    stats["height"] = int(height)
    stats["width"] = int(width)

    # Total (all channels flattened)
    total_values = img.ravel()
    total_stats = compute_channel_stats(total_values)
    for key, value in total_stats.items():
        stats[f"total_{key}"] = value

    # Per-channel statistics
    for c in range(n_channels):
        channel_values = img[c].ravel()
        channel_stats = compute_channel_stats(channel_values)

        # Focus metric is computed on 2D channel slice
        focus = compute_focus_gradient_energy(img[c])
        channel_stats["focus_gradient_energy"] = focus

        for key, value in channel_stats.items():
            stats[f"channel_{c + 1}_{key}"] = value

    return stats


def collect_tiff_files(input_dir: Path) -> List[Path]:
    """
    Collect all TIFF files (*.tif, *.tiff) from a directory (non-recursive).
    """
    tiffs: List[Path] = []
    for ext in ("*.tif", "*.tiff"):
        tiffs.extend(sorted(input_dir.glob(ext)))
    return tiffs


# ----------------------------- script entrypoint ----------------------------- #

def main(input_dir: str, output_dir: str, num_workers: Optional[int] = None) -> None:
    """
    Compute extended image statistics for all TIFF files in input_dir and
    save the result as image_statistics_extended.csv in output_dir.

    Parameters
    ----------
    input_dir : str
        Directory containing input TIFF files.
    output_dir : str
        Directory where image_statistics_extended.csv will be saved.
    num_workers : int or None
        Number of parallel worker processes to use.
        If None, uses os.cpu_count().
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    tiff_files = collect_tiff_files(input_path)
    if not tiff_files:
        raise FileNotFoundError(f"No TIFF files found in {input_path}")

    print(f"Found {len(tiff_files)} TIFF files in {input_path}")

    # Determine number of workers
    if num_workers is None or num_workers <= 0:
        num_workers = os.cpu_count() or 1

    print(f"Using {num_workers} worker processes.")

    # Parallel processing with tqdm progress bar
    results: List[Dict[str, float]] = []
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        for stats in tqdm(
            executor.map(process_image, tiff_files),
            total=len(tiff_files),
            desc="Processing images"
        ):
            results.append(stats)

    df = pd.DataFrame(results)
    csv_path = output_path / "image_statistics_extended.csv"
    df.to_csv(csv_path, index=False)
    print(f"\nSaved extended image statistics to: {csv_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute extended image quality statistics for TIFF images.")
    parser.add_argument("input_dir", type=str, help="Directory containing input TIFF files.")
    parser.add_argument("output_dir", type=str, help="Directory where image_statistics_extended.csv will be saved.")
    parser.add_argument(
        "--num-workers",
        type=int,
        default=None,
        help="Number of worker processes to use (default: use all available cores)."
    )
    args = parser.parse_args()

    main(args.input_dir, args.output_dir, num_workers=args.num_workers)

