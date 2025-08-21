#!/usr/bin/env python3
"""
Convert TME-A_ML6 segmentation masks from TIFF to compressed NPZ for TNBC2-MIBI.

- Reads raw TIFF masks from `raw_datasets/TNBC2-MIBI/TME-Analyzer/segmentations`
- Writes `.npz` files to `datasets/TNBC2-MIBI/TNBC2-MIBI8/TME-A_ML6/segmentations`
- Filename format: `P{patient_id:02d}_ROI01.npz`
- Uses a tqdm progress bar for resumable bulk conversion.

Paths derived from `config.yaml` via `clearit.config`.
"""
import numpy as np
import tifffile
from pathlib import Path
from tqdm import tqdm

# Load data paths from central config
try:
    from clearit.config import RAW_DATASETS_DIR, DATASETS_DIR
except ImportError as e:
    raise RuntimeError(
        "Failed to import data paths from clearit.config. "
        "Ensure `config.yaml` is present and `clearit.config` can load it." 
    ) from e

# Dataset-specific paths
DATASET_NAME = "TNBC2-MIBI"
INPUT_DIR = RAW_DATASETS_DIR / DATASET_NAME / "TME-Analyzer" / "segmentations"
OUTPUT_DIR = DATASETS_DIR / DATASET_NAME / "TNBC2-MIBI8" / "TME-A_ML6" / "segmentations"


def main():
    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Collect all cell segmentation TIFFs
    tiff_paths = sorted(INPUT_DIR.glob("*_cell_segmentation.tif"))
    if not tiff_paths:
        print(f"No TIFF masks found in {INPUT_DIR}")
        return

    # Convert each with progress bar
    for src_path in tqdm(tiff_paths, desc="Converting TME-A_ML6 masks"):
        # Extract patient ID from filename
        stem = src_path.stem  # e.g. '...Point{pid}_cell_segmentation'
        try:
            pid = int(stem.split('Point')[-1].split('_')[0])
        except ValueError:
            raise ValueError(f"Cannot parse patient ID from filename: {src_path.name}")

        dest_name = f"P{pid:02d}_ROI01.npz"
        dest_path = OUTPUT_DIR / dest_name

        # Read TIFF and save NPZ
        mask = tifffile.imread(src_path)
        np.savez_compressed(dest_path, mask)

    # Summary
    print(f"Converted {len(tiff_paths)} masks and saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
