#!/usr/bin/env python3
"""
Convert DeepCell_MC17 segmentation masks from TIFF to compressed NPZ for TNBC2-MIBI44.

- Reads raw TIFF masks from `raw_datasets/TNBC2-MIBI/TNBC_shareCellData`
- Writes `.npz` files to `datasets/TNBC2-MIBI/TNBC2-MIBI44/DeepCell_MC17/segmentations`
- Filename format: `P{patient_id:02d}_ROI01.npz`
- Uses a tqdm progress bar for bulk conversion.

Paths derived from `config.yaml` via `clearit.config`.
"""
import numpy as np
import tifffile
from pathlib import Path
from tqdm import tqdm

# Load data paths from central config
try:
    from clearit.config import RAW_DATASETS_DIR, DATASETS_DIR
except ImportError:
    raise RuntimeError(
        "Failed to import data paths from clearit.config. "
        "Ensure `config.yaml` is present and `clearit.config` can load it."
    )

# Dataset-specific paths
DATASET_NAME = "TNBC2-MIBI"
INPUT_DIR = RAW_DATASETS_DIR / DATASET_NAME / "TNBC_shareCellData"
OUTPUT_DIR = DATASETS_DIR / DATASET_NAME / "TNBC2-MIBI44" / "DeepCell_MC17" / "segmentations"


def main():
    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Find all DeepCell TIFFs
    tiff_paths = sorted(INPUT_DIR.glob("*_labeledcellData.tiff"))
    if not tiff_paths:
        print(f"No DeepCell TIFF masks found in {INPUT_DIR}")
        return

    # Convert each with progress bar
    for src_path in tqdm(tiff_paths, desc="Converting DeepCell masks"):
        fname = src_path.name
        # Parse patient ID from filename after last 'p'
        try:
            pid = int(fname.split('p')[-1].split('_')[0])
        except ValueError:
            raise ValueError(f"Cannot parse patient ID from filename: {fname}")

        dest_name = f"P{pid:02d}_ROI01.npz"
        dest_path = OUTPUT_DIR / dest_name

        # Read TIFF and save NPZ
        mask = tifffile.imread(src_path)
        np.savez_compressed(dest_path, mask)

    # Summary
    print(f"Converted {len(tiff_paths)} DeepCell masks to {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
