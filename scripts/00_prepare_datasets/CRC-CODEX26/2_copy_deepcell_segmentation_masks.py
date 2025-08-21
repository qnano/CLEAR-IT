#!/usr/bin/env python3
"""
Copy and rename DeepCell segmentation masks for CRC-CODEX26.

- Reads raw .npz masks from:
  `raw_datasets/CRC-CODEX26/CellSighter Test Dataset CRC Multiplexed Images/cells`
- Writes renamed masks to:
  `datasets/CRC-CODEX26/DeepCell_MC14/segmentations/P{pid:02d}_ROI01.npz`
- Uses tqdm for progress reporting.

Paths derived from `config.yaml` via `clearit.config`.
"""
import shutil
from pathlib import Path
from tqdm import tqdm

# Load config paths
try:
    from clearit.config import RAW_DATASETS_DIR, DATASETS_DIR
except ImportError:
    raise RuntimeError(
        "Failed to import data paths from clearit.config. "
        "Ensure config.yaml is present and clearit.config can load it."
    )

# Constants
DATASET_NAME = "CRC-CODEX26"
# Raw cells directory (note spaces in folder name)
RAW_CELLS_DIR = (RAW_DATASETS_DIR
                 / DATASET_NAME
                 / "CellSighter Test Dataset CRC Multiplexed Images"
                 / "cells")
# Processed segmentation directory
OUT_SEG_DIR = (DATASETS_DIR
               / DATASET_NAME
               / "DeepCell_MC14"
               / "segmentations")

# Ensure output directory exists
OUT_SEG_DIR.mkdir(parents=True, exist_ok=True)

# Copy and rename .npz files with progress bar
npz_files = sorted(RAW_CELLS_DIR.glob("*.npz"))
if not npz_files:
    print(f"No segmentation masks found in {RAW_CELLS_DIR}")
else:
    for src in tqdm(npz_files, desc="Copying DeepCell masks"):
        # Filename like 'P01_something.npz' or similar; take leading Pxx
        stem = src.stem  # e.g. 'P01_cellmask'
        # Extract patient id after 'P'
        try:
            pid = int(stem.split('P')[-1][:2])
        except ValueError:
            raise ValueError(f"Cannot parse patient ID from filename: {src.name}")
        dst = OUT_SEG_DIR / f"P{pid:02d}_ROI01.npz"
        shutil.copy(src, dst)
    print(f"Copied {len(npz_files)} masks to {OUT_SEG_DIR}")
