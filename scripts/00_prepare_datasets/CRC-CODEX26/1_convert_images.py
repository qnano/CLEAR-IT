#!/usr/bin/env python3
"""
Convert CRC-CODEX26 NPZ antibody stacks to multichannel TIFFs,
renaming and reordering to match the CLEAR-IT convention.

- Reads raw NPZs from:
  `raw_datasets/CRC-CODEX26/CellSighter Test Dataset CRC Multiplexed Images/data/antibodies`
- Copies channel list:
  `channels_codex_CRC_celltune.txt` → `datasets/CRC-CODEX26/channels.txt`
- Converts each `.npz`:
    * Load array under key 'data' (shape H×W×C)
    * Move axis to C×H×W
    * Rename `Pxx_A_FOVy` → `Pxx_ROIy` (zero-padded)
    * Save as compressed TIFF under `datasets/CRC-CODEX26/images`
- Uses tqdm for progress reporting
"""
import shutil
import numpy as np
import tifffile
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
ROOT_RAW = RAW_DATASETS_DIR / DATASET_NAME / "CellSighter Test Dataset CRC Multiplexed Images"
RAW_IMAGES = ROOT_RAW / "data" / "antibodies"
CHANNELS_SRC = ROOT_RAW / "channels_codex_CRC_celltune.txt"

OUT_ROOT = DATASETS_DIR / DATASET_NAME
OUT_IMAGES = OUT_ROOT / "images"
OUT_CHANNELS = OUT_ROOT / "channels.txt"

# Ensure output directories
OUT_IMAGES.mkdir(parents=True, exist_ok=True)

# Copy channel names file
if not CHANNELS_SRC.exists():
    raise FileNotFoundError(f"Channel list not found: {CHANNELS_SRC}")
shutil.copy(CHANNELS_SRC, OUT_CHANNELS)
print(f"Copied channel list to {OUT_CHANNELS}")

# Convert NPZ stacks to TIFFs
npz_files = sorted(RAW_IMAGES.glob('*.npz'))
if not npz_files:
    print(f"No NPZ files found in {RAW_IMAGES}")
else:
    for npz_path in tqdm(npz_files, desc="Converting CODEX images"):
        stem = npz_path.stem  # e.g. 'P01_A_FOV1'
        pid = stem.split('_')[0]  # 'P01'
        out_name = f"{pid}_ROI01.tiff"
        out_path = OUT_IMAGES / out_name

        # Load data array
        arr = np.load(npz_path)
        # default key 'data', else first available
        key = 'data' if 'data' in arr else arr.files[0]
        data = arr[key]  # shape: H x W x C
        if data.ndim != 3:
            raise ValueError(f"Unexpected array shape {data.shape} in {npz_path.name}")
        # Reorder to C x H x W
        img = np.moveaxis(data, 2, 0)

        # Save as compressed uint16 TIFF
        tifffile.imwrite(
            out_path,
            img,
            photometric='minisblack',
            dtype='uint16',
            compression='zlib'
        )
    print(f"Processed {len(npz_files)} images into {OUT_IMAGES}")
