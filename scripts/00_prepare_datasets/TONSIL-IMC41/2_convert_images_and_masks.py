#!/usr/bin/env python3
"""
Convert TIFF images and corresponding DeepCell segmentation masks for TONSIL-IMC41, renaming files
and updating the temporary labels.csv with new filenames.

- Reads:
  * `datasets/TONSIL-IMC41/channels.txt` for channel list
  * `datasets/TONSIL-IMC41/OPTIMAL_MC21/temp_labels.csv` for existing labels and ROIs
  * Raw images from `raw_datasets/TONSIL-IMC41/20230804_Tonsil_Matlab_Codes_V4/1_ImageData/*.ome.tiff`
  * Raw masks from `.../5_CP_pipeline/OUTPUTS/CellMasks/CellMasks{ROI:03d}.npy`
- Writes:
  * `datasets/TONSIL-IMC41/images/P{pid:02d}_ROI{new:02d}.tiff` with selected channels
  * `datasets/TONSIL-IMC41/OPTIMAL_MC21/segmentations/P{pid:02d}_ROI{new:02d}.npy`
  * Updates `temp_labels.csv` with `fname` = new filename

Filenames are assigned per patient by order in the temp_labels file. The first ROI for a patient → ROI01, second → ROI02, etc.

Uses `config.yaml` for base paths, `tifffile` for TIFF handling, and `tqdm` for progress.
"""
import shutil
import numpy as np
import tifffile
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from clearit.config import RAW_DATASETS_DIR, DATASETS_DIR

# Paths
DATASET = "TONSIL-IMC41"
RAW_ROOT = RAW_DATASETS_DIR / DATASET / "20230804_Tonsil_Matlab_Codes_V4"
OUT_ROOT = DATASETS_DIR / DATASET
CHANNELS_FILE = OUT_ROOT / "channels.txt"
TMP_LABELS = OUT_ROOT / "OPTIMAL_MC21" / "temp_labels.csv"
IMG_OUT = OUT_ROOT / "images"
SEG_OUT = OUT_ROOT / "OPTIMAL_MC21" / "segmentations"

# Ensure output directories
IMG_OUT.mkdir(parents=True, exist_ok=True)
SEG_OUT.mkdir(parents=True, exist_ok=True)

# Load channels
channels = CHANNELS_FILE.read_text().splitlines()

# Load temp labels and unique ROIs
df_labels = pd.read_csv(TMP_LABELS)
df_unique = df_labels[["patient_id","fname_old","ROI"]].drop_duplicates().reset_index(drop=True)

# Assign new filenames per patient
new_fname_map = {}
for pid, group in df_unique.groupby("patient_id"):
    for new_idx, (_, row) in enumerate(group.iterrows(), start=1):
        key = (pid, row["fname_old"], row["ROI"])
        new_fname_map[key] = f"P{int(pid):02d}_ROI{new_idx:02d}"

# Process each unique ROI with progress bar
for (pid, fname_old, roi), new_fname in tqdm(new_fname_map.items(), desc="Converting images and masks"):
    # Paths
    in_img = RAW_ROOT / "1_ImageData" / f"{fname_old}.ome.tiff"
    out_img = IMG_OUT / f"{new_fname}.tiff"
    in_mask = RAW_ROOT / "5_CP_pipeline" / "OUTPUTS" / "CellMasks" / f"CellMasks{int(roi):03d}.npy"
    out_mask = SEG_OUT / f"{new_fname}.npy"

    # Convert image if needed
    if not out_img.exists():
        selected = []
        with tifffile.TiffFile(in_img) as tif:
            for page in tif.pages:
                arr = page.asarray()
                name = page.tags.values()[13].value
                if name == "COX2(Er166Di)":
                    name = "p16(Er166Di)"
                if name in channels:
                    selected.append(arr)
        tifffile.imwrite(
            out_img,
            selected,
            photometric='minisblack',
            dtype=np.uint16,
            compression='zlib'
        )
    # Copy mask if exists
    if in_mask.exists() and not out_mask.exists():
        shutil.copy(in_mask, out_mask)

    # Update df_labels
    df_labels.loc[
        (df_labels['patient_id'] == pid) &
        (df_labels['fname_old'] == fname_old) &
        (df_labels['ROI'] == roi),
        'fname'
    ] = new_fname

# Save updated temp_labels.csv
df_labels.to_csv(TMP_LABELS, index=False)
print(f"Updated temp_labels.csv with new filenames at {TMP_LABELS}")
