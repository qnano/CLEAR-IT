#!/usr/bin/env python3
"""
Compute cell expressions for TNBC1-MxIF8, moving images
from raw to processed and supporting resumable, parallel runs.

- Reads raw images from `raw_datasets/TNBC1-MxIF8/images`
- Copies segmentation masks from raw to processed for completeness
- Reads segmentation masks from `datasets/TNBC1-MxIF8/TME-A_ML6/segmentations`
- Saves per-image expression CSVs to `datasets/TNBC1-MxIF8/tmp_expressions`
- Moves each processed image to `datasets/TNBC1-MxIF8/images`
- Ensures `channels.txt` is copied from raw to processed
- Combines all per-image CSVs into final `cell_expressions.csv`

Paths derived from `config.yaml` via `clearit.config`.
"""
import argparse
import os
import shutil
import numpy as np
import pandas as pd
import tifffile
from pathlib import Path
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed

# Load data paths from central config
try:
    from clearit.config import RAW_DATASETS_DIR, DATASETS_DIR
except ImportError as e:
    raise RuntimeError(
        "Failed to import data paths from clearit.config. "
        "Ensure `config.yaml` is present in the project root and loaded by `clearit.config`."
    ) from e

# Dataset-specific constants
DATASET_NAME = "TNBC1-MxIF8"
RAW_IMAGES_DIR = RAW_DATASETS_DIR / DATASET_NAME / "images"
PROCESSED_IMAGES_DIR = DATASETS_DIR / DATASET_NAME / "images"
RAW_SEG_DIR = RAW_DATASETS_DIR / DATASET_NAME / "TME-Analyzer" / "segmentations"
PROCESSED_SEG_DIR = DATASETS_DIR / DATASET_NAME / "TME-A_ML6" / "segmentations"
TMP_DIR = DATASETS_DIR / DATASET_NAME / "tmp_expressions"
OUTPUT_CSV = DATASETS_DIR / DATASET_NAME / "TME-A_ML6" / "cell_expressions.csv"


def process_single_image(fname: str, df_labels: pd.DataFrame, channels: list) -> pd.DataFrame:
    """
    Compute expression dataframe for a single image basename.
    """
    image_path = RAW_IMAGES_DIR / f"{fname}.tiff"
    mask_path = PROCESSED_SEG_DIR / f"{fname}.npz"

    image = tifffile.imread(image_path)
    seg = np.load(mask_path)['image']

    df_img = df_labels[df_labels['fname'] == fname]

    data = {'cell_id': [], 'fname': []}
    for ch in channels:
        data[ch] = []

    for _, row in df_img.iterrows():
        cid = int(row['cell_id'])
        area = float(row['cell_area'])
        y, x = int(row['cell_y']), int(row['cell_x'])

        if seg[y, x] != cid:
            continue
        mask = (seg == cid)

        data['cell_id'].append(cid)
        data['fname'].append(fname)

        for i, ch in enumerate(channels):
            pix = image[i][mask]
            total = np.sum(pix)
            data[ch].append(total / area if area > 0 else 0)

    return pd.DataFrame(data)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--workers', type=int, default=os.cpu_count(),
                        help='Number of parallel workers (default: all cores)')
    args = parser.parse_args()
    workers = args.workers

    # Create necessary directories
    for d in [PROCESSED_IMAGES_DIR, TMP_DIR, PROCESSED_SEG_DIR]:
        d.mkdir(parents=True, exist_ok=True)

    # Ensure channels.txt in processed directory
    raw_channels = RAW_DATASETS_DIR / DATASET_NAME / "channels.txt"
    if not raw_channels.exists():
        raise FileNotFoundError(f"channels.txt not found in raw datasets: {raw_channels}")
    proc_channels = DATASETS_DIR / DATASET_NAME / "channels.txt"
    if not proc_channels.exists():
        shutil.copy(raw_channels, proc_channels)
        print(f"Copied channels.txt to processed folder: {proc_channels}")

    # Copy segmentation masks for completeness
    if RAW_SEG_DIR.exists():
        for f in RAW_SEG_DIR.iterdir():
            if f.suffix.lower() == ".npz" and f.name.endswith("_cell_segmentation.npz"):
                dest_name = f.name.replace("_cell_segmentation", "")
                dest_path = PROCESSED_SEG_DIR / dest_name
                if not dest_path.exists():
                    shutil.copy(f, dest_path)
        print(f"Copied segmentation masks to processed folder: {PROCESSED_SEG_DIR}")

    # Load labels and channels
    labels_csv = DATASETS_DIR / DATASET_NAME / "TME-A_ML6" / "labels.csv"
    df_labels = pd.read_csv(labels_csv)
    channels = proc_channels.read_text().splitlines()

    # Identify processed vs to-do images
    all_images = sorted(RAW_IMAGES_DIR.glob('*.tiff'))
    to_process = []
    for img_file in all_images:
        fname = img_file.stem
        tmp_file = TMP_DIR / f"{fname}.csv"
        dst_img = PROCESSED_IMAGES_DIR / img_file.name
        if tmp_file.exists():
            if not dst_img.exists():
                img_file.rename(dst_img)
            continue
        to_process.append(fname)

    # Parallel processing
    if to_process:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            future_to_fname = {executor.submit(process_single_image, fname, df_labels, channels): fname
                               for fname in to_process}
            for future in tqdm(as_completed(future_to_fname), total=len(future_to_fname),
                               desc="Computing per-image expressions"):
                fname = future_to_fname[future]
                df_expr = future.result()
                tmp_file = TMP_DIR / f"{fname}.csv"
                df_expr.to_csv(tmp_file, index=False)
                # Move image
                src_img = RAW_IMAGES_DIR / f"{fname}.tiff"
                dest_img = PROCESSED_IMAGES_DIR / f"{fname}.tiff"
                src_img.rename(dest_img)

    # Combine all per-image CSVs
    tmp_files = sorted(TMP_DIR.glob('*.csv'))
    if not tmp_files:
        print("No expression files found in tmp_expressions. Nothing to save.")
        return

    df_all = pd.concat([pd.read_csv(f) for f in tmp_files], ignore_index=True)
    df_all.to_csv(OUTPUT_CSV, index=False)
    print(f"Combined cell expressions saved to {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
