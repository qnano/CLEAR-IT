#!/usr/bin/env python3
"""
Generate `labels.csv` and `cell_expressions.csv` for the TNBC2-MIBI44 DeepCell_MC17 dataset,
with resumable, parallel processing and per-patient temporary CSVs.

Paths derived from `config.yaml` via `clearit.config`.
"""
import argparse
import os
import numpy as np
import pandas as pd
import tifffile
from scipy.ndimage import center_of_mass
from pathlib import Path
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed

# Load data paths from central config
try:
    from clearit.config import RAW_DATASETS_DIR, DATASETS_DIR
except ImportError as e:
    raise RuntimeError(
        "Failed to import data paths from clearit.config. "
        "Ensure config.yaml is present and clearit.config can load it."
    ) from e

# Constants and paths
DATASET_NAME = "TNBC2-MIBI"
RAW_DIR = RAW_DATASETS_DIR / DATASET_NAME
OUT_PARENT = DATASETS_DIR / DATASET_NAME

# DeepCell annotation directories
CELL_DATA_CSV = RAW_DIR / "TNBC_shareCellData" / "cellData.csv"
SEG_DIR = OUT_PARENT / "TNBC2-MIBI44" / "DeepCell_MC17" / "segmentations"
IMAGE_DIR = OUT_PARENT / "TNBC2-MIBI44" / "images"
TMP_DIR = OUT_PARENT / "TNBC2-MIBI44" / "DeepCell_MC17" / "tmp_expressions"
OUTPUT_DIR = OUT_PARENT / "TNBC2-MIBI44" / "DeepCell_MC17"
CHANNELS_FILE = OUT_PARENT / "TNBC2-MIBI44" / "channels.txt"
CLASS_NAMES_OUTPUT = OUTPUT_DIR / "class_names.csv"
LABELS_OUTPUT = OUTPUT_DIR / "labels.csv"
EXPRESSIONS_OUTPUT = OUTPUT_DIR / "cell_expressions.csv"

# Class names for DeepCell
CLASS_NAMES = [
    "Unidentified", "Tregs", "CD4 T", "CD8 T", "CD3 T", "NK", "B", "Neutrophils", 
    "Macrophages", "DC", "DC/Mono", "Mono/Neu", "Other immune", "Endothelial", 
    "Mesenchymal-like", "Tumor", "Keratin-positive tumor"
]


def compute_centroids(seg):
    """Compute valid centroids for each cell ID in segmentation mask."""
    ids = np.unique(seg)
    # skip background and border labels
    ids = [i for i in ids if i > 1]
    centroids = center_of_mass(seg, labels=seg, index=ids)
    return {int(cid): (int(round(c[1])), int(round(c[0])))
            for cid, c in zip(ids, centroids)
            if not np.isnan(c).any() and np.isfinite(c).all()}


def process_patient(pid, df_cell_data, channels):
    """Process one patient: compute labels and expressions, return two DataFrames."""
    # Load segmentation mask
    seg_path = SEG_DIR / f"P{pid:02d}_ROI01.npz"
    if not seg_path.exists():
        return pd.DataFrame(), pd.DataFrame()
    seg = np.load(seg_path)['arr_0']

    # Compute centroids
    centroids = compute_centroids(seg)

    # Load image
    img_path = IMAGE_DIR / f"P{pid:02d}_ROI01.tiff"
    image = tifffile.imread(img_path)

    # Build expression dict per cell
    expr = {cid: [] for cid in centroids}
    for i, ch in enumerate(channels):
        img = image[i]
        for cid in list(expr.keys()):
            mask = (seg == cid)
            total = np.sum(img[mask])
            area = np.sum(mask)
            expr[cid].append(total / area if area else 0)

    # Build DataFrames
    labels = []
    expressions = []
    subset = df_cell_data[df_cell_data['SampleID'] == pid]
    for _, row in subset.iterrows():
        cid = int(row['cellLabelInImage'])
        if cid not in centroids:
            continue
        x, y = centroids[cid]
        label = (0 if row['Group']==1 else 
                 (row['immuneGroup'] if row['Group']==2 else row['Group']+10))
        fname = f"P{pid:02d}_ROI01"
        labels.append({
            'cell_id': cid,
            'cell_area': row['cellSize'],
            'cell_x': x,
            'cell_y': y,
            'label': label,
            'fname': fname
        })
        expressions.append({ 'cell_id': cid, 'fname': fname, **{ch: expr[cid][i] for i,ch in enumerate(channels)} })

    return pd.DataFrame(labels), pd.DataFrame(expressions)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--workers', type=int, default=os.cpu_count(),
                        help='Number of parallel workers')
    args = parser.parse_args()
    workers = args.workers

    # Create directories
    for d in [TMP_DIR]: d.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load cell data
    df_cell = pd.read_csv(CELL_DATA_CSV)

    # Load channels
    channels = CHANNELS_FILE.read_text().splitlines()

    # Prepare patient list and resume check
    pids = sorted(df_cell['SampleID'].unique())
    to_process = []
    for pid in pids:
        tmp_lab = TMP_DIR / f"P{pid:02d}_labels.csv"
        tmp_exp = TMP_DIR / f"P{pid:02d}_expressions.csv"
        if tmp_lab.exists() and tmp_exp.exists():
            continue
        to_process.append(pid)

    # Parallel execution
    if to_process:
        with ProcessPoolExecutor(max_workers=workers) as exe:
            futures = {exe.submit(process_patient, pid, df_cell, channels): pid for pid in to_process}
            for fut in tqdm(as_completed(futures), total=len(futures), desc="Processing patients"):
                pid = futures[fut]
                lab_df, exp_df = fut.result()
                # Save per-patient CSVs
                if not lab_df.empty:
                    lab_df.to_csv(TMP_DIR / f"P{pid:02d}_labels.csv", index=False)
                    exp_df.to_csv(TMP_DIR / f"P{pid:02d}_expressions.csv", index=False)

    # Combine all
    lab_files = sorted(TMP_DIR.glob("*labels.csv"))
    exp_files = sorted(TMP_DIR.glob("*expressions.csv"))
    if lab_files:
        pd.concat([pd.read_csv(f) for f in lab_files], ignore_index=True).to_csv(LABELS_OUTPUT, index=False)
        print(f"Labels CSV saved to {LABELS_OUTPUT}")
    if exp_files:
        pd.concat([pd.read_csv(f) for f in exp_files], ignore_index=True).to_csv(EXPRESSIONS_OUTPUT, index=False)
        print(f"Cell expressions CSV saved to {EXPRESSIONS_OUTPUT}")

    # Save class names
    dfcn = pd.DataFrame({'name': CLASS_NAMES, 'label': list(range(len(CLASS_NAMES)))})
    dfcn.to_csv(CLASS_NAMES_OUTPUT, index=False)
    print(f"Class names CSV saved to {CLASS_NAMES_OUTPUT}")

if __name__ == "__main__":
    main()
