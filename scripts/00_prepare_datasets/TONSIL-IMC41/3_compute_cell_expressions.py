#!/usr/bin/env python3
"""
Compute cell expressions and final labels for TONSIL-IMC41 OPTIMAL_MC21,
with resumable, parallel processing.

- Input:
    * `datasets/TONSIL-IMC41/channels.txt`
    * `datasets/TONSIL-IMC41/OPTIMAL_MC21/temp_labels.csv` (contains 'fname', 'cell_x', 'cell_y', etc.)
    * Raw images: `datasets/TONSIL-IMC41/images/{fname}.tiff`
    * Raw masks:  `datasets/TONSIL-IMC41/OPTIMAL_MC21/segmentations/{fname}.npy`
- Creates:
    * `datasets/TONSIL-IMC41/OPTIMAL_MC21/tmp_labels/{fname}_labels.csv`
    * `datasets/TONSIL-IMC41/OPTIMAL_MC21/tmp_exprs/{fname}_expressions.csv`
- After all processed, concatenates to:
    * `datasets/TONSIL-IMC41/OPTIMAL_MC21/labels.csv`
    * `datasets/TONSIL-IMC41/OPTIMAL_MC21/cell_expressions.csv`

Uses `config.yaml` for base paths, `ProcessPoolExecutor` for parallelism, and `tqdm` for progress.
"""
import os
import numpy as np
import pandas as pd
import tifffile
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
from clearit.config import DATASETS_DIR

# Paths
DATASET = "TONSIL-IMC41"
OUT_ROOT = DATASETS_DIR / DATASET
OPTIMAL_DIR = OUT_ROOT / "OPTIMAL_MC21"
CHANNELS_FILE = OUT_ROOT / "channels.txt"
TEMP_LABELS = OPTIMAL_DIR / "temp_labels.csv"
IMG_DIR = OUT_ROOT / "images"
SEG_DIR = OPTIMAL_DIR / "segmentations"
TMP_LABEL_DIR = OPTIMAL_DIR / "tmp_labels"
TMP_EXPR_DIR  = OPTIMAL_DIR / "tmp_exprs"
FINAL_LABELS  = OPTIMAL_DIR / "labels.csv"
FINAL_EXPR    = OPTIMAL_DIR / "cell_expressions.csv"

# Ensure directories
for d in [TMP_LABEL_DIR, TMP_EXPR_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# Load channels and temp labels
df_temp = pd.read_csv(TEMP_LABELS)
channels = CHANNELS_FILE.read_text().splitlines()

# Processing function
def process_fname(fname):
    df_f = df_temp[df_temp['fname'] == fname]
    # Load segmentation mask
    mask_path = SEG_DIR / f"{fname}.npy"
    if not mask_path.exists():
        return pd.DataFrame(), pd.DataFrame()
    seg = np.load(mask_path)
    # Load image stack
    img_path = IMG_DIR / f"{fname}.tiff"
    image = tifffile.imread(img_path)  # shape CxHxW

    labs = []
    exprs = []
    for _, row in df_f.iterrows():
        x = int(row['cell_x']) - 1
        y = int(row['cell_y']) - 1
        area = int(row['cell_area'])
        label_val = row['label']
        cell_id = int(seg[y, x])
        if cell_id == 0:
            continue
        # centroid cell might not match, but use this cell_id
        mask_bool = (seg == cell_id)
        # record label info
        labs.append({
            'cell_id': cell_id,
            'fname': fname,
            'ROI': int(row['ROI']),
            'cell_x': x,
            'cell_y': y,
            'cellSize': area,
            'label': label_val
        })
        # record expressions
        expr = {'cell_id': cell_id, 'fname': fname}
        for i, ch in enumerate(channels):
            pix = image[i][mask_bool]
            expr[ch] = pix.sum() / area if area else 0
        exprs.append(expr)

    return pd.DataFrame(labs), pd.DataFrame(exprs)

# Determine worklist
fnames = df_temp['fname'].unique().tolist()
todo = []
for fname in fnames:
    if not (TMP_EXPR_DIR / f"{fname}_expressions.csv").exists():
        todo.append(fname)

# Parallel processing
def run_parallel():
    with ProcessPoolExecutor(max_workers=os.cpu_count()) as exe:
        futures = {exe.submit(process_fname, fn): fn for fn in todo}
        for fut in tqdm(as_completed(futures), total=len(futures), desc="Computing TONSIL expressions"):
            fn = futures[fut]
            lab_df, expr_df = fut.result()
            if not lab_df.empty:
                lab_df.to_csv(TMP_LABEL_DIR / f"{fn}_labels.csv", index=False)
                expr_df.to_csv(TMP_EXPR_DIR / f"{fn}_expressions.csv", index=False)

run_parallel()

# Combine and save final outputs
lab_files = sorted(TMP_LABEL_DIR.glob("*_labels.csv"))
expr_files= sorted(TMP_EXPR_DIR.glob("*_expressions.csv"))
if lab_files:
    pd.concat([pd.read_csv(f) for f in lab_files], ignore_index=True).to_csv(FINAL_LABELS, index=False)
    print(f"Saved labels.csv to {FINAL_LABELS}")
if expr_files:
    df_all = pd.concat([pd.read_csv(f) for f in expr_files], ignore_index=True)
    df_all = df_all[['cell_id','fname']+channels]
    df_all.to_csv(FINAL_EXPR, index=False)
    print(f"Saved cell_expressions.csv to {FINAL_EXPR}")
