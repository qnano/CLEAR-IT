#!/usr/bin/env python3
"""
Generate `labels.csv` and `cell_expressions.csv` for the CRC-CODEX26 DeepCell_MC14 dataset,
with resumable, parallel processing and per-file intermediate CSVs.

Paths derived from `config.yaml` via `clearit.config`:
- RAW_DATASETS_DIR/dataset/CellSighter Test Dataset CRC Multiplexed Images/cells2labels (temp_labels)
- DATASETS_DIR/dataset/DeepCell_MC14/segmentations (masks)
- DATASETS_DIR/dataset/images (tiffs)
- DATASETS_DIR/dataset/channels.txt (channel names)

Temporary outputs:
- DATASETS_DIR/dataset/DeepCell_MC14/tmp_labels/{fname}_labels.csv
- DATASETS_DIR/dataset/DeepCell_MC14/tmp_exprs/{fname}_expressions.csv

Final outputs:
- DATASETS_DIR/dataset/DeepCell_MC14/labels.csv
- DATASETS_DIR/dataset/DeepCell_MC14/cell_expressions.csv
"""
import os
import numpy as np
import pandas as pd
import tifffile
from scipy.ndimage import center_of_mass
from pathlib import Path
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed

# Load central config paths
try:
    from clearit.config import DATASETS_DIR, RAW_DATASETS_DIR
except ImportError:
    raise RuntimeError(
        "Failed to import paths from clearit.config. "
        "Ensure config.yaml is present and clearit.config can load it."
    )

# Dataset constants
DATASET = "CRC-CODEX26"
OUT_ROOT = DATASETS_DIR / DATASET
TMP_LABEL_DIR = OUT_ROOT / "DeepCell_MC14" / "tmp_labels"
TMP_EXPR_DIR  = OUT_ROOT / "DeepCell_MC14" / "tmp_exprs"
SEG_DIR       = OUT_ROOT / "DeepCell_MC14" / "segmentations"
IMG_DIR       = OUT_ROOT / "images"
CHANNELS_FILE = OUT_ROOT / "channels.txt"
LABELS_TEMP   = OUT_ROOT / "DeepCell_MC14" / "temp_labels.csv"
LABELS_OUT    = OUT_ROOT / "DeepCell_MC14" / "labels.csv"
EXPRS_OUT     = OUT_ROOT / "DeepCell_MC14" / "cell_expressions.csv"

# Ensure directories
for d in [TMP_LABEL_DIR, TMP_EXPR_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# Load channels
if not CHANNELS_FILE.exists():
    raise FileNotFoundError(f"Channels file not found: {CHANNELS_FILE}")
channels = CHANNELS_FILE.read_text().splitlines()

# Load temporary labels
if not LABELS_TEMP.exists():
    raise FileNotFoundError(f"Temporary labels not found: {LABELS_TEMP}")
df_temp = pd.read_csv(LABELS_TEMP)

# Helper: compute centroids
def compute_centroids(seg):
    ids = np.unique(seg)
    ids = ids[ids>0]
    cents = center_of_mass(seg, labels=seg, index=ids)
    return {int(cid): (int(round(c[1])), int(round(c[0])))
            for cid,c in zip(ids,cents)
            if not np.isnan(c).any() and np.isfinite(c).all()}

# Per-file processing
def process_fname(fname_group):
    fname, group = fname_group
    seg_path = SEG_DIR / f"{fname}.npz"
    if not seg_path.exists():
        return pd.DataFrame(), pd.DataFrame()
    seg = np.load(seg_path)['data']
    cents = compute_centroids(seg)
    img_path = IMG_DIR / f"{fname}.tiff"
    image = tifffile.imread(img_path)

    labels = []
    exprs = []
    for _, row in group.iterrows():
        cid = int(row['cell_id'])
        if cid not in cents: continue
        x,y = cents[cid]
        area = int((seg==cid).sum())
        labels.append({
            'cell_id': cid,
            'cell_area': area,
            'cell_x': x,
            'cell_y': y,
            'label': row['label'],
            'fname': fname
        })
        vals = [(image[i][seg==cid].sum()/area if area else 0) for i in range(len(channels))]
        record = {'cell_id': cid, 'fname': fname}
        record.update({ch: vals[i] for i,ch in enumerate(channels)})
        exprs.append(record)

    df_lab = pd.DataFrame(labels)
    df_expr= pd.DataFrame(exprs)
    return df_lab, df_expr

# Determine worklist by fname
all_fnames = df_temp['fname'].unique()
todo = []
for fname in all_fnames:
    if not (TMP_LABEL_DIR/f"{fname}_labels.csv").exists():
        todo.append(fname)

# Parallel execution
workers = os.cpu_count()
if todo:
    with ProcessPoolExecutor(max_workers=workers) as exe:
        futures = {exe.submit(process_fname, (fname, df_temp[df_temp['fname']==fname])): fname for fname in todo}
        for fut in tqdm(as_completed(futures), total=len(futures), desc="Processing DeepCell files"):
            fname = futures[fut]
            df_lab, df_expr = fut.result()
            if not df_lab.empty:
                df_lab.to_csv(TMP_LABEL_DIR/f"{fname}_labels.csv", index=False)
                df_expr.to_csv(TMP_EXPR_DIR/f"{fname}_expressions.csv", index=False)

# Combine and save outputs
lab_files = sorted(TMP_LABEL_DIR.glob("*_labels.csv"))
expr_files= sorted(TMP_EXPR_DIR.glob("*_expressions.csv"))
if lab_files:
    pd.concat([pd.read_csv(f) for f in lab_files], ignore_index=True).to_csv(LABELS_OUT, index=False)
    print(f"Labels CSV saved to {LABELS_OUT}")
if expr_files:
    df_all = pd.concat([pd.read_csv(f) for f in expr_files], ignore_index=True)
    # reorder columns
    df_all = df_all[['cell_id','fname']+channels]
    df_all.to_csv(EXPRS_OUT, index=False)
    print(f"Cell expressions CSV saved to {EXPRS_OUT}")
