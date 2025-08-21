#!/usr/bin/env python3
"""
Generate temporary labels.csv and class_names.csv for CRC-CODEX26 DeepCell_MC14.

- Reads label NPZs from:
  `raw_datasets/CRC-CODEX26/CellSighter Test Dataset CRC Multiplexed Images/cells2labels`
- Aggregates each file into DataFrame with columns [cell_id, label, fname]
  where `fname` is `P{pid:02d}_ROI01`.
- Saves `temp_labels.csv` under `datasets/CRC-CODEX26/DeepCell_MC14`.
- Reads mapping_id_to_cell_type_name.txt, evals to dict,
  saves `class_names.csv` alongside.
- Uses tqdm for progress reporting.
"""
import ast
import numpy as np
import pandas as pd
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
RAW_LABEL_DIR = ROOT_RAW / "cells2labels"
MAPPING_PATH = ROOT_RAW / "mapping_id_to_cell_type_name.txt"
OUT_DIR = DATASETS_DIR / DATASET_NAME / "DeepCell_MC14"
TMP_LABELS_CSV = OUT_DIR / "temp_labels.csv"
CLASS_NAMES_CSV = OUT_DIR / "class_names.csv"

# Ensure output directory
OUT_DIR.mkdir(parents=True, exist_ok=True)

def process_npz(file_path: Path) -> pd.DataFrame:
    # Patient ID from filename Pxx_...
    stem = file_path.stem  # e.g. 'P01_cellLabels'
    try:
        pid = int(stem.split('_')[0].lstrip('P'))
    except ValueError:
        raise ValueError(f"Cannot parse patient ID from filename: {file_path.name}")
    fname = f"P{pid:02d}_ROI01"

    arr = np.load(file_path)
    # determine key: 'data' or first
    key = 'data' if 'data' in arr else arr.files[0]
    labels = arr[key]

    # iterate, skip id 0 and label -1
    records = []
    for cid, lbl in enumerate(labels):
        if cid == 0 or lbl == -1:
            continue
        records.append({
            'cell_id': cid,
            'label': int(lbl),
            'fname': fname
        })
    return pd.DataFrame(records)

# Aggregate temp labels
all_dfs = []
for npz_path in tqdm(sorted(RAW_LABEL_DIR.glob('*.npz')), desc="Processing label NPZs"):
    df = process_npz(npz_path)
    all_dfs.append(df)

if all_dfs:
    temp_df = pd.concat(all_dfs, ignore_index=True)
    temp_df.to_csv(TMP_LABELS_CSV, index=False)
    print(f"temp_labels.csv saved to {TMP_LABELS_CSV}")
else:
    print(f"No .npz label files found in {RAW_LABEL_DIR}")

# Save class names
if not MAPPING_PATH.exists():
    raise FileNotFoundError(f"Mapping file not found: {MAPPING_PATH}")
with open(MAPPING_PATH, 'r') as f:
    mapping_dict = ast.literal_eval(f.read())
df_cn = pd.DataFrame({
    'label': list(mapping_dict.keys()),
    'name': list(mapping_dict.values())
})
df_cn.to_csv(CLASS_NAMES_CSV, index=False)
print(f"class_names.csv saved to {CLASS_NAMES_CSV}")
