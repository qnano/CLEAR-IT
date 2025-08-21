#!/usr/bin/env python3
"""
Generate temporary labels.csv, class_names.csv, and channels.txt for TONSIL-IMC41 OPTIMAL_MC21.

- Reads CSV/FCS/XLSX under:
  `raw_datasets/TONSIL-IMC41/20230804_Tonsil_Matlab_Codes_V4`
- Processes:
    * `2_image_extraction/Tonsil_EPCAM_Clusters_V2.csv`
    * `fcs_files/Combined_dataset.fcs`
    * `2_image_extraction/Cluster_Assignments_and_Merges_TONSIL.xlsx` (sheet "TIER 2")
- Merges, renames, computes `fname_old` for multiple ROIs per patient.
- Saves:
    * `datasets/TONSIL-IMC41/OPTIMAL_MC21/temp_labels.csv` (subset of columns)
    * `datasets/TONSIL-IMC41/OPTIMAL_MC21/class_names.csv`
    * `datasets/TONSIL-IMC41/channels.txt`

Uses `config.yaml` for base paths; no CLI flags.
"""
import numpy as np
import pandas as pd
import fcsparser
from pathlib import Path

# Load config paths
from clearit.config import RAW_DATASETS_DIR, DATASETS_DIR

# Constants
DATASET = "TONSIL-IMC41"
RAW_ROOT = RAW_DATASETS_DIR / DATASET / "20230804_Tonsil_Matlab_Codes_V4"
OUT_ROOT = DATASETS_DIR / DATASET
OPTIMAL_DIR = OUT_ROOT / "OPTIMAL_MC21"
TMP_LABEL_CSV = OPTIMAL_DIR / "temp_labels.csv"
CLASS_NAMES_CSV = OPTIMAL_DIR / "class_names.csv"
CHANNELS_TXT = OUT_ROOT / "channels.txt"

# Ensure output dirs
OPTIMAL_DIR.mkdir(parents=True, exist_ok=True)

# Load sources
csv_path = RAW_ROOT / "2_image_extraction" / "Tonsil_EPCAM_Clusters_V2.csv"
fcs_path = RAW_ROOT / "fcs_files" / "Combined_dataset.fcs"
xlsx_path = RAW_ROOT / "2_image_extraction" / "Cluster_Assignments_and_Merges_TONSIL.xlsx"

# Process labels CSV
df_lbl = pd.read_csv(csv_path)
df_lbl = df_lbl[["ROI_number","ConsensusClusteringAssignments","Centroid_X","Centroid_Y"]]
df_lbl[["ROI_number","Centroid_X","Centroid_Y"]] = df_lbl[["ROI_number","Centroid_X","Centroid_Y"]].astype(int)

# Process expressions FCS
_, df_expr = fcsparser.parse(fcs_path)
cols_int = ["Batch_number","Patient_number","Area","ROI_number","Centroid_X","Centroid_Y"]
df_expr[cols_int] = df_expr[cols_int].astype(int)
df_expr = df_expr.drop(columns=["Pathology_index","Circularity","Eccentricity"])

# Merge labels + expressions
df_merge = pd.merge(df_expr, df_lbl, on=["ROI_number","Centroid_X","Centroid_Y"])

# Rename columns
df_merge = df_merge.rename(columns={
    "Area": "cell_area",
    "Centroid_X": "cell_x",
    "Centroid_Y": "cell_y",
    "ROI_number": "ROI",
    "Patient_number": "patient_id",
    "Batch_number": "batch_id"
})

# Compute fname_old per ROI
def make_filename(x):
    BATCH = (x - 1) // 2 + 1
    if x % 2 == 1:
        ROI, se = 3, "START"
    else:
        ROI = 4 if BATCH == 5 else 6
        se = "END"
    return f"BATCH{BATCH:03}_ROI{ROI:03}_TONSIL_{se}"

df_merge["fname_old"] = df_merge["ROI"].apply(make_filename)

# Process mapping XLSX
df_map_full = pd.read_excel(xlsx_path, sheet_name="TIER 2")
df_map_full = df_map_full.rename(columns={"cSOM_s": "ConsensusClusteringAssignments"})
df_map = df_map_full[["ConsensusClusteringAssignments","Tier2ClusterAssignment","Name"]]
df_map["label"] = df_map["Tier2ClusterAssignment"] - 1

# Merge with mapping
df_merge = pd.merge(df_merge, df_map.drop(columns=["Tier2ClusterAssignment","Name"]), on="ConsensusClusteringAssignments")

# Select only required columns
cols_to_keep = [
    "cell_area",
    "cell_x",
    "cell_y",
    "ROI",
    "patient_id",
    "batch_id",
    "ConsensusClusteringAssignments",
    "fname_old",
    "label"
]
df_out = df_merge[cols_to_keep]

# Save temp_labels.csv
df_out.to_csv(TMP_LABEL_CSV, index=False)
print(f"Saved temp_labels.csv to {TMP_LABEL_CSV}")

# Save class_names.csv (label,name)
df_cn = df_map[["label","Name"]].drop_duplicates().sort_values("label")
df_cn.to_csv(CLASS_NAMES_CSV, index=False)
print(f"Saved class_names.csv to {CLASS_NAMES_CSV}")

# Save channels.txt: first 41 columns of df_expr
channel_cols = df_expr.columns[:41]
with CHANNELS_TXT.open('w') as f:
    for col in channel_cols:
        f.write(f"{col}\n")
print(f"Saved channels.txt to {CHANNELS_TXT}")
