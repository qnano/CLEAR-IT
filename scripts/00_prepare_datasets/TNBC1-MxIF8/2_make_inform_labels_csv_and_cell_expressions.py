#!/usr/bin/env python3
"""
Generate `labels.csv` and `cell_expressions.csv` for the TNBC1-MxIF8 dataset
using the inForm multiclass (MC7) and multilabel (ML6) label sets.

Paths are derived from the global `config.yaml` via `clearit.config`.

Usage:
    python 2_make_inform_labels_csv_and_cell_expressions.py
"""
import pandas as pd
from pathlib import Path
from tqdm import tqdm

# Load data paths from central config
try:
    from clearit.config import RAW_DATASETS_DIR, DATASETS_DIR
except ImportError as e:
    raise RuntimeError(
        "Failed to import data paths from clearit.config. "
        "Make sure you have a valid `config.yaml` at the project root and "
        "that `clearit/config.py` loads it correctly."
    ) from e

# Dataset-specific constants
DATASET_NAME = "TNBC1-MxIF8"
INFORM_SUBPATH = Path("inForm") / "tables"
OUTPUT_PARENT = DATASETS_DIR / DATASET_NAME
MC7_SUBDIR = "inForm_MC7"
ML6_SUBDIR = "inForm_ML6"

# Label mappings and orders
MULTICLASS_MAPPING = {
    "other": 0,
    "CK": 1,
    "CD3": 2,
    "CD3 CD8": 3,
    "CD20": 4,
    "CD56": 5,
    "CD68": 6,
}
MULTICLASS_ORDER = ["other", "CK", "CD3", "CD3 CD8", "CD20", "CD56", "CD68"]
MULTICLASS_TO_MULTILABEL = {
    0: [0, 0, 0, 0, 0, 0],
    1: [1, 0, 0, 0, 0, 0],
    2: [0, 1, 0, 0, 0, 0],
    3: [0, 1, 1, 0, 0, 0],
    4: [0, 0, 0, 1, 0, 0],
    5: [0, 0, 0, 0, 1, 0],
    6: [0, 0, 0, 0, 0, 1],
}
MULTILABEL_ORDER = ["CK+", "CD3+", "CD8+", "CD20+", "CD56+", "CD68+"]


def process_csv_file(file_path: Path):
    """
    Read a single inForm cell segment CSV and return
    (labels_df, expressions_df) for that file.
    """
    df = pd.read_csv(file_path)
    fname = file_path.stem.replace("_cell_seg_data", "")

    # --- Prepare labels (MC7) ---
    labels_rename = {
        "Cell ID": "cell_id",
        "Entire Cell Area (pixels)": "cell_area",
        "Cell X Position": "cell_x",
        "Cell Y Position": "cell_y",
        "Phenotype": "label",
    }
    df_labels = df.rename(columns=labels_rename)
    # Filter out only known phenotypes
    df_labels = df_labels[df_labels["label"].isin(MULTICLASS_MAPPING.keys())].copy()
    df_labels["label"] = df_labels["label"].map(MULTICLASS_MAPPING)
    df_labels["fname"] = fname
    df_labels = df_labels[["cell_id", "cell_area", "cell_x", "cell_y", "label", "fname"]]

    # --- Prepare expressions ---
    expr_rename = {
        "Cell ID": "cell_id",
        "Entire Cell DAPI Mean (Normalized Counts, Total Weighting)": "DAPI",
        "Entire Cell Opal 690 Mean (Normalized Counts, Total Weighting)": "CK",
        "Entire Cell Opal 520 Mean (Normalized Counts, Total Weighting)": "CD3",
        "Entire Cell Opal 540 Mean (Normalized Counts, Total Weighting)": "CD68",
        "Entire Cell Opal 570 Mean (Normalized Counts, Total Weighting)": "CD8",
        "Entire Cell Opal 620 Mean (Normalized Counts, Total Weighting)": "CD56",
        "Entire Cell Opal 650 Mean (Normalized Counts, Total Weighting)": "CD20",
        "Entire Cell Autofluorescence Mean (Normalized Counts, Total Weighting)": "background",
    }
    df_expr = df.rename(columns=expr_rename)
    df_expr["fname"] = fname
    expr_cols = ["cell_id", "fname"] + list(expr_rename.values())[1:]
    df_expr = df_expr[expr_cols]

    return df_labels, df_expr


def generate_and_save():
    # Input directory
    input_dir = RAW_DATASETS_DIR / DATASET_NAME / INFORM_SUBPATH
    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")

    # Prepare output dirs
    mc7_out = OUTPUT_PARENT / MC7_SUBDIR
    ml6_out = OUTPUT_PARENT / ML6_SUBDIR
    mc7_out.mkdir(parents=True, exist_ok=True)
    ml6_out.mkdir(parents=True, exist_ok=True)

    # Aggregate dataframes
    all_labels = pd.DataFrame()
    all_exprs = pd.DataFrame()
    for csv_file in tqdm(sorted(input_dir.iterdir()), desc="Processing inForm CSVs"):
        # Only process cell segmentation files, not tissue segmentation
        if csv_file.name.endswith("_cell_seg_data.csv"):
            labels_df, expr_df = process_csv_file(csv_file)
            all_labels = pd.concat([all_labels, labels_df], ignore_index=True)
            all_exprs = pd.concat([all_exprs, expr_df], ignore_index=True)
        else:
            continue

    # --- Save MC7 outputs ---
    # labels
    labels_path = mc7_out / "labels.csv"
    all_labels.to_csv(labels_path, index=False)
    print(f"[MC7] labels saved to {labels_path}")
    # class names
    names_mc7 = pd.DataFrame({"name": MULTICLASS_ORDER, "label": list(range(len(MULTICLASS_ORDER)))})
    names_mc7.to_csv(mc7_out / "class_names.csv", index=False)
    print(f"[MC7] class_names saved to {mc7_out / 'class_names.csv'}")
    # expressions
    expr_path = mc7_out / "cell_expressions.csv"
    all_exprs.to_csv(expr_path, index=False)
    print(f"[MC7] cell_expressions saved to {expr_path}")

    # --- Save ML6 outputs ---
    ml6_df = all_labels.copy()
    ml6_df["label"] = ml6_df["label"].apply(lambda x: MULTICLASS_TO_MULTILABEL[x])
    ml6_labels_path = ml6_out / "labels.csv"
    ml6_df.to_csv(ml6_labels_path, index=False)
    print(f"[ML6] labels saved to {ml6_labels_path}")
    names_ml6 = pd.DataFrame({"name": MULTILABEL_ORDER, "index": list(range(len(MULTILABEL_ORDER)))})
    names_ml6.to_csv(ml6_out / "class_names.csv", index=False)
    print(f"[ML6] class_names saved to {ml6_out / 'class_names.csv'}")

if __name__ == "__main__":
    generate_and_save()
