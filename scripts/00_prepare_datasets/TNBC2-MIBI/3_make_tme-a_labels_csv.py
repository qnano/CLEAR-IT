#!/usr/bin/env python3
"""
Generate `labels.csv` and `class_names.csv` for the TNBC2-MIBI dataset
using the TME-A_ML6 label set.

Paths derived from `config.yaml` via `clearit.config`.
"""
import ast
import pandas as pd
from pathlib import Path
from tqdm import tqdm

# Load paths from central config
try:
    from clearit.config import RAW_DATASETS_DIR, DATASETS_DIR
except ImportError as e:
    raise RuntimeError(
        "Failed to import data paths from clearit.config. "
        "Ensure `config.yaml` is present in the project root and loaded by `clearit.config`."
    ) from e

# Constants
DATASET_NAME = "TNBC2-MIBI"
CLASS_ORDER = ["CK+", "CD3+", "CD8+", "CD20+", "CD56+", "CD68+"]
INPUT_DIR = RAW_DATASETS_DIR / DATASET_NAME / "TME-Analyzer" / "tables"
OUTPUT_DIR = DATASETS_DIR / DATASET_NAME / "TNBC2-MIBI8" / "TME-A_ML6"


def process_xls_file(file_path: Path, class_order):
    df = pd.read_csv(file_path, delimiter='\t')
    # Parse patient ID and build fname
    stem = file_path.stem  # e.g., '...PointXX'
    try:
        pid = int(stem.split('Point')[-1])
    except ValueError:
        raise ValueError(f"Cannot parse patient ID from filename: {file_path.name}")
    fname = f"P{pid:02d}_ROI01"

    # Rename relevant columns
    rename_map = {
        "cell number": "cell_id",
        "Cell Area": "cell_area",
        "Cell Centroid": "centroid",
        "Phenotypes": "label",
    }
    df = df.rename(columns=rename_map)

    # Cast types
    df["cell_id"] = df["cell_id"].astype(int)
    df["cell_area"] = df["cell_area"].astype(int)

    # Extract centroid coordinates
    df["cell_x"] = df["centroid"].apply(lambda c: int(round(ast.literal_eval(c)[1])))
    df["cell_y"] = df["centroid"].apply(lambda c: int(round(ast.literal_eval(c)[0])))
    df = df.drop(columns=["centroid"])

    # Convert labels to binary list
    df["label"] = df["label"].apply(
        lambda l: convert_labels_to_binary_list(ast.literal_eval(l), class_order)
    )
    df["fname"] = fname

    return df[["cell_id", "cell_area", "cell_x", "cell_y", "label", "fname"]]


def convert_labels_to_binary_list(labels, class_order):
    bin_list = [0] * len(class_order)
    for lbl in labels:
        if lbl in class_order:
            bin_list[class_order.index(lbl)] = 1
    return bin_list


def main():
    # Verify input exists
    if not INPUT_DIR.exists():
        raise FileNotFoundError(f"Input tables directory not found: {INPUT_DIR}")

    # Prepare output
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Aggregate
    all_data = []
    for xls_path in tqdm(sorted(INPUT_DIR.glob("*.xls")), desc="Processing XLS files"):
        df = process_xls_file(xls_path, CLASS_ORDER)
        all_data.append(df)

    if not all_data:
        print(f"No .xls files found in {INPUT_DIR}")
        return

    combined = pd.concat(all_data, ignore_index=True)
    # Save labels.csv
    labels_csv = OUTPUT_DIR / "labels.csv"
    combined.to_csv(labels_csv, index=False)
    print(f"Labels CSV saved to {labels_csv}")

    # Save class_names.csv
    cn = pd.DataFrame({"name": CLASS_ORDER, "index": list(range(len(CLASS_ORDER)))})
    classnames_csv = OUTPUT_DIR / "class_names.csv"
    cn.to_csv(classnames_csv, index=False)
    print(f"Class names CSV saved to {classnames_csv}")

if __name__ == "__main__":
    main()
