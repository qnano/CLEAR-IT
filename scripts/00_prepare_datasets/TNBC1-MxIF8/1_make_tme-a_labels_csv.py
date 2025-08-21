#!/usr/bin/env python3
"""
Generate `labels.csv` and `class_names.csv` for the TNBC1-MxIF8 dataset
using the TME-A_ML6 label set.

Paths are derived from the global `config.yaml` via `clearit.config`.
"""
import ast
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
TABLES_SUBPATH = Path("TME-Analyzer") / "tables"
OUTPUT_SUBDIR = "TME-A_ML6"
CLASS_ORDER = ["CK+", "CD3+", "CD8+", "CD20+", "CD56+", "CD68+"]


def process_csv_file(file_path: Path, class_order):
    df = pd.read_csv(file_path)
    fname = file_path.stem

    df = df.rename(columns={
        "cell number": "cell_id",
        "Cell Area": "cell_area",
        "Cell Centroid": "centroid",
        "Phenotypes": "label"
    })

    df["cell_id"] = df["cell_id"].astype(int)
    df["cell_area"] = df["cell_area"].astype(int)

    # Extract x, y from centroid list
    df["cell_x"] = df["centroid"].apply(lambda c: int(round(ast.literal_eval(c)[1])))
    df["cell_y"] = df["centroid"].apply(lambda c: int(round(ast.literal_eval(c)[0])))
    df = df.drop(columns=["centroid"])

    # Convert Phenotypes to binary list
    df["label"] = df["label"].apply(
        lambda l: convert_labels_to_binary_list(ast.literal_eval(l), class_order)
    )
    df["fname"] = fname

    return df[["cell_id", "cell_area", "cell_x", "cell_y", "label", "fname"]]


def convert_labels_to_binary_list(labels, class_order):
    binary = [0] * len(class_order)
    for lab in labels:
        if lab in class_order:
            idx = class_order.index(lab)
            binary[idx] = 1
    return binary


def generate_labels_csv():
    # Construct input and output paths from config
    input_dir = RAW_DATASETS_DIR / DATASET_NAME / TABLES_SUBPATH
    if not input_dir.exists():
        raise FileNotFoundError(
            f"Input directory not found: {input_dir}\n"
            "Ensure your data_root and raw_datasets are set correctly in config.yaml."
        )

    output_dir = DATASETS_DIR / DATASET_NAME / OUTPUT_SUBDIR
    output_dir.mkdir(parents=True, exist_ok=True)

    all_data = pd.DataFrame()
    for csv_file in tqdm(sorted(input_dir.iterdir()), desc="Processing CSV files"):
        if csv_file.suffix.lower() == ".csv":
            df = process_csv_file(csv_file, CLASS_ORDER)
            all_data = pd.concat([all_data, df], ignore_index=True)

    labels_path = output_dir / "labels.csv"
    all_data.to_csv(labels_path, index=False)
    print(f"labels.csv saved to {labels_path}")

    # Save class names
    class_names = pd.DataFrame({
        "name": CLASS_ORDER,
        "index": list(range(len(CLASS_ORDER)))
    })
    class_names_path = output_dir / "class_names.csv"
    class_names.to_csv(class_names_path, index=False)
    print(f"class_names.csv saved to {class_names_path}")


if __name__ == "__main__":
    generate_labels_csv()
