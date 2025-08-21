#!/usr/bin/env python3
"""
Convert DeepCell_MC17 labels to DeepCell_ML6 multilabel set for TNBC2-MIBI8.

- Reads `labels.csv` from `datasets/TNBC2-MIBI44/DeepCell_MC17`
- Maps each multiclass label to a 6-element multilabel list
- Writes new `labels.csv` to `datasets/TNBC2-MIBI8/DeepCell_ML6`
- Copies `class_names.csv` from the TNBC2-MIBI8/TME-A_ML6 folder into DeepCell_ML6
"""
import shutil
import pandas as pd
from pathlib import Path

# Load data paths from central config
try:
    from clearit.config import DATASETS_DIR
except ImportError:
    raise RuntimeError(
        "Failed to import DATASETS_DIR from clearit.config. "
        "Ensure config.yaml is present and clearit.config can load it."
    )

# Constants
IN_LABELS       = DATASETS_DIR / "TNBC2-MIBI" / "TNBC2-MIBI44"   / "DeepCell_MC17" / "labels.csv"
IN_CLASS_NAMES  = DATASETS_DIR / "TNBC2-MIBI" / "TNBC2-MIBI8"    / "TME-A_ML6"     / "class_names.csv"
OUT_DIR         = DATASETS_DIR / "TNBC2-MIBI" / "TNBC2-MIBI8"    / "DeepCell_ML6"
OUT_LABELS      = OUT_DIR       / "labels.csv"
OUT_CLASS_NAMES = OUT_DIR       / "class_names.csv"

# Label mapping for multiclass->multilabel
LABEL_MAPPING = {
    0:  [0,0,0,0,0,0], # Unidentified           -> other
    7:  [0,0,0,0,0,0], # Neutrophils            -> other
    9:  [0,0,0,0,0,0], # DC                     -> other
    10: [0,0,0,0,0,0], # DC/Mono                -> other
    11: [0,0,0,0,0,0], # Mono/Neu               -> other
    12: [0,0,0,0,0,0], # Other immune           -> other
    13: [0,0,0,0,0,0], # Endothelial            -> other
    14: [0,0,0,0,0,0], # Mesenchymal-like       -> other
    15: [0,0,0,0,0,0], # Tumor                  -> other
    1:  [0,1,0,0,0,0], # Tregs                  -> CD3+
    2:  [0,1,0,0,0,0], # CD4 T                  -> CD3+
    4:  [0,1,0,0,0,0], # CD3 T                  -> CD3+
    3:  [0,1,1,0,0,0], # CD8 T                  -> CD3+, CD8+
    5:  [0,0,0,0,1,0], # NK                     -> CD56+
    6:  [0,0,0,1,0,0], # B                      -> CD20+
    8:  [0,0,0,0,0,1], # Macrophages            -> CD68+
    16: [1,0,0,0,0,0]  # Keratin-positive tumor -> CK+
}


def main():
    # Read existing labels
    if not IN_LABELS.exists():
        raise FileNotFoundError(f"Input labels not found: {IN_LABELS}")
    df = pd.read_csv(IN_LABELS)

    # Map each label to multilabel list
    df['label'] = df['label'].apply(lambda l: LABEL_MAPPING.get(l, [0,0,0,0,0,0]))

    # Ensure output directory
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Save new labels.csv
    df.to_csv(OUT_LABELS, index=False)
    print(f"Converted labels saved to {OUT_LABELS}")

    # Copy class_names.csv
    if not IN_CLASS_NAMES.exists():
        raise FileNotFoundError(f"Source class_names.csv not found: {IN_CLASS_NAMES}")
    shutil.copy(IN_CLASS_NAMES, OUT_CLASS_NAMES)
    print(f"Copied class names to {OUT_CLASS_NAMES}")

if __name__ == "__main__":
    main()
