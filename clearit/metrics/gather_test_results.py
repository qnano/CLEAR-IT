# clearit/metrics/gather_test_results.py

from pathlib import Path
from typing import List
import numpy as np
import pandas as pd
import yaml

from clearit.config import OUTPUTS_DIR, MODELS_DIR

def get_classifier_test_results(
    test_id: str,
    df_labels: pd.DataFrame,
    class_strings: List[str],
    *,
    outputs_dir: Path = OUTPUTS_DIR,
    models_dir: Path = MODELS_DIR,
) -> pd.DataFrame:
    """
    Load a test run's predictions (new per-file CSV format) and compute
    per-class TP/TN/FP/FN and confidence scores.

    Inputs
    ------
    test_id       : e.g. "T0113". We load from OUTPUTS_DIR/tests/<test_id>/
    df_labels     : DataFrame containing at least ['fname','cell_x','cell_y'].
                    (You can pass any filtered subset of labels.csv.)
    class_strings : List of class names, len == num_classes.

    Behavior
    --------
    - Reads OUTPUTS_DIR/tests/<test_id>/conf_test.yaml to find head_id (and label_mode).
    - Loads thresholds from MODELS_DIR/heads/<head_id>/conf_head.yaml.
      Falls back to 0.5 for any missing thresholds.
    - Concatenates all {fname}.csv files in the test folder; each file is expected to
      contain columns: 'cell_x','cell_y', 'sigmoid_0..K-1', 'target_0..K-1'.
      A 'fname' column is added based on the CSV filename stem.
    - Merges predictions with df_labels on ['fname','cell_x','cell_y'] (inner join).
    - Computes, for every class:
        • a categorical result column with values in {'TP','TN','FP','FN'}
        • absolute_confidence = |sigmoid - threshold|
        • true_confidence     = confidence of the true label (TP/TN/FP/FN-aware)
        • scaled_confidence   = distance to threshold scaled to [0,1] on each side

    Returns
    -------
    DataFrame: the merged rows (one per predicted cell) plus
               per-class result + confidence columns.
    """
    test_dir = outputs_dir / "tests" / test_id
    if not test_dir.exists():
        raise FileNotFoundError(f"Test directory not found: {test_dir}")

    # --- discover which head was used so we can load thresholds ---
    conf_test_path = test_dir / "conf_test.yaml"
    if not conf_test_path.exists():
        raise FileNotFoundError(f"Missing conf_test.yaml in {test_dir}")
    conf_test = yaml.safe_load(conf_test_path.read_text())

    head_id = conf_test.get("head_id")
    if not head_id:
        raise ValueError(f"'head_id' missing in {conf_test_path}")

    head_cfg_path = models_dir / "heads" / head_id / "conf_head.yaml"
    if not head_cfg_path.exists():
        raise FileNotFoundError(f"Head config not found: {head_cfg_path}")
    head_cfg = yaml.safe_load(head_cfg_path.read_text())

    num_classes = int(head_cfg["num_classes"])
    if len(class_strings) != num_classes:
        raise ValueError(
            f"len(class_strings)={len(class_strings)} but head expects num_classes={num_classes}"
        )

    thresholds = head_cfg.get("thresholds")
    if thresholds is None or len(thresholds) != num_classes:
        # Fallback: 0.5 everywhere if thresholds missing
        thresholds = [0.5] * num_classes
    thresholds = np.asarray(thresholds, dtype=np.float32)

    # --- load all per-image CSVs and stack ---
    csv_paths = sorted(p for p in test_dir.glob("*.csv") if p.name != "conf_test.yaml")
    if not csv_paths:
        raise FileNotFoundError(f"No prediction CSVs found in {test_dir}")

    frames = []
    for p in csv_paths:
        df_p = pd.read_csv(p)
        # ensure required cols exist
        req = {"cell_x", "cell_y"} | {f"sigmoid_{i}" for i in range(num_classes)} | {f"target_{i}" for i in range(num_classes)}
        missing = req - set(df_p.columns)
        if missing:
            raise ValueError(f"{p.name} is missing columns: {sorted(missing)}")
        df_p["fname"] = p.stem
        # make sure coords are ints (avoid merge surprises)
        df_p["cell_x"] = df_p["cell_x"].astype(int)
        df_p["cell_y"] = df_p["cell_y"].astype(int)
        frames.append(df_p)

    df_preds = pd.concat(frames, ignore_index=True)

    # --- normalize df_labels merge keys ---
    for col in ("fname", "cell_x", "cell_y"):
        if col not in df_labels.columns:
            raise KeyError(f"df_labels is missing required column '{col}'")
    dfL = df_labels.copy()
    dfL["cell_x"] = dfL["cell_x"].astype(int)
    dfL["cell_y"] = dfL["cell_y"].astype(int)

    # --- merge predictions ↔ labels on exact coords ---
    merged = pd.merge(
        df_preds,
        dfL,
        on=["fname", "cell_x", "cell_y"],
        how="inner",
        suffixes=("_pred", "_lbl"),
    )
    if len(merged) == 0:
        raise ValueError("Merge produced 0 rows. Check that df_labels matches test outputs.")

    # --- pull sigmoids & targets into arrays ---
    sigmoids = np.column_stack([merged[f"sigmoid_{i}"].to_numpy(dtype=np.float32) for i in range(num_classes)])
    targets  = np.column_stack([merged[f"target_{i}"].to_numpy(dtype=np.int64) for i in range(num_classes)])

    # --- predictions based on thresholds ---
    preds = (sigmoids > thresholds[None, :]).astype(np.int64)

    # --- compute per-class results + confidence columns ---
    out = merged.copy()
    for i, cls in enumerate(class_strings):
        p = preds[:, i]
        t = targets[:, i]
        s = sigmoids[:, i]
        thr = thresholds[i]

        tp = (p == 1) & (t == 1)
        tn = (p == 0) & (t == 0)
        fp = (p == 1) & (t == 0)
        fn = (p == 0) & (t == 1)

        # categorical outcome
        result_col = np.full(len(out), "", dtype=object)
        result_col[tp] = "TP"
        result_col[tn] = "TN"
        result_col[fp] = "FP"
        result_col[fn] = "FN"
        out[cls] = result_col

        # absolute confidence: distance from the threshold
        out[f"{cls}_absolute_confidence"] = np.abs(s - thr)

        # true-label confidence:
        # - if correct: confidence of the true label (s for 1, 1-s for 0)
        # - if wrong:   confidence of the predicted label (s for FP, 1-s for FN)
        tl_conf = np.empty(len(out), dtype=np.float32)
        # correct
        tl_conf[tp] = s[tp]
        tl_conf[tn] = 1.0 - s[tn]
        # wrong
        tl_conf[fp] = s[fp]
        tl_conf[fn] = 1.0 - s[fn]
        out[f"{cls}_true_confidence"] = tl_conf

        # scaled confidence to [0,1] on either side of threshold
        sc_conf = np.empty(len(out), dtype=np.float32)
        # predicted 1: scale (thr..1) → (0..1)
        denom_pos = max(1e-8, 1.0 - thr)
        sc_conf[p == 1] = (s[p == 1] - thr) / denom_pos
        # predicted 0: scale (0..thr) → (0..1)
        denom_neg = max(1e-8, thr)
        sc_conf[p == 0] = (thr - s[p == 0]) / denom_neg
        out[f"{cls}_scaled_confidence"] = sc_conf

    return out
