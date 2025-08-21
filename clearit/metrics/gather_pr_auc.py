# clearit/metrics/gather_pr_auc.py
"""
Gather overall micro-averaged PR-AUC values from multiple experiment folders,
ready for boxplot_performance.

Each entry associates:
  - a directory path containing CSV files with multilabel predictions (sigmoid_* columns)
  - a group label (e.g. dataset or encoder name)
  - a configuration label (e.g. hyperparameter setting)

The function concatenates all CSVs in each folder, splits into `chunks` contiguous subsets,
computes micro-averaged PR-AUC per chunk, and returns a tidy DataFrame with columns:
    Configuration, Group, Value

Usage:
    entries = [
        {"path": "/path/to/exp1", "group": "TNBC1-MxIF", "config": "base"},
        {"path": "/path/to/exp2", "group": "TNBC1-MxIF", "config": "loss-opt"},
        # ...
    ]
    df = gather_pr_auc(entries, chunks=10)

Optional:
    pass `metric_fn` to compute a different metric instead of PR-AUC.
"""
import os
import glob
import pandas as pd
import numpy as np
from sklearn.metrics import average_precision_score
from typing import Optional, Sequence, Callable

def gather_pr_auc(
    entries: Sequence[dict],
    chunks: int = 10,
    *,
    metric_fn: Optional[Callable[[np.ndarray, np.ndarray], float]] = None,
    class_labels: Optional[Sequence[str]] = None,
    total: bool = False,
) -> pd.DataFrame:
    """
    entries: list of dicts with keys:
      - path: directory containing CSVs with sigmoid_* and target_* columns
      - group: hue label (e.g. dataset name)
      - config: x-axis configuration label

    chunks: number of splits per entry

    metric_fn: function(y_true, y_pred) -> float; defaults to average_precision_score

    class_labels: list of class names to use (must match the order of sigmoid_ columns);
                  if None, inferred from column names.  Ignored when total=True.

    total: if True, compute *one* macro-AUC per chunk (averaging across classes)
           and emit only that; if False, emit one row per (class, chunk) as before.

    Returns a DataFrame with columns ['Configuration','Group','Value'].
    """

    if metric_fn is None:
        metric_fn = average_precision_score

    records = []
    for entry in entries:
        folder = entry['path']
        group  = entry['group']
        cfg    = entry['config']

        # collect all CSVs under folder/**
        files = glob.glob(os.path.join(folder, '**', '*.csv'), recursive=True)
        if not files:
            continue

        # read & concat
        dfs = []
        for f in files:
            try:
                dfs.append(pd.read_csv(f))
            except Exception:
                pass
        if not dfs:
            continue
        all_df = pd.concat(dfs, ignore_index=True)

        # find predict/target cols
        pred_cols = [c for c in all_df.columns if c.startswith('sigmoid_')]
        true_cols = [c.replace('sigmoid_','target_') for c in pred_cols]
        if not pred_cols or not all(c in all_df.columns for c in true_cols):
            continue

        # determine class labels (only if total=False)
        if not total:
            if class_labels is None:
                labels = [c.split('sigmoid_')[1] for c in pred_cols]
            else:
                if len(class_labels) != len(pred_cols):
                    raise ValueError(
                        "Provided class_labels length does not match number of sigmoid_ columns"
                    )
                labels = list(class_labels)

        y_pred = all_df[pred_cols].values       # shape (N, n_classes)
        y_true = all_df[true_cols].values
        N = len(all_df)

        # split indices exactly like your old code
        cuts = np.linspace(0, N, chunks+1, dtype=int)
        splits = [np.arange(cuts[i], cuts[i+1]) for i in range(chunks)]

        for idxs in splits:
            if len(idxs) == 0:
                continue

            if total:
                for j in range(y_pred.shape[1]):
                    val = metric_fn(y_true[idxs, j], y_pred[idxs, j])
                    records.append({
                        'Configuration': cfg,
                        'Group'        : group,
                        'Value'        : val
                    })

            else:
                # emit one row per class × chunk
                for j, label in enumerate(labels):
                    val = metric_fn(y_true[idxs, j], y_pred[idxs, j])
                    records.append({
                        'Configuration': cfg,
                        'Group':         label,
                        'Value':         val
                    })

    return pd.DataFrame(records)