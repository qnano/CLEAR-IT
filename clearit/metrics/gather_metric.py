# clearit/metrics/gather_metric.py
import os
import glob
import pandas as pd
from typing import Dict, Any, Sequence, Optional
from sklearn.metrics import f1_score, precision_score, recall_score

def gather_metric(
    entries: Sequence[Dict[str, Any]],
    *,
    metric: str = "f1",           # 'f1' | 'precision' | 'recall'
    average: str = "micro",       # passed to sklearn metric
    results_subdir: str = "results",
    usecols: Optional[Sequence[str]] = ("target","prediction"),
) -> pd.DataFrame:
    """
    For each entry['path'] (a run folder Nxx-kyy), read ALL per-image CSVs under
    `results_subdir`, concatenate, compute the selected metric on (target, prediction),
    and emit one row: ['Configuration','Group','Value'].

    This returns many rows per (Configuration, Group) — perfect for box/whiskers.
    """
    metric_map = {
        "f1": f1_score,
        "precision": precision_score,
        "recall": recall_score,
    }
    if metric not in metric_map:
        raise ValueError(f"metric must be one of {list(metric_map)}")
    metric_fn = metric_map[metric]

    rows = []
    for e in entries:
        run_dir = e["path"]
        cfg     = e["config"]
        group   = e["group"]

        files = glob.glob(os.path.join(run_dir, results_subdir, "*.csv"))
        if not files:
            continue

        dfs = []
        for f in files:
            try:
                dfs.append(pd.read_csv(f, usecols=usecols))
            except Exception:
                # fall back without usecols if columns vary
                try:
                    dfs.append(pd.read_csv(f))
                except Exception:
                    pass
        if not dfs:
            continue

        df_all = pd.concat(dfs, ignore_index=True)
        if not {"target","prediction"}.issubset(df_all.columns):
            # nothing to compute
            continue

        y_true = df_all["target"]
        y_pred = df_all["prediction"]
        val = metric_fn(y_true, y_pred, average=average)

        rows.append({
            "Configuration": cfg,
            "Group":         group,
            "Value":         float(val),
        })

    return pd.DataFrame(rows)
