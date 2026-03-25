import os
import glob
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Sequence, Dict, Any, Optional, Callable, Tuple, Union


def _compute_single_run(
    path: str,
    metric_fn: Callable[[np.ndarray, np.ndarray], float]
) -> float:
    """
    Load all CSVs under `path`, compute PR-AUC per class,
    take the median across classes, then return the mean of those medians.
    """
    # Identify CSV files in the folder or single CSV path
    if os.path.isfile(path) and path.endswith('.csv'):
        files = [path]
    else:
        files = glob.glob(os.path.join(path, '**', '*.csv'), recursive=True)
    if not files:
        raise ValueError(f"No CSV files found under {path!r}")

    # Read and concatenate all CSVs
    dfs = []
    for f in files:
        try:
            dfs.append(pd.read_csv(f))
        except Exception:
            continue
    if not dfs:
        raise ValueError(f"Could not read any CSV in {path!r}")
    df = pd.concat(dfs, ignore_index=True)

    # Locate prediction and target columns
    pred_cols = [c for c in df.columns if c.startswith('sigmoid_')]
    true_cols = [c.replace('sigmoid_', 'target_') for c in pred_cols]
    if not pred_cols or any(t not in df.columns for t in true_cols):
        raise ValueError(f"Missing sigmoid_/target_ columns in {path!r}")

    # Compute PR-AUC per class
    y_pred = df[pred_cols].values
    y_true = df[true_cols].values
    medians = []
    for j in range(y_pred.shape[1]):
        medians.append(metric_fn(y_true[:, j], y_pred[:, j]))

    # Return the mean of per-class medians
    return float(np.mean(medians))

def _collect_run_results(
    tasks: Sequence[Tuple[str, Any, Any]],
    metric_fn: Callable[[np.ndarray, np.ndarray], float],
    max_workers: Optional[int] = None
) -> pd.DataFrame:
    """
    Compute run-level mean-of-medians values in parallel for the provided tasks.
    Returns columns: ['Configuration', 'Group', 'run_mom'].
    """
    run_results = []
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        future_to_meta = {
            executor.submit(_compute_single_run, path, metric_fn): (cfg, grp)
            for path, cfg, grp in tasks
        }
        for future in as_completed(future_to_meta):
            cfg, grp = future_to_meta[future]
            mom = future.result()
            run_results.append({
                'Configuration': cfg,
                'Group':         grp,
                'run_mom':       mom
            })
    return pd.DataFrame(run_results)


def gather_region_fast(
    entries: Sequence[Dict[str, Any]],
    *,
    metric_fn: Optional[Callable[[np.ndarray, np.ndarray], float]] = None,
    max_workers: Optional[int] = None,
    return_runs: bool = False
) -> Union[pd.DataFrame, Tuple[pd.DataFrame, pd.DataFrame]]:
    """
    Parallel retrieval of mean-of-medians PR-AUC per run.

    Parameters
    ----------
    entries : sequence of dicts
        Each dict must have keys:
        - 'path': Path to a results folder or CSV file
        - 'config': Configuration label (e.g., number of patients)
        - 'group': Group/hue label (e.g., 'Random sampling')
    metric_fn : callable, optional
        Function(y_true, y_pred) -> float; defaults to average_precision_score.
    max_workers : int, optional
        Number of parallel workers; defaults to number of CPU cores.

    Returns
    -------
    pd.DataFrame or (pd.DataFrame, pd.DataFrame)
        Columns: ['Configuration', 'Group', 'low', 'high']
        where 'low' and 'high' are the min and max of mean-of-medians
        across all runs for each (Configuration, Group) pair.

        If return_runs=True, additionally returns a second dataframe with
        columns ['Configuration', 'Group', 'run_mom'].
    """
    if metric_fn is None:
        metric_fn = average_precision_score

    # Prepare tasks: one per results folder
    tasks = [(e['path'], e['config'], e['group']) for e in entries]

    # Aggregate min/max of run_mom per (Configuration, Group)
    df_runs = _collect_run_results(tasks, metric_fn, max_workers=max_workers)
    if df_runs.empty:
        if return_runs:
            return df_runs, df_runs.copy()
        return df_runs

    summary = (
        df_runs
          .groupby(['Configuration', 'Group'])['run_mom']
          .agg(low='min', high='max')
          .reset_index()
    )

    # Preserve configuration order from the input entries.
    configs = list(dict.fromkeys(e['config'] for e in entries))
    summary['Configuration'] = pd.Categorical(
        summary['Configuration'], categories=configs, ordered=True
    )

    summary = summary.sort_values('Configuration').reset_index(drop=True)

    if return_runs:
        df_runs['Configuration'] = pd.Categorical(
            df_runs['Configuration'], categories=configs, ordered=True
        )
        df_runs = df_runs.sort_values('Configuration').reset_index(drop=True)
        return summary, df_runs

    return summary

def gather_points_fast(
    entries: Sequence[Dict[str,Any]],
    *,
    metric_fn: Optional[Callable[[np.ndarray, np.ndarray], float]] = None,
    max_workers: Optional[int] = None,
    return_runs: bool = False
) -> Union[pd.DataFrame, Tuple[pd.DataFrame, pd.DataFrame]]:
    """
    Parallel computation of mean-of-medians (one value per results folder),
    then for each (Configuration,Group) returns the mean of those run-mom values.

    Returns:
      - summary dataframe (default): ['Configuration','Group','mean_of_medians']
      - if return_runs=True: tuple(summary, runs) where runs has
        ['Configuration', 'Group', 'run_mom']
    """
    if metric_fn is None:
        metric_fn = average_precision_score

    tasks = [(e['path'], e['config'], e['group']) for e in entries]

    df_runs = _collect_run_results(tasks, metric_fn, max_workers=max_workers)
    if df_runs.empty:
        if return_runs:
            return df_runs, df_runs.copy()
        return df_runs

    summary = (
        df_runs
          .groupby(['Configuration','Group'])['run_mom']
          .mean()
          .rename('mean_of_medians')
          .reset_index()
    )

    # Preserve configuration order from the input entries.
    configs = list(dict.fromkeys(e['config'] for e in entries))
    summary['Configuration'] = pd.Categorical(
        summary['Configuration'], categories=configs, ordered=True
    )
    summary = summary.sort_values('Configuration').reset_index(drop=True)

    if return_runs:
        df_runs['Configuration'] = pd.Categorical(
            df_runs['Configuration'], categories=configs, ordered=True
        )
        df_runs = df_runs.sort_values('Configuration').reset_index(drop=True)
        return summary, df_runs

    return summary
