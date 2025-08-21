# clearit/io/select.py
from typing import List
import pandas as pd

def select_cells_by_outcome_and_confidence(
    df_results: pd.DataFrame,
    markers: List[str],
    outcome: str,                 # "TP"|"TN"|"FP"|"FN"
    order: str,                   # "highest"|"lowest"
    num_per_marker: int,
) -> pd.DataFrame:
    """
    For each marker `m`, keep rows with df_results[m]==outcome,
    sort by f"{m}_true_confidence" (asc/desc), take N, concat,
    then drop duplicates by 'cell_id' if present else by (fname,x,y).
    """
    dfs = []
    ascending = (order == "lowest")
    for m in markers:
        conf_col = f"{m}_true_confidence"
        if conf_col not in df_results.columns:
            raise KeyError(f"Missing column '{conf_col}' in df_results.")
        d = df_results[df_results[m] == outcome].sort_values(conf_col, ascending=ascending).head(num_per_marker)
        dfs.append(d)
    if not dfs:
        return pd.DataFrame()
    out = pd.concat(dfs, ignore_index=True)
    if "cell_id" in out.columns:
        return out.drop_duplicates(subset=["cell_id"])
    return out.drop_duplicates(subset=["fname", "cell_x", "cell_y"])
