# clearit/io/classification_report.py
from typing import List
import pandas as pd

def classification_report(df, class_columns: List[str]) -> "pd.DataFrame":
    """
    Build a TP/FP/TN/FN count summary per class column.
    """
    rows = []
    for col in class_columns:
        counts = df[col].value_counts()
        tp = int(counts.get("TP", 0))
        fp = int(counts.get("FP", 0))
        tn = int(counts.get("TN", 0))
        fn = int(counts.get("FN", 0))
        rows.append({
            "Class": col,
            "TP": tp, "FP": fp, "TN": tn, "FN": fn,
            "Actual Positives": tp + fn,
        })
    return pd.DataFrame(rows)
