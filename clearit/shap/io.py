# clearit/shap/io.py
from __future__ import annotations
from pathlib import Path
from typing import Dict, Any, Tuple
import numpy as np
import pandas as pd
import yaml

# --- helpers -----------------------------------------------------------------

def _to_primitive(obj):
    """Convert common numpy scalars/containers to plain Python types for YAML."""
    import numpy as _np
    if obj is None or isinstance(obj, (bool, int, float, str)):
        return obj
    if isinstance(obj, (_np.integer,)):   return int(obj)
    if isinstance(obj, (_np.floating,)):  return float(obj)
    if isinstance(obj, (_np.bool_,)):     return bool(obj)
    if isinstance(obj, (_np.str_, _np.bytes_)): return str(obj)
    if isinstance(obj, _np.ndarray):      return obj.tolist()
    if isinstance(obj, (list, tuple, set)):
        return [_to_primitive(x) for x in obj]
    if isinstance(obj, dict):
        return {str(_to_primitive(k)): _to_primitive(v) for k, v in obj.items()}
    return str(obj)

# --- public API --------------------------------------------------------------

def build_shap_metadata(
    *,
    test_id: str,
    dataset_name: str,
    annotation_name: str,
    head_cfg: Dict[str, Any],
    selection: Dict[str, Any],
    channel_strings: list,
    class_strings: list,
    desired_channel_order: list | None,
    background: Dict[str, Any],
    shap_config: Dict[str, Any],
) -> Dict[str, Any]:
    """Return a minimal, YAML-safe metadata dict (primitives only)."""
    thresholds = head_cfg.get("thresholds")
    if thresholds is not None:
        thresholds = [float(x) for x in thresholds]

    md: Dict[str, Any] = {
        "provenance": {
            "test_id": str(test_id),
            "dataset_name": str(dataset_name),
            "annotation_name": str(annotation_name),
            "head_id": str(head_cfg.get("id", "")),
            "encoder_id": str(head_cfg.get("base_encoder", head_cfg.get("encoder_id", ""))),
        },
        "model": {
            "num_classes": int(head_cfg["num_classes"]),
            "num_channels": int(head_cfg["num_channels"]),
            "img_size": int(head_cfg.get("img_size", 64)),
            "proj_layers": int(head_cfg.get("proj_layers", 0)),
            "thresholds": thresholds,
        },
        "selection": {
            "outcome": str(selection["outcome"]),
            "order": str(selection["order"]),
            "num_per_marker": int(selection["num_per_marker"]),
        },
        "labels": {
            "channel_strings": list(map(str, channel_strings)),
            "class_strings": list(map(str, class_strings)),
            "desired_channel_order": (list(map(int, desired_channel_order))
                                      if desired_channel_order is not None else None),
        },
        "background": {
            "strategy": str(background.get("strategy", "zeros")),
            "num_batches": int(background.get("num_batches", 1)),
        },
        "shap": {
            "check_additivity": bool(shap_config.get("check_additivity", False)),
            "smoothed": bool(shap_config.get("smoothed", False)),
            "sigma": (None if shap_config.get("sigma") is None else float(shap_config["sigma"])),
            "array_shape": tuple(int(x) for x in shap_config.get("array_shape", ())),
            "array_dtype": str(shap_config.get("array_dtype", "")),
        },
    }
    return md

def save_shap_bundle(
    base_path: Path,
    shap_values: np.ndarray,
    df_filtered: pd.DataFrame,
    metadata: Dict[str, Any],
    *,
    dtype: str = "float32",
    compressed: bool = True,
) -> Dict[str, Path]:
    """
    Write three files next to each other (path is implied by base_path stem):
      • <base>.npz       — SHAP array under key 'shap' (N,C,H,W,K), float32
      • <base>.csv.gz    — sample table aligned to array order
      • <base>.yaml      — minimal plotting metadata (primitives only)
    """
    base_path = Path(base_path)
    base_path.parent.mkdir(parents=True, exist_ok=True)

    # 1) SHAP array
    arr = np.asarray(shap_values).astype(dtype, copy=False)
    npz_path = base_path.with_suffix(".npz")
    if compressed:
        np.savez_compressed(npz_path, shap=arr)
    else:
        np.savez(npz_path, shap=arr)

    # 2) Table (CSV.gz; no parquet dependencies)
    cols_keep = [c for c in ("fname", "cell_x", "cell_y", "label", "cell_id") if c in df_filtered.columns]
    extra_cols = [c for c in df_filtered.columns if c not in cols_keep]
    df_to_save = df_filtered[cols_keep + extra_cols].reset_index(drop=True)
    csv_path = base_path.with_suffix(".csv.gz")
    df_to_save.to_csv(csv_path, index=False, compression="gzip")

    # 3) Metadata (YAML; primitives only)
    yml_path = base_path.with_suffix(".yaml")
    md = _to_primitive(metadata)
    with open(yml_path, "w") as f:
        yaml.safe_dump(md, f, sort_keys=False)

    return {"npz": npz_path, "table": csv_path, "yaml": yml_path}

def load_shap_bundle(base_path: Path) -> Tuple[np.ndarray, pd.DataFrame, Dict[str, Any]]:
    """
    Load files written by save_shap_bundle():
      • <base>.npz
      • <base>.csv.gz
      • <base>.yaml
    """
    base_path = Path(base_path)
    with np.load(base_path.with_suffix(".npz")) as npz:
        shap_values = npz["shap"]
    df = pd.read_csv(base_path.with_suffix(".csv.gz"))
    with open(base_path.with_suffix(".yaml"), "r") as f:
        meta = yaml.safe_load(f)
    return shap_values, df, meta
