from __future__ import annotations
from pathlib import Path
from typing import Iterable, List, Optional, Tuple, Dict
import numpy as np
import pandas as pd
import h5py


def inspect_hdf5(h5_path: Path) -> Dict[str, List[str]]:
    """
    Return a dict with group names and dataset paths found in an HDF5 file.
    Useful to quickly see available patients and tables.
    """
    out: Dict[str, List[str]] = {"groups": [], "datasets": []}
    with h5py.File(h5_path, "r") as f:
        def _visit(name, obj):
            if isinstance(obj, h5py.Group):
                out["groups"].append("/" + name if name else "/")
            elif isinstance(obj, h5py.Dataset):
                out["datasets"].append("/" + name)
        f.visititems(_visit)
    return out


def list_patients(h5_path: Path) -> List[str]:
    """
    Return patient group names at the HDF5 root (e.g., 'P01', 'P02', ...).
    """
    with h5py.File(h5_path, "r") as f:
        return sorted([k for k in f.keys() if k.startswith("P")])


def _cap_indices(n: int, per_patient_cap: Optional[int], global_remaining: Optional[int], rng: np.random.Generator) -> np.ndarray:
    """
    Compute a per-patient selection of indices given per-patient and global caps.
    Sampling is random without replacement when needed.
    """
    take = n
    if per_patient_cap is not None:
        take = min(take, int(per_patient_cap))
    if global_remaining is not None:
        take = min(take, int(global_remaining))
    if take >= n:
        return np.arange(n, dtype=int)
    return rng.choice(n, size=take, replace=False)


def load_hdf5_split(
    h5_path: Path,
    patients: Iterable[str],
    load_expressions: bool = False,
    load_labels: bool = False,
    per_patient_cap: Optional[int] = None,
    global_cap: Optional[int] = None,
    seed: int = 42,
) -> Tuple[np.ndarray, pd.DataFrame, Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Load per-patient feature matrices (and optionally expressions/labels) from an HDF5 file.

    HDF5 layout expected:
        /Pxx/features        -> float array [Ni, D]
        /Pxx/expressions     -> optional float array [Ni, E]
        /Pxx/labels          -> optional array [Ni]

    Returns
    -------
    X : ndarray [N, D]
    meta : DataFrame with at least columns ['patient', 'idx_within_patient']
    E : ndarray [N, E] or None
    y : ndarray [N] or None
    """
    rng = np.random.default_rng(seed)

    X_blocks: List[np.ndarray] = []
    E_blocks: List[np.ndarray] = []
    y_blocks: List[np.ndarray] = []
    rows: List[dict] = []

    remaining = None if global_cap is None else int(global_cap)

    with h5py.File(h5_path, "r") as f:
        for pid in patients:
            grp = f[pid]
            if "features" not in grp:
                raise KeyError(f"Missing dataset: '{pid}/features'")
            feats = grp["features"][()]  # (Ni, D)
            n_i = int(feats.shape[0])

            idx_keep = _cap_indices(n_i, per_patient_cap, remaining, rng)
            if remaining is not None:
                remaining -= len(idx_keep)

            X_blocks.append(feats[idx_keep])

            if load_expressions and "expressions" in grp:
                E_blocks.append(grp["expressions"][()][idx_keep])

            if load_labels and "labels" in grp:
                y_blocks.append(grp["labels"][()][idx_keep])

            rows.extend([{"patient": pid, "idx_within_patient": int(i)} for i in idx_keep])

            if remaining is not None and remaining <= 0:
                break

    X = np.vstack(X_blocks) if X_blocks else np.empty((0, 0), dtype=np.float32)
    meta = pd.DataFrame.from_records(rows)
    E = np.vstack(E_blocks) if E_blocks else None
    y = np.concatenate(y_blocks) if y_blocks else None
    return X, meta, E, y
