# clearit/augmentations/utils.py
from __future__ import annotations

from pathlib import Path
from functools import lru_cache
from typing import Dict, Any, Optional
import copy
import math
import yaml


@lru_cache(maxsize=1)
def _default_transforms() -> Dict[str, Any]:
    """
    Load default transforms from configs/pretrainer_defaults.yaml.
    Returns {} if the file or the 'transforms' block is missing/invalid.
    """
    cfg_path = Path(__file__).resolve().parents[1] / "configs" / "pretrainer_defaults.yaml"
    try:
        cfg = yaml.safe_load(cfg_path.read_text())
    except Exception:
        return {}
    base = cfg.get("transforms", {}) or {}
    return base if isinstance(base, dict) else {}


def make_transformdict(overrides: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Build a transform dict by taking YAML defaults and overlaying `overrides`.
    No hardcoded augmentation keys; unknown keys are preserved.
    """
    base = copy.deepcopy(_default_transforms())
    if overrides and isinstance(overrides, dict):
        base.update(overrides)
    return base


def get_crop_size_preload(transformdict: Dict[str, Any], img_size: int = 64) -> int:
    """
    Compute the preloaded crop size needed to support translation/zoom at runtime.

    Parameters
    ----------
    transformdict : dict
        Use `make_transformdict(...)` first so required keys exist.
    img_size : int
        Target patch size fed into the network after aug.

    Returns
    -------
    int : crop size to preload from the original image.
    """
    # expected keys exist if caller used make_transformdict
    t    = abs(float(transformdict.get("translate", 0)))
    smax = float(transformdict.get("zoomfactor_max", 1.0))  # zoom-out factor
    crop_size_preload = int(img_size * math.sqrt(smax) + 2 * t)
    return crop_size_preload
