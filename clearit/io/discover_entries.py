# clearit/io/discover_entries.py
import os
import re
import glob
from typing import Dict, List, Any

def discover_entries(
    groups_to_dirs: Dict[str, str],
    *,
    name_regex: str = r"^N(\d+)-k(\d+)$",
    results_subdir: str = "results"
) -> List[Dict[str, Any]]:
    """
    Walk each base dir and return one entry per run folder named like Nxx-kyy.
    Each entry has:
      - path   : path/to/Nxx-kyy
      - config : int(xx)
      - group  : the key from groups_to_dirs (e.g. 'features', 'expressions', ...)
    Only folders that contain `results_subdir` with at least one CSV are kept.
    """
    pat = re.compile(name_regex)
    entries: List[Dict[str,Any]] = []

    for group, base in groups_to_dirs.items():
        if not os.path.isdir(base):
            continue
        for name in os.listdir(base):
            run_dir = os.path.join(base, name)
            if not os.path.isdir(run_dir):
                continue
            m = pat.match(name)
            if not m:
                continue
            # extract the N (patients) as configuration
            cfg = int(m.group(1))
            res_dir = os.path.join(run_dir, results_subdir)
            if not os.path.isdir(res_dir):
                continue
            # must have at least one CSV inside results/
            has_csv = bool(glob.glob(os.path.join(res_dir, "*.csv")))
            if not has_csv:
                continue
            entries.append({"path": run_dir, "config": cfg, "group": group})

    # Preserve a stable ordering by (group, config, path)
    entries.sort(key=lambda e: (e["group"], e["config"], e["path"]))
    return entries