# clearit/config.py
"""
Configuration loader for the CLEAR-IT project.

Reads `config.yaml` at the project root and exposes:
- DATA_ROOT: Path to the CLEAR-IT-Data folder
- DATASETS_DIR: Path to processed datasets (`datasets/`)
- RAW_DATASETS_DIR: Path to raw datasets (`raw_datasets/`)
- MODELS_DIR: Path to pretrained models (`models/`)
- RESULTS_DIR: Path to results (`results/`)
- OUTPUTS_DIR: Path to outputs (`outputs/`)
- EXPERIMENTS_DIR: Path to experiment recipes (`experiments/`)
"""
import yaml
from pathlib import Path

# Locate and load the configuration file
CONFIG_FILE = Path(__file__).parent.parent / "config.yaml"
if not CONFIG_FILE.exists():
    raise FileNotFoundError(f"config.yaml not found at expected location: {CONFIG_FILE}")

# Parse the YAML
with open(CONFIG_FILE, 'r') as f:
    cfg = yaml.safe_load(f)

# Paths configuration
paths_cfg = cfg.get("paths", {})
DATA_ROOT = Path(paths_cfg.get("data_root", ".")).expanduser()

# Allow independent overrides, fallback to DATA_ROOT subpaths
DATASETS_DIR = Path(paths_cfg.get("datasets_dir", DATA_ROOT / "datasets")).expanduser()
RAW_DATASETS_DIR = Path(paths_cfg.get("raw_datasets_dir", DATA_ROOT / "raw_datasets")).expanduser()
MODELS_DIR = Path(paths_cfg.get("models_dir", DATA_ROOT / "models")).expanduser()
RESULTS_DIR = Path(paths_cfg.get("results_dir", DATA_ROOT / "results")).expanduser()
OUTPUTS_DIR = Path(paths_cfg.get("outputs_dir", DATA_ROOT / "outputs")).expanduser()
EXPERIMENTS_DIR = Path(paths_cfg.get("experiments_dir", DATA_ROOT / "experiments")).expanduser()
