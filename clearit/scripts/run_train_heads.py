# clearit/scripts/run_train_heads.py
"""
Wrapper to train classification heads from a YAML recipe.

Usage:
    python scripts/run_train_heads.py --recipe path/to/train_heads.yaml

This script:
  - Loads global paths from clearit.config (config.yaml must exist at project root)
  - Parses a `type: train_head` recipe YAML
  - For each head entry:
      * Builds df_samples by loading labels.csv and filtering via provided NPZ index list
      * Instantiates HeadTrainer with overrides from recipe
      * Initializes model, optimizer
      * Creates DataLoaders via ClassificationDataManager
      * Executes training and saves checkpoint and config
"""
import os
import yaml
import torch
import numpy as np
import pandas as pd
from pathlib import Path
import argparse

from clearit.config import DATA_ROOT, MODELS_DIR, DATASETS_DIR
from clearit.trainers.head_trainer import HeadTrainer
from clearit.data.classification.manager import ClassificationDataManager

def run_train_heads(recipe_path: Path):
    # Load recipe YAML
    rec = yaml.safe_load(recipe_path.read_text())
    if rec.get('type') != 'train_head':
        raise ValueError("Recipe type must be 'train_head'")

    encoders_root = MODELS_DIR / 'encoders'
    heads_root    = MODELS_DIR / 'heads'
    datasets_root = DATASETS_DIR

    # Device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if device == 'cpu':
        print("Warning: CUDA not available; training on CPU may be slow.")

    # Workers and randomness
    default_workers = min(4, os.cpu_count() or 1)
    num_workers = rec.get('num_workers', default_workers)
    random_state = rec.get('random_state', None)

    for job in rec.get('heads', []):
        hid = job['id']
        # use the `base_encoder` field
        eid = job['base_encoder']
        ds  = job['dataset_name']
        an  = job['annotation_name']
        idx_npz_rel = job['data_index_list']

        # Build a cache key
        cache_key = f"{ds}/{an}/{idx_npz_rel}"

        # Build df_samples
        labels_csv = datasets_root / ds / an / 'labels.csv'
        if not labels_csv.exists():
            raise FileNotFoundError(f"Labels file not found: {labels_csv}")
        df_labels = pd.read_csv(labels_csv)

        idx_npz = DATASETS_DIR / ds / an / idx_npz_rel
        arr = np.load(idx_npz)
        indices = arr.get('arr_0', None)
        if indices is None:
            # take first array if not named arr_0
            indices = arr[arr.files[0]]
        df_samples = df_labels.iloc[indices].reset_index(drop=True)

        # Prepare overrides
        overrides = {
            k: v for k, v in job.items()
        }
        # inject our cache_key
        overrides['cache_key'] = cache_key

        # Define and create necessary directories
        encoder_dir = encoders_root / eid
        head_dir    = heads_root / hid
        head_dir.mkdir(parents=True, exist_ok=True)

        # Instantiate trainer
        ht = HeadTrainer(
            encoder_dir = str(encoder_dir),
            head_dir    = str(head_dir),
            overrides   = overrides,
        )

        # Skip if already in progress or done
        status = ht.config.get('status', 0)
        if status in (0.5, 1):
            print(f"→ head {hid} status={status} → skipping")
            continue

        ht.save_config()
        ht.initialize_model()
        ht.model.to(ht.device) 
        ht.initialize_optimizer()

        # Build DataLoaders
        train_loader, val_loader = ClassificationDataManager.get_dataloader(
            dataset_name=ds,
            df_samples=df_samples,
            config=ht.config,
            device=device,
            num_workers=num_workers,
            test_size=ht.config.get('train_size', 0.2),
            random_state=random_state,
            verbose=True
        )

        # Train, save, repeat
        ht.train(train_loader, val_loader)
        ht.save_model()
        ht.save_config()

        print(f"Finished head training for head {hid} on encoder {eid}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run head training from recipe YAML')
    parser.add_argument('--recipe', type=str, required=True,
                        help='Path to the head training recipe YAML')
    args = parser.parse_args()
    run_train_heads(Path(args.recipe))
