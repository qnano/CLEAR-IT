# clearit/scripts/run_pretrain.py
"""
Wrapper to run SimCLR pre-training from a YAML recipe.

Usage:
    python scripts/run_pretrain.py --recipe path/to/recipe.yaml

This script:
  - Loads global paths from clearit.config (config.yaml must exist at project root)
  - Parses the pretrain recipe YAML
  - For each encoder entry in recipe:
      * Builds the sample DataFrame by loading labels and filtering via provided NPZ index list
      * Instantiates Pretrainer with overrides from recipe
      * Initializes model and optimizer
      * Creates DataLoader via PretrainDataManager
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
from clearit.trainers.encoder_trainer import EncoderTrainer
from clearit.data.pretrain.manager import PretrainDataManager

def run_pretrain(recipe_path: Path, lazy_crops: bool = False):
    # Load recipe YAML
    rec = yaml.safe_load(recipe_path.read_text())
    if rec.get('type') != 'pretrain':
        raise ValueError("Recipe type must be 'pretrain'")

    # Roots from config
    models_root = MODELS_DIR / 'encoders'
    datasets_root = DATASETS_DIR

    # Select device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if device == 'cpu':
        print("Warning: CUDA not available; training on CPU may be slow.")

    # Determine workers
    default_workers = min(4, os.cpu_count() or 1)
    num_workers = rec.get('num_workers', default_workers)

    # Iterate jobs
    for job in rec['encoders']:
        eid     = job['id']
        ds      = job['dataset_name']
        an      = job['annotation_name']
        idx_npz_rel = job['data_index_list']
        cache_key = f"{ds}/{an}/{idx_npz_rel}"
        idx_npz = DATASETS_DIR / ds / an / idx_npz_rel

        # build df_samples exactly as you do now, then…
        df_labels = pd.read_csv(DATASETS_DIR / ds / an / 'labels.csv')
        arr = np.load(idx_npz)
        files = arr.files
        if len(files)==1:
            indices = arr[files[0]]
            channels = np.zeros_like(indices, dtype=int)
        else:
            indices, channels = arr[files[0]], arr[files[1]]
        df = df_labels.iloc[indices].reset_index(drop=True)
        df['channel'] = channels

        # model dir + overrides
        model_dir = MODELS_DIR / 'encoders' / eid
        overrides = {
            **{k:v for k,v in job.items()
               if k not in ('id','dataset_name','annotation_name','data_index_list')},
            'cache_key': cache_key,
            # inject metadata so trainer writes it out:
            'id':               eid,
            'dataset_name':     ds,
            'annotation_name':  an,
            'data_index_list':  idx_npz_rel,
        }
        if lazy_crops:
            overrides['lazy_crops'] = True

        pt = EncoderTrainer(str(model_dir), **overrides)

        # skip already in progress/completed
        if pt.config.get('status',0) != 0:
            print(f"  → encoder {eid} status={pt.config['status']} → skipping")
            continue

        # save our freshly‐injected defaults
        pt.save_config()

        pt.initialize_model(device=device)
        pt.initialize_optimizer()
        loader = PretrainDataManager.get_dataloader(
            dataset_name=ds,
            df_samples=df,
            config=pt.config,
            num_workers=num_workers,
            verbose=True
        )
        pt.train(loader)
        pt.save_model()
        pt.save_config()
        print(f"Finished pretraining for encoder {eid}")



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run pretraining from recipe YAML')
    parser.add_argument(
        '--recipe', 
        type=str, 
        required=True,
        help='Path to the pretraining recipe YAML'
    )
    parser.add_argument(
        '--lazy-crops',
        action='store_true',
        dest='lazy_crops',
        help='Enable on-the-fly (lazy) cropping instead of preloading all crops'
    )
    args = parser.parse_args()
    run_pretrain(Path(args.recipe), lazy_crops=args.lazy_crops)
