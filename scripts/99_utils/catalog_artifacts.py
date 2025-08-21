#!/usr/bin/env python3
"""
99_utils/catalog_artifacts.py

Catalog every encoder, head, and test under a unified models/ folder into a CSV.

Usage:
  python scripts/99_utils/catalog_artifacts.py \
      --models-dir /path/to/models \
      [--output-csv catalog_final.csv]

Outputs a CSV (default models-dir/catalog.csv) with columns:
  id, artifact_type, path, dataset_name, annotation_name, data_index_list, ...
"""
import argparse
import yaml
import pandas as pd
from pathlib import Path


def flatten(d, parent_key='', sep='.'):
    """Flatten nested dicts into a single-level dict with sep-delimited keys."""
    items = {}
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.update(flatten(v, new_key, sep=sep))
        else:
            items[new_key] = v
    return items


def main():
    parser = argparse.ArgumentParser(description="Catalog encoders, heads, and tests into a CSV")
    parser.add_argument('--models-dir', type=Path, required=True,
                        help='Root directory containing encoders/, heads/, tests/')
    parser.add_argument('--output-csv', type=Path, default=None,
                        help='Optional path to write the catalog CSV; defaults to <models-dir>/catalog.csv')
    args = parser.parse_args()
    models_dir = args.models_dir
    output_csv = args.output_csv or (models_dir / 'catalog.csv')

    records = []

    # Encoders
    for enc_dir in sorted((models_dir / 'encoders').iterdir()):
        conf = enc_dir / 'conf_enc.yaml'
        if not conf.exists():
            continue
        cfg = yaml.safe_load(conf.read_text()) or {}
        flat = flatten(cfg)
        records.append({
            'id':             enc_dir.name,
            'artifact_type':  'encoder',
            'path':           enc_dir.relative_to(models_dir).as_posix(),
            **flat
        })

    # Heads
    for head_dir in sorted((models_dir / 'heads').iterdir()):
        conf = head_dir / 'conf_head.yaml'
        if not conf.exists():
            continue
        cfg = yaml.safe_load(conf.read_text()) or {}
        flat = flatten(cfg)
        records.append({
            'id':             head_dir.name,
            'artifact_type':  'head',
            'path':           head_dir.relative_to(models_dir).as_posix(),
            **flat
        })

    # Tests
    for test_dir in sorted((models_dir / 'tests').iterdir()):
        conf = test_dir / 'conf_test.yaml'
        if not conf.exists():
            continue
        cfg = yaml.safe_load(conf.read_text()) or {}
        flat = flatten(cfg)
        records.append({
            'id':             test_dir.name,
            'artifact_type':  'test',
            'path':           test_dir.relative_to(models_dir).as_posix(),
            **flat
        })

    # Build DataFrame
    df = pd.DataFrame(records)

    # Reorder columns: id, artifact_type, path, dataset_name, annotation_name, data_index_list, then others
    base_cols = ['id', 'artifact_type', 'path', 'dataset_name', 'annotation_name', 'data_index_list']
    other_cols = [c for c in df.columns if c not in base_cols]
    df = df[[c for c in base_cols if c in df.columns] + other_cols]

    # Write CSV
    df.to_csv(output_csv, index=False)
    print(f"Wrote catalog of {len(df)} artifacts to {output_csv}")

if __name__ == '__main__':
    main()
