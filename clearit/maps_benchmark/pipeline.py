"""Run MAPS-style low-label benchmarking against CLEAR-IT embedding HDF5 files.

This orchestration layer is CLEAR-IT-specific, but it drives a vendored
MAPS-derived benchmark runtime located in ``clearit.maps_benchmark``.
See ``clearit/maps_benchmark/README.md`` for attribution details.
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import Iterable

import h5py
import numpy as np
import pandas as pd
import yaml

from clearit.config import DATASETS_DIR, EMBEDDINGS_DIR, OUTPUTS_DIR

from .trainer import Trainer


TARGET_GROUP_KEYS = {
    "tnbc1-mxif8": "patient_id",
    "tnbc2-mibi8": "patient_id",
    "crc-codex26": "patient_id",
    "tonsil-imc41": "fname",
}


def run_maps_benchmark_recipe(recipe_path: Path, *, overwrite: bool = False) -> None:
    recipe_path = Path(recipe_path)
    recipe = yaml.safe_load(recipe_path.read_text())
    if recipe.get("type") != "maps_benchmark":
        raise ValueError("Recipe type must be 'maps_benchmark'")

    target = recipe["target"]
    target_key = target["key"]
    if target_key not in TARGET_GROUP_KEYS:
        raise KeyError(f"Unknown benchmark target key: {target_key}")

    dataset_name = target["dataset_name"]
    annotation_name = target["annotation_name"]
    self_slug = target["self_slug"]
    published_name = target.get("published_name", Path(dataset_name).name)
    scenario = recipe.get("output", {}).get("scenario", "labelreduction_with_sampling")

    embedding_path = _resolve_embedding_path(dataset_name, annotation_name, self_slug)
    labels_df = _load_labels(dataset_name, annotation_name)
    group_key = TARGET_GROUP_KEYS[target_key]
    metadata_by_group = _build_metadata_by_group(labels_df, group_key=group_key)
    available_groups = _load_hdf5_groups(embedding_path)

    split_cfg = recipe["split"]
    test_ids = [str(x) for x in split_cfg.get("test_ids", [])]
    ranked_train_ids = _load_ranked_train_ids(split_cfg, available_groups=available_groups)

    _validate_split_ids(
        ranked_train_ids=ranked_train_ids,
        test_ids=test_ids,
        available_groups=available_groups,
        metadata_by_group=metadata_by_group,
    )

    sampling = recipe["sampling"]
    k_values = [int(k) for k in sampling.get("k_values", list(range(10)))]
    sample_sizes = sampling["sample_sizes"]
    sampled_train_ids = _build_sampled_train_ids(
        ranked_ids=ranked_train_ids,
        sample_sizes=sample_sizes,
        k_values=k_values,
        seed=int(sampling.get("seed", 42)),
    )

    training_cfg = recipe.get("training", {})
    for modality in recipe["modalities"]:
        modality_name = modality["name"]
        data_parts = list(modality["data_parts"])
        num_features = _infer_num_features(embedding_path, data_parts)
        train_batch_size = int(modality["batch_size"])
        predict_batch_size = int(modality.get("predict_batch_size", 1024))
        learning_rate = float(modality["learning_rate"])
        dropout = float(modality["dropout"])
        norm = bool(modality.get("norm", True))

        for plan in sample_sizes:
            label_n = int(plan["label"])
            for k in k_values:
                train_ids = sampled_train_ids[(label_n, k)]
                run_dir = (
                    OUTPUTS_DIR
                    / "maps_benchmark"
                    / published_name
                    / scenario
                    / modality_name
                    / f"N{label_n:02}-k{k:02}"
                )
                results_dir = run_dir / "results"
                if _should_skip_run(results_dir, test_ids=test_ids, overwrite=overwrite):
                    print(f"Skipping existing benchmark run: {run_dir}")
                    continue

                run_dir.mkdir(parents=True, exist_ok=True)
                print(f"Running {published_name} | {modality_name} | N={label_n} | k={k}")

                trainer = Trainer(
                    results_dir=str(run_dir),
                    data_parts=data_parts,
                    hidden_dim=int(training_cfg.get("hidden_dim", 512)),
                    network_depth=int(training_cfg.get("network_depth", 4)),
                    num_features=num_features,
                    num_classes=int(training_cfg["num_classes"]),
                    batch_size=train_batch_size,
                    learning_rate=learning_rate,
                    dropout=dropout,
                    max_epochs=int(training_cfg.get("max_epochs", 500)),
                    min_epochs=int(training_cfg.get("min_epochs", 250)),
                    patience=int(training_cfg.get("patience", 100)),
                    seed=int(training_cfg.get("seed", 7325111)),
                    num_workers=int(training_cfg.get("num_workers", 4)),
                    norm=norm,
                    verbose=int(training_cfg.get("verbose", 0)),
                )
                trainer.fit(embedding_path, train_ids)

                predictor = Trainer(
                    model_checkpoint_path=str(run_dir / "best_checkpoint.pt"),
                    results_dir=str(run_dir),
                    data_parts=data_parts,
                    hidden_dim=int(training_cfg.get("hidden_dim", 512)),
                    network_depth=int(training_cfg.get("network_depth", 4)),
                    num_features=num_features,
                    num_classes=int(training_cfg["num_classes"]),
                    batch_size=predict_batch_size,
                    learning_rate=learning_rate,
                    dropout=dropout,
                    max_epochs=int(training_cfg.get("max_epochs", 500)),
                    min_epochs=int(training_cfg.get("min_epochs", 250)),
                    patience=int(training_cfg.get("patience", 100)),
                    seed=int(training_cfg.get("seed", 7325111)),
                    num_workers=int(training_cfg.get("num_workers", 4)),
                    norm=norm,
                    verbose=int(training_cfg.get("verbose", 0)),
                )
                pred_labels, _, gt_labels, _, row_group_ids = predictor.predict(embedding_path, test_ids)
                _write_results_csvs(
                    results_dir=results_dir,
                    test_ids=test_ids,
                    pred_labels=pred_labels,
                    gt_labels=gt_labels,
                    row_group_ids=row_group_ids,
                    metadata_by_group=metadata_by_group,
                )


def _resolve_embedding_path(dataset_name: str, annotation_name: str, self_slug: str) -> Path:
    dataset_dir = Path(dataset_name).name
    path = EMBEDDINGS_DIR / dataset_dir / annotation_name / "01_features-expressions" / f"{self_slug}.hdf5"
    if not path.exists():
        raise FileNotFoundError(f"Embedding HDF5 not found: {path}")
    return path


def _load_labels(dataset_name: str, annotation_name: str) -> pd.DataFrame:
    labels_path = DATASETS_DIR / dataset_name / annotation_name / "labels.csv"
    if not labels_path.exists():
        raise FileNotFoundError(f"Labels file not found: {labels_path}")
    labels_df = pd.read_csv(labels_path, usecols=["fname", "cell_x", "cell_y", "label"])
    labels_df["fname"] = labels_df["fname"].astype(str)
    labels_df["patient_id"] = labels_df["fname"].str.split("_").str[0]
    return labels_df.sort_values(["fname", "cell_x", "cell_y"]).reset_index(drop=True)


def _build_metadata_by_group(labels_df: pd.DataFrame, *, group_key: str) -> dict[str, pd.DataFrame]:
    grouped = {}
    for group_id, df_group in labels_df.groupby(group_key, sort=True):
        grouped[str(group_id)] = df_group.reset_index(drop=True)
    return grouped


def _load_hdf5_groups(embedding_path: Path) -> list[str]:
    with h5py.File(embedding_path, "r") as handle:
        return list(handle.keys())


def _load_ranked_train_ids(split_cfg: dict, *, available_groups: list[str]) -> list[str]:
    training_pool = split_cfg["training_pool"]
    source = training_pool["source"]
    if source == "public_patient_rankings":
        ranking_subdir = training_pool["ranking_subdir"]
        ranking_column = training_pool.get("column", "patient_id")
        ranking_path = OUTPUTS_DIR / "ranking" / ranking_subdir / "patient_rankings.csv"
        ranked = pd.read_csv(ranking_path)[ranking_column].astype(str).tolist()
    elif source == "inline":
        ranked = [str(x) for x in training_pool["ordered_ids"]]
    else:
        raise ValueError(f"Unknown training pool source: {source}")

    exclude = {str(x) for x in training_pool.get("exclude_ids", [])}
    return [group_id for group_id in ranked if group_id in available_groups and group_id not in exclude]


def _validate_split_ids(
    *,
    ranked_train_ids: list[str],
    test_ids: list[str],
    available_groups: list[str],
    metadata_by_group: dict[str, pd.DataFrame],
) -> None:
    available_set = set(available_groups)
    missing_train = [group_id for group_id in ranked_train_ids if group_id not in available_set]
    missing_test = [group_id for group_id in test_ids if group_id not in available_set]
    if missing_train:
        raise KeyError(f"Training groups missing from embedding HDF5: {missing_train}")
    if missing_test:
        raise KeyError(f"Test groups missing from embedding HDF5: {missing_test}")
    overlap = sorted(set(ranked_train_ids).intersection(test_ids))
    if overlap:
        raise ValueError(f"Train/test overlap detected: {overlap}")
    missing_meta = [group_id for group_id in test_ids if group_id not in metadata_by_group]
    if missing_meta:
        raise KeyError(f"Test groups missing from labels metadata: {missing_meta}")


def _build_sampled_train_ids(
    *,
    ranked_ids: list[str],
    sample_sizes: list[dict],
    k_values: list[int],
    seed: int,
) -> dict[tuple[int, int], list[str]]:
    random.seed(seed)
    max_k = max(k_values) + 1
    sampled = {}
    for plan in sample_sizes:
        label_n = int(plan["label"])
        draw_n = int(plan.get("draw", label_n))
        combos = _generate_combinations(ranked_ids.copy(), draw_n, max_k)
        for k in k_values:
            sampled[(label_n, k)] = list(combos[k])
    return sampled


def _generate_combinations(names: list[str], n: int, k: int) -> list[list[str]]:
    if n > len(names):
        raise ValueError(f"Cannot draw {n} groups from a pool of size {len(names)}")
    if n == 1:
        all_draws = []
        names_copy = names.copy()
        for _ in range(k):
            if len(names_copy) == 0:
                names_copy = names.copy()
            chosen = random.choice(names_copy)
            names_copy.remove(chosen)
            all_draws.append([chosen])
        return all_draws

    all_combos = set()
    while len(all_combos) < k:
        combo = tuple(random.sample(names, n))
        all_combos.add(combo)
    return [list(combo) for combo in all_combos]


def _infer_num_features(embedding_path: Path, data_parts: Iterable[str]) -> int:
    with h5py.File(embedding_path, "r") as handle:
        first_group = next(iter(handle.keys()))
        total = 0
        for part in data_parts:
            if part not in handle[first_group]:
                raise KeyError(f"Embedding group {first_group} is missing required dataset: {part}")
            total += int(handle[first_group][part].shape[1])
    return total


def _should_skip_run(results_dir: Path, *, test_ids: list[str], overwrite: bool) -> bool:
    if overwrite:
        return False
    if not results_dir.exists():
        return False
    expected_files = {f"{group_id}.csv" for group_id in test_ids if "_ROI" in group_id}
    existing_files = {path.name for path in results_dir.glob("*.csv")}
    return expected_files.issubset(existing_files) if expected_files else bool(existing_files)


def _write_results_csvs(
    *,
    results_dir: Path,
    test_ids: list[str],
    pred_labels: np.ndarray,
    gt_labels: np.ndarray,
    row_group_ids: np.ndarray,
    metadata_by_group: dict[str, pd.DataFrame],
) -> None:
    results_dir.mkdir(parents=True, exist_ok=True)

    cursor = 0
    for group_id in test_ids:
        df_meta = metadata_by_group[group_id].reset_index(drop=True)
        group_len = len(df_meta)
        pred_slice = pred_labels[cursor:cursor + group_len]
        gt_slice = gt_labels[cursor:cursor + group_len]
        group_slice = row_group_ids[cursor:cursor + group_len]
        if len(pred_slice) != group_len:
            raise ValueError(f"Prediction length mismatch for group {group_id}")
        if not np.all(group_slice == group_id):
            raise ValueError(f"Prediction group ordering mismatch for group {group_id}")

        df_out = df_meta[["fname", "cell_x", "cell_y"]].copy()
        df_out["prediction"] = pred_slice.astype(int)
        df_out["target"] = gt_slice.astype(int)
        grouped = df_out.groupby("fname", sort=False)
        for fname, df_fname in grouped:
            df_write = df_fname[["prediction", "target", "cell_x", "cell_y"]]
            df_write.to_csv(results_dir / f"{fname}.csv", index=False)
        cursor += group_len

    if cursor != len(pred_labels):
        raise ValueError("Unconsumed predictions remain after writing benchmark results")
