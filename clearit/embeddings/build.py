"""
Build CLEAR-IT embedding HDF5 files from prepared datasets and published encoders.

This module intentionally uses the config-rooted CLEAR-IT dataset/model layout
instead of relying on archived MAPS or internal debug artifacts. The current
implementation reads prepared dataset tables from ``DATASETS_DIR`` and writes
the canonical embedding tree to ``EMBEDDINGS_DIR``.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import h5py
import numpy as np
import pandas as pd
import torch
import yaml
from tqdm import tqdm

from clearit.config import DATASETS_DIR, EMBEDDINGS_DIR, MODELS_DIR
from clearit.inference.utils import get_embeddings


@dataclass(frozen=True)
class TargetLayout:
    key: str
    default_scale: float
    self_group_key: str
    cross_group_key: str
    include_self_coords: bool


TARGET_LAYOUTS = {
    "tnbc1-mxif8": TargetLayout(
        key="tnbc1-mxif8",
        default_scale=255.0,
        self_group_key="patient_id",
        cross_group_key="patient_id",
        include_self_coords=False,
    ),
    "tnbc2-mibi8": TargetLayout(
        key="tnbc2-mibi8",
        default_scale=265.0,
        self_group_key="patient_id",
        cross_group_key="fname",
        include_self_coords=False,
    ),
    "crc-codex26": TargetLayout(
        key="crc-codex26",
        default_scale=65536.0,
        self_group_key="patient_id",
        cross_group_key="patient_id",
        include_self_coords=True,
    ),
    "tonsil-imc41": TargetLayout(
        key="tonsil-imc41",
        default_scale=65536.0,
        self_group_key="fname",
        cross_group_key="fname",
        include_self_coords=True,
    ),
}


def run_embed_recipe(
    recipe_path: Path,
    *,
    overwrite: bool = False,
    device: torch.device | None = None,
) -> None:
    recipe = yaml.safe_load(recipe_path.read_text())
    if recipe.get("type") != "embed":
        raise ValueError("Recipe type must be 'embed'")

    target = recipe["target"]
    target_key = target["key"]
    if target_key not in TARGET_LAYOUTS:
        raise KeyError(f"Unknown embedding target key: {target_key}")
    layout = TARGET_LAYOUTS[target_key]

    dataset_name = target["dataset_name"]
    annotation_name = target["annotation_name"]
    self_slug = target["self_slug"]

    batch_size_default = int(recipe.get("batch_size", 256))
    num_workers_default = int(recipe.get("num_workers", 0))
    lazy_crops_default = bool(recipe.get("lazy_crops", True))

    labels_df = _load_labels(dataset_name, annotation_name)
    num_classes = _infer_num_classes(dataset_name, annotation_name)

    jobs = recipe.get("jobs", [])
    if not jobs:
        raise ValueError("Embed recipe must contain at least one job")

    needs_self = any(job["source_slug"] == self_slug for job in jobs)
    self_df = _build_self_table(target_key, labels_df, dataset_name, annotation_name) if needs_self else None
    cross_df = _build_cross_table(labels_df)

    for job in jobs:
        encoder_id = job["encoder_id"]
        source_slug = job["source_slug"]
        source_order = int(job["source_order"])
        is_self_job = source_slug == self_slug

        output_path = _resolve_output_path(
            dataset_name=dataset_name,
            annotation_name=annotation_name,
            self_slug=self_slug,
            source_slug=source_slug,
            source_order=source_order,
        )
        output_path.parent.mkdir(parents=True, exist_ok=True)

        if output_path.exists() and not overwrite:
            print(f"Skipping existing file: {output_path}")
            continue

        base_df = self_df if is_self_job else cross_df
        if base_df is None:
            raise RuntimeError("Failed to construct base table for self embeddings")

        encoder_cfg = _load_encoder_config(encoder_id)
        img_size = int(job.get("img_size", encoder_cfg["img_size"]))
        batch_size = int(job.get("batch_size", batch_size_default))
        num_workers = int(job.get("num_workers", num_workers_default))
        lazy_crops = bool(job.get("lazy_crops", lazy_crops_default))
        proj_layers = int(job.get("proj_layers", 0))
        scale = float(job.get("scale", layout.default_scale))
        channel_indices = job.get("feature_channel_indices")
        encoder_width = int(encoder_cfg["encoder_features"])

        print(f"Writing {output_path}")
        with h5py.File(output_path, "w") as handle:
            group_key = layout.self_group_key if is_self_job else layout.cross_group_key
            groups = sorted(base_df[group_key].astype(str).unique().tolist())
            for group_name in tqdm(groups, desc=f"{source_slug} -> {target_key}"):
                group_df = base_df[base_df[group_key] == group_name]
                if group_df.empty:
                    continue

                group_df = _sort_cells(group_df)
                feature_matrix = _compute_feature_matrix(
                    group_df=group_df,
                    dataset_name=dataset_name,
                    annotation_name=annotation_name,
                    encoder_id=encoder_id,
                    img_size=img_size,
                    batch_size=batch_size,
                    num_workers=num_workers,
                    lazy_crops=lazy_crops,
                    num_classes=num_classes,
                    scale=scale,
                    proj_layers=proj_layers,
                    device=device,
                )
                if channel_indices is not None:
                    feature_matrix = _select_feature_channels(
                        feature_matrix,
                        channel_indices=channel_indices,
                        encoder_feature_width=encoder_width,
                    )

                if is_self_job:
                    _write_self_group(
                        handle=handle,
                        group_name=group_name,
                        group_df=group_df,
                        feature_matrix=feature_matrix,
                        include_coords=layout.include_self_coords,
                    )
                else:
                    _write_cross_group(
                        handle=handle,
                        group_name=group_name,
                        group_df=group_df,
                        feature_matrix=feature_matrix,
                    )


def _load_labels(dataset_name: str, annotation_name: str) -> pd.DataFrame:
    labels_path = DATASETS_DIR / dataset_name / annotation_name / "labels.csv"
    if not labels_path.exists():
        raise FileNotFoundError(f"Labels file not found: {labels_path}")

    df = pd.read_csv(labels_path)
    required = {"fname", "cell_x", "cell_y", "label"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Labels file is missing required columns {sorted(missing)}: {labels_path}")

    df["fname"] = df["fname"].astype(str)
    df["patient_id"] = df["fname"].str.split("_").str[0]
    return df


def _build_cross_table(labels_df: pd.DataFrame) -> pd.DataFrame:
    cols = ["fname", "patient_id", "cell_x", "cell_y", "label"]
    return labels_df.loc[:, cols].copy()


def _build_self_table(
    target_key: str,
    labels_df: pd.DataFrame,
    dataset_name: str,
    annotation_name: str,
) -> pd.DataFrame:
    expr_path = DATASETS_DIR / dataset_name / annotation_name / "cell_expressions.csv"
    if not expr_path.exists():
        raise FileNotFoundError(f"Expression file not found: {expr_path}")

    expr_df = pd.read_csv(expr_path)
    expr_df["fname"] = expr_df["fname"].astype(str)

    area_col = "cellSize" if target_key == "tonsil-imc41" else "cell_area"
    base_cols = ["cell_id", "fname", "patient_id", "cell_x", "cell_y", "label", area_col]
    missing = [col for col in base_cols if col not in labels_df.columns]
    if missing:
        raise ValueError(f"Labels file is missing self-embedding columns {missing}")

    merged = labels_df[base_cols].merge(
        expr_df,
        on=["cell_id", "fname"],
        how="inner",
        validate="one_to_one",
    )

    expr_cols = _expression_columns(target_key, merged)
    missing_expr = [col for col in expr_cols if col not in merged.columns]
    if missing_expr:
        raise ValueError(f"Missing expression columns for {target_key}: {missing_expr}")

    keep_cols = ["fname", "patient_id", "cell_x", "cell_y", "label"] + expr_cols
    return merged.loc[:, keep_cols].copy()


def _expression_columns(target_key: str, merged_df: pd.DataFrame) -> list[str]:
    if target_key == "tnbc1-mxif8":
        return ["DAPI", "CK", "CD3", "CD68", "CD8", "CD56", "CD20", "cell_area", "background"]
    if target_key == "tnbc2-mibi8":
        expr_cols = [
            c for c in merged_df.columns
            if c not in {"cell_id", "fname", "patient_id", "cell_x", "cell_y", "label", "cell_area"}
        ]
        return expr_cols + ["cell_area"]
    if target_key == "crc-codex26":
        expr_cols = [
            c for c in merged_df.columns
            if c not in {"cell_id", "fname", "patient_id", "cell_x", "cell_y", "label", "cell_area"}
        ]
        return expr_cols + ["cell_area"]
    if target_key == "tonsil-imc41":
        expr_cols = [
            c for c in merged_df.columns
            if c not in {"cell_id", "fname", "patient_id", "cell_x", "cell_y", "label", "cellSize", "ROI"}
        ]
        return expr_cols + ["cellSize"]
    raise KeyError(f"Unknown target key: {target_key}")


def _infer_num_classes(dataset_name: str, annotation_name: str) -> int:
    class_names_path = DATASETS_DIR / dataset_name / annotation_name / "class_names.csv"
    if not class_names_path.exists():
        raise FileNotFoundError(f"class_names.csv not found: {class_names_path}")

    df = pd.read_csv(class_names_path)
    if "label" in df.columns:
        return int(df["label"].max()) + 1
    if "index" in df.columns:
        return int(df["index"].max()) + 1
    return len(df)


def _load_encoder_config(encoder_id: str) -> dict:
    config_path = MODELS_DIR / "encoders" / encoder_id / "conf_enc.yaml"
    if not config_path.exists():
        raise FileNotFoundError(f"Encoder config not found: {config_path}")
    return yaml.safe_load(config_path.read_text())


def _resolve_output_path(
    *,
    dataset_name: str,
    annotation_name: str,
    self_slug: str,
    source_slug: str,
    source_order: int,
) -> Path:
    dataset_dir = Path(dataset_name).name
    root = EMBEDDINGS_DIR / dataset_dir / annotation_name
    if source_slug == self_slug:
        return root / "01_features-expressions" / f"{self_slug}.hdf5"
    return root / "02_cross-encoder" / f"{source_order:02d}_{source_slug}.hdf5"


def _sort_cells(df: pd.DataFrame) -> pd.DataFrame:
    sort_cols = [col for col in ("fname", "cell_x", "cell_y") if col in df.columns]
    return df.sort_values(sort_cols).reset_index(drop=True)


def _compute_feature_matrix(
    *,
    group_df: pd.DataFrame,
    dataset_name: str,
    annotation_name: str,
    encoder_id: str,
    img_size: int,
    batch_size: int,
    num_workers: int,
    lazy_crops: bool,
    num_classes: int,
    scale: float,
    proj_layers: int,
    device: torch.device | None,
) -> np.ndarray:
    df_samples = group_df.loc[:, ["fname", "cell_x", "cell_y", "label"]].copy()
    df_samples["scale"] = scale
    df_samples.attrs["img_size"] = img_size
    df_samples.attrs["label_mode"] = "multiclass"
    df_samples.attrs["num_classes"] = num_classes
    df_samples.attrs["lazy_crops"] = lazy_crops

    embeddings = get_embeddings(
        df_samples=df_samples,
        dataset_name=dataset_name,
        annotation_name=annotation_name,
        encoder_id=encoder_id,
        batch_size=batch_size,
        num_workers=num_workers,
        proj_layers=proj_layers,
        device=device,
    )
    return embeddings.cpu().numpy().astype(np.float32, copy=False)


def _select_feature_channels(
    feature_matrix: np.ndarray,
    *,
    channel_indices: Iterable[int],
    encoder_feature_width: int,
) -> np.ndarray:
    channel_indices = list(channel_indices)
    if not channel_indices:
        return feature_matrix

    total_channels = feature_matrix.shape[1] // encoder_feature_width
    if feature_matrix.shape[1] % encoder_feature_width != 0:
        raise ValueError("Feature matrix width is not divisible by encoder feature width")
    if min(channel_indices) < 0 or max(channel_indices) >= total_channels:
        raise ValueError(
            f"Feature channel selection {channel_indices} exceeds available channels 0..{total_channels - 1}"
        )

    blocks = []
    for idx in channel_indices:
        start = idx * encoder_feature_width
        stop = start + encoder_feature_width
        blocks.append(feature_matrix[:, start:stop])
    return np.concatenate(blocks, axis=1)


def _write_self_group(
    *,
    handle: h5py.File,
    group_name: str,
    group_df: pd.DataFrame,
    feature_matrix: np.ndarray,
    include_coords: bool,
) -> None:
    grp = handle.create_group(str(group_name))
    expr_cols = [c for c in group_df.columns if c not in {"fname", "patient_id", "cell_x", "cell_y", "label"}]
    expressions = group_df[expr_cols].to_numpy(dtype=np.float32, copy=False)
    labels = group_df["label"].to_numpy(dtype=np.int64, copy=False)
    _create_matrix_dataset(grp, "expressions", expressions)
    _create_matrix_dataset(grp, "features", feature_matrix)
    _create_vector_dataset(grp, "labels", labels)

    if include_coords:
        coords = group_df[["cell_x", "cell_y"]].to_numpy(dtype=np.float32, copy=False)
        _create_matrix_dataset(grp, "coords", coords)


def _write_cross_group(
    *,
    handle: h5py.File,
    group_name: str,
    group_df: pd.DataFrame,
    feature_matrix: np.ndarray,
) -> None:
    grp = handle.create_group(str(group_name))
    labels = group_df["label"].to_numpy(dtype=np.int64, copy=False)
    cell_x = group_df["cell_x"].to_numpy(dtype=np.int64, copy=False).reshape(-1, 1)
    cell_y = group_df["cell_y"].to_numpy(dtype=np.int64, copy=False).reshape(-1, 1)

    _create_matrix_dataset(grp, "features", feature_matrix)
    _create_vector_dataset(grp, "labels", labels)
    _create_matrix_dataset(grp, "cell_x", cell_x)
    _create_matrix_dataset(grp, "cell_y", cell_y)


def _create_matrix_dataset(group: h5py.Group, name: str, values: np.ndarray) -> None:
    chunk_rows = min(1024, max(1, values.shape[0]))
    chunk_cols = values.shape[1]
    group.create_dataset(
        name,
        data=values,
        chunks=(chunk_rows, chunk_cols),
        compression="lzf",
    )


def _create_vector_dataset(group: h5py.Group, name: str, values: np.ndarray) -> None:
    chunk_rows = min(1024, max(1, values.shape[0]))
    group.create_dataset(
        name,
        data=values,
        chunks=(chunk_rows,),
        compression="lzf",
    )
