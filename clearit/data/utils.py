import os
import numpy as np
import torch
import time
import pandas as pd
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import tifffile

from clearit.config import DATASETS_DIR

def load_images_to_memory(dataset_name: str, unique_files, verbose=False, max_workers=None):
    """
    Given a dataset_name and a list of file stems (no extension),
    load each image from DATASETS_DIR/{dataset_name}/images/{stem}.tiff (or .tif)
    in parallel, returning a dict {stem: ndarray of shape (C,H,W)}.

    Parameters:
      dataset_name: name of the dataset directory under datasets/
      unique_files: iterable of filename stems (no extension)
      verbose: if True, print loading messages
      max_workers: number of threads (defaults to number of stems or CPU count)
    """
    images_dir = Path(DATASETS_DIR) / dataset_name / "images"
    image_dict = {}

    # Determine number of workers
    if max_workers is None:
        max_workers = os.cpu_count() or 1
    # cap workers to number of files
    max_workers = min(max_workers, len(unique_files) if hasattr(unique_files, '__len__') else max_workers)

    def _load_one(stem):
        for ext in (".tiff", ".tif"):
            path = images_dir / f"{stem}{ext}"
            if path.exists():
                if verbose:
                    print(f"  → Loading {path}")
                arr = tifffile.imread(str(path))
                return stem, arr
        raise FileNotFoundError(f"No image for '{stem}' in {images_dir}")

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(_load_one, stem): stem for stem in unique_files}
        for fut in as_completed(futures):
            stem, arr = fut.result()
            image_dict[stem] = arr
    return image_dict


def extract_crops(
    df_samples: pd.DataFrame,
    *,
    dataset_name: str,
    crop_size: int,
    mode: str = "single_channel",  # "single_channel" or "all_channels"
    max_workers: int = None
) -> (pd.DataFrame, torch.FloatTensor):
    """
    Extract square patches from multi-channel TIFF images.

    Behavior:
    - Load each full image once and reflect-pad by half the crop size.
    - Extract crops in a canonical order for cache stability.
    - Return a mapping DataFrame with both `original_index` and key columns.

    Parameters
    ----------
    df_samples : DataFrame
        Must contain: 'fname', 'cell_x', 'cell_y'. If mode == 'single_channel',
        must also contain 'channel'. If 'scale' is missing, defaults to 255.0.
    dataset_name : str
        e.g. "TNBC2-MIBI/TNBC2-MIBI8"
    crop_size : int
        Patch width/height (pixels).
    mode : str
        "single_channel" for output shape `(N, 1, H, W)`.
        "all_channels" for output shape `(N, C, H, W)`.
    max_workers : int or None
        Thread pool size for IO and crop extraction. Defaults to CPU count,
        capped by number of unique image stems.

    Returns
    -------
    crop_df : DataFrame
        Columns include:
        - `original_index`, the row index from `df_samples`
        - `fname`, `cell_x`, `cell_y`, and `channel` for single-channel mode
        - `crop_tensor_index`, the row in the returned tensor
    crops_tensor : FloatTensor
        Shape:
        - `(N, 1, H, W)` for single-channel mode
        - `(N, C, H, W)` for all-channel mode
        dtype `float32`, normalized to `[0, 1]` by dividing by `scale`.
    """
    # Validate inputs and set defaults.
    if not {"fname", "cell_x", "cell_y"}.issubset(df_samples.columns):
        raise ValueError("df_samples must contain columns: 'fname', 'cell_x', 'cell_y'.")

    df = df_samples.copy()
    if "scale" not in df.columns:
        # Project convention: even if TIFF dtype is uint16, dynamic range is 0..255
        df["scale"] = 255.0

    # Preserve the source row index in original_index.
    df["original_index"] = df.index

    # Build the canonical key set and sort for order-invariant extraction.
    key_cols = ["fname", "cell_x", "cell_y"]
    if mode == "single_channel":
        key_cols.append("channel")
    df_sorted = df.sort_values(key_cols).reset_index(drop=True)

    total = len(df_sorted)
    half = crop_size // 2
    stems = df_sorted["fname"].unique().tolist()

    # Determine thread count.
    if max_workers is None:
        max_workers = os.cpu_count() or 1
    max_workers = min(max_workers, max(1, len(stems)))

    # Load and pad each image in parallel.
    def _load_pad(stem: str):
        img_dir = Path(DATASETS_DIR) / dataset_name / "images"
        arr = None
        for ext in (".tiff", ".tif"):
            path = img_dir / f"{stem}{ext}"
            if path.exists():
                arr = tifffile.imread(str(path))
                break
        if arr is None:
            raise FileNotFoundError(f"No image for '{stem}' in {img_dir}")
        return stem, np.pad(arr, ((0, 0), (half, half), (half, half)), mode="reflect")

    t0 = time.time()
    padded = {}
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = {ex.submit(_load_pad, s): s for s in stems}
        for fut in as_completed(futures):
            stem, img = fut.result()
            padded[stem] = img
    dt = time.time() - t0
    print(f"Loaded & padded {len(padded)} images in {dt:.1f}s ({len(padded)/max(dt,1e-6):.1f} f/s)")

    # Determine the output tensor shape.
    sample0 = next(iter(padded.values()))
    C = sample0.shape[0] if mode == "all_channels" else 1
    crops = torch.empty((total, C, crop_size, crop_size), dtype=torch.float32)

    # Crop rows from the canonical ordering.
    def _crop_row(idx_row):
        idx, row = idx_row
        stem = row["fname"]
        img = padded[stem]  # (C, H+2h, W+2h)
        x, y = int(row["cell_x"]), int(row["cell_y"])

        if mode == "all_channels":
            patch = img[:, y:y + crop_size, x:x + crop_size]  # (C, H, W)
        else:
            ch = int(row.get("channel", 0))
            patch = img[ch, y:y + crop_size, x:x + crop_size][None, ...]  # (1, H, W)

        # Normalize to [0,1] using scalar scale (float)
        patch = patch.astype(np.float32) / float(row["scale"])
        return idx, patch

    t1 = time.time()
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = {ex.submit(_crop_row, (i, r)): i for i, r in df_sorted.iterrows()}
        for fut in as_completed(futures):
            i, patch = fut.result()
            crops[i] = torch.from_numpy(patch)
    dt2 = time.time() - t1
    print(f"Extracted {total} crops in {dt2:.1f}s ({total/max(dt2,1e-6):.1f} crops/s)")

    # Build the mapping table with key columns and original_index.
    crop_df = df_sorted.loc[:, key_cols + ["original_index"]].copy()
    crop_df["crop_tensor_index"] = crop_df.index  # index in the returned tensor

    # Release the padded image cache.
    padded.clear()

    return crop_df, crops
