#!/usr/bin/env python3
"""
Copy and process images for TNBC2-MIBI44 and TNBC2-MIBI8 datasets.

- Copies all 44-channel TIFFs to `datasets/TNBC2-MIBI/TNBC2-MIBI44/images`
  and writes `channels.txt` from `channels44.txt`.
- For each TIFF, selects 8 channels per `channels8.txt` and writes to
  `datasets/TNBC2-MIBI/TNBC2-MIBI8/images` with compressed output.

Paths derived from `config.yaml` via `clearit.config`.
"""
import shutil
from pathlib import Path
import tifffile
from tqdm import tqdm

# Load data paths from central config
try:
    from clearit.config import RAW_DATASETS_DIR, DATASETS_DIR
except ImportError as e:
    raise RuntimeError(
        "Failed to import data paths from clearit.config. "
        "Ensure config.yaml is present and clearit.config can load it."
    ) from e

# Constants
DATASET_NAME = "TNBC2-MIBI"
INPUT_DIR = RAW_DATASETS_DIR / DATASET_NAME
OUTPUT_PARENT = DATASETS_DIR / DATASET_NAME

# 44-channel output
OUT44_DIR = OUTPUT_PARENT / "TNBC2-MIBI44" / "images"
CHANNELS44_SRC = INPUT_DIR / "channels44.txt"
CHANNELS44_DST = OUTPUT_PARENT / "TNBC2-MIBI44" / "channels.txt"

# 8-channel output
OUT8_DIR = OUTPUT_PARENT / "TNBC2-MIBI8" / "images"
CHANNELS8_SRC = INPUT_DIR / "channels8.txt"
CHANNELS8_DST = OUTPUT_PARENT / "TNBC2-MIBI8" / "channels.txt"

# Raw images directory
RAW_IMAGES_DIR = INPUT_DIR / "images"

# Ensure output directories exist
for d in [OUT44_DIR, OUT8_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# Copy channels.txt for 44-channel dataset
if not CHANNELS44_SRC.exists():
    raise FileNotFoundError(f"44-channel list not found: {CHANNELS44_SRC}")
shutil.copy(CHANNELS44_SRC, CHANNELS44_DST)
print(f"Copied channels.txt for 44-channel to {CHANNELS44_DST}")

# Process raw images for 44 channels with progress bar
for src_path in tqdm(sorted(RAW_IMAGES_DIR.glob('*.tiff')), desc="Processing 44-channel images"):
    pid = int(src_path.stem.split('Point')[-1])
    dst_name = f"P{pid:02d}_ROI01.tiff"
    dst_path = OUT44_DIR / dst_name
    shutil.copy(src_path, dst_path)

print(f"Copied {len(list(RAW_IMAGES_DIR.glob('*.tiff')))} images to 44-channel folder: {OUT44_DIR}")

# Copy channels.txt for 8-channel dataset
if not CHANNELS8_SRC.exists():
    raise FileNotFoundError(f"8-channel list not found: {CHANNELS8_SRC}")
shutil.copy(CHANNELS8_SRC, CHANNELS8_DST)
print(f"Copied channels.txt for 8-channel to {CHANNELS8_DST}")

# Read channels to keep
with open(CHANNELS8_SRC, 'r') as f:
    keep = [line.strip() for line in f if line.strip()]

# Process raw images for 8 channels with progress bar
for src_path in tqdm(sorted(RAW_IMAGES_DIR.glob('*.tiff')), desc="Processing 8-channel images"):
    pid = int(src_path.stem.split('Point')[-1])
    dst_name = f"P{pid:02d}_ROI01.tiff"
    dst_path = OUT8_DIR / dst_name

    # Read multi-page TIFF and map channel names to arrays using raw tag index
    with tifffile.TiffFile(src_path) as tif:
        # Non-standard format: use tag index 13 to get channel name
        channel_order_dict = {
            page.tags.values()[13].value: page.asarray()
            for page in tif.pages
        }

    # Select in specified order, with error messages for missing channels
    missing = [ch for ch in keep if ch not in channel_order_dict]
    if missing:
        raise KeyError(
            f"Channels not found in TIFF {src_path.name}: {missing} "
            f"Available channels: {list(channel_order_dict.keys())}"
        )
    selected = [channel_order_dict[ch] for ch in keep]

    # Save output with compression
    tifffile.imwrite(
        dst_path,
        selected,
        photometric='minisblack',
        dtype='uint16',
        compression='zlib'
    )

print(f"Saved {len(list(RAW_IMAGES_DIR.glob('*.tiff')))} images to 8-channel folder: {OUT8_DIR}")
