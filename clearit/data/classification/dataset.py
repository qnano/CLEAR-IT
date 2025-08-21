# clearit/data/classification/dataset.py
import ast
import torch
from torch.utils.data import Dataset
import tifffile
import numpy as np
from pathlib import Path
from functools import lru_cache

from clearit.config import DATASETS_DIR


class SingleCellDatasetClassification(Dataset):
    """
    Given a big tensor of N crops (each of shape [C,H,W]) and a DataFrame
    with `crop_tensor_index`, `label`, `cell_x`, `cell_y`, `fname`,
    returns (img, label, locs, fname) per __getitem__.
    """
    def __init__(self, crops_tensor, df_samples, label_mode, num_classes, dataset_name):
        self.crops_tensor = crops_tensor
        self.df = df_samples
        self.label_mode = label_mode
        self.num_classes = num_classes
        self.dataset_name = dataset_name

    def __getitem__(self, idx):
        # Access the row in the dataframe corresponding to this sample
        row = self.df.iloc[idx]
        crop_id = row['crop_tensor_index']
        img_out = self.crops_tensor[crop_id].float()

        lbl = row['label']
        # if it came in as a string from CSV, parse it
        if isinstance(lbl, str):
            lbl = ast.literal_eval(lbl)

        # Now convert to tensor according to mode
        if self.label_mode == "multilabel":
            # lbl should now be a list of 0/1
            label = torch.tensor(lbl, dtype=torch.float32)
        else:  # multiclass: lbl should be an int
            label = torch.tensor(int(lbl), dtype=torch.long)

        locs = (row['cell_x'], row['cell_y'])
        fname = row['fname']

        return img_out, label, locs, fname

    def __len__(self):
        return len(self.df)

# class ClassificationCropDataset(Dataset):
#     """
#     Lazy on-the-fly cropping for classification:
#     - Loads/pads full multi-channel images on demand (LRU-cached).
#     - Returns an all-channel crop: shape (C, H, W), float32 in [0,1].
#     - Emits (img, label, (cell_x, cell_y), fname), identical to the eager dataset.
#     """

#     def __init__(self, df_samples, dataset_name, crop_size=64,
#                  label_mode='multilabel', num_classes=1,
#                  scale_col='scale', transform=None):
#         # Ensure a scale column is present (project convention: 0..255 range)
#         df = df_samples.copy()
#         if scale_col not in df.columns:
#             df[scale_col] = 255.0

#         self.df = df.reset_index(drop=True)
#         self.dataset_name = dataset_name
#         self.crop_size = crop_size
#         self.half = crop_size // 2
#         self.scale_col = scale_col
#         self.label_mode = label_mode
#         self.num_classes = num_classes
#         self.transform = transform

#     def __len__(self):
#         return len(self.df)

#     @lru_cache(maxsize=32)
#     def _load_and_pad_image(self, stem):
#         """Load a full multi-channel TIFF, reflect-pad by half the crop on each side."""
#         images_dir = Path(DATASETS_DIR) / self.dataset_name / 'images'
#         arr = None
#         for ext in ('.tiff', '.tif'):
#             p = images_dir / f'{stem}{ext}'
#             if p.exists():
#                 arr = tifffile.imread(str(p))
#                 break
#         if arr is None:
#             raise FileNotFoundError(f'No image for {stem} in {images_dir}')
#         # arr: (C, H, W) → pad to (C, H+2h, W+2h)
#         return np.pad(arr,
#                       ((0, 0), (self.half, self.half), (self.half, self.half)),
#                       mode='reflect')

#     def __getitem__(self, idx):
#         row = self.df.iloc[idx]
#         stem = row['fname']
#         padded = self._load_and_pad_image(stem)

#         x, y = int(row['cell_x']), int(row['cell_y'])
#         crop_np = padded[:, y: y + self.crop_size, x: x + self.crop_size]  # (C, H, W)

#         # Normalize to [0,1] using the provided scalar scale
#         crop_t = torch.from_numpy(crop_np.astype(np.float32) / float(row[self.scale_col]))

#         # Parse label
#         lbl = row['label']
#         if isinstance(lbl, str):
#             lbl = ast.literal_eval(lbl)

# clearit/data/classification/dataset.py

import ast
import math
import numpy as np
import torch
from torch.utils.data import Dataset
import tifffile
from pathlib import Path
from functools import lru_cache

from clearit.config import DATASETS_DIR

class ClassificationCropDataset(Dataset):
    """
    Lazy on-the-fly cropping for classification:
    Returns (img, label, (cell_x,cell_y), fname) where
      img: FloatTensor [C,H,W] in [0,1]
    """

    def __init__(self, df_samples, dataset_name, crop_size=64,
                 label_mode='multilabel', num_classes=1,
                 scale_col='scale', transform=None):
        df = df_samples.copy()
        if scale_col not in df.columns:
            df[scale_col] = 255.0
        self.df = df.reset_index(drop=True)
        self.dataset_name = dataset_name
        self.crop_size = crop_size
        self.half = crop_size // 2
        self.scale_col = scale_col
        self.label_mode = label_mode
        self.num_classes = int(num_classes)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    @lru_cache(maxsize=32)
    def _load_and_pad_image(self, stem: str):
        images_dir = Path(DATASETS_DIR) / self.dataset_name / 'images'
        arr = None
        for ext in ('.tiff', '.tif'):
            p = images_dir / f'{stem}{ext}'
            if p.exists():
                arr = tifffile.imread(str(p))
                break
        if arr is None:
            raise FileNotFoundError(f'No image for {stem} in {images_dir}')
        return np.pad(arr,
                      ((0, 0), (self.half, self.half), (self.half, self.half)),
                      mode='reflect')

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # --- required fields
        stem = row.get('fname', None)
        x = row.get('cell_x', None)
        y = row.get('cell_y', None)
        if stem is None or (isinstance(stem, float) and math.isnan(stem)) or str(stem).strip()=="":
            raise ValueError(f"[ClassificationCropDataset] Missing fname at idx={idx}")
        if x is None or y is None or (isinstance(x, float) and math.isnan(x)) or (isinstance(y, float) and math.isnan(y)):
            raise ValueError(f"[ClassificationCropDataset] Missing cell_x/cell_y at idx={idx}, fname={stem}")

        # --- crop all channels
        padded = self._load_and_pad_image(str(stem))
        x, y = int(x), int(y)
        crop_np = padded[:, y:y+self.crop_size, x:x+self.crop_size]  # (C,H,W)

        # --- normalize
        scale = row.get(self.scale_col, 255.0)
        try:
            scale = float(scale)
        except Exception as e:
            raise ValueError(f"[ClassificationCropDataset] Non-numeric scale at idx={idx}, fname={stem}: {scale}") from e
        if not np.isfinite(scale) or scale <= 0:
            raise ValueError(f"[ClassificationCropDataset] Invalid scale={scale} at idx={idx}, fname={stem}")

        crop_t = torch.from_numpy(crop_np.astype(np.float32) / scale)

        # --- parse label
        lbl_raw = row.get('label', None)
        if self.label_mode == "multilabel":
            if isinstance(lbl_raw, str):
                try:
                    lbl_list = ast.literal_eval(lbl_raw)
                except Exception as e:
                    raise ValueError(f"[ClassificationCropDataset] Could not parse multilabel string at idx={idx}, fname={stem}: {lbl_raw}") from e
            elif isinstance(lbl_raw, (list, tuple, np.ndarray)):
                lbl_list = list(lbl_raw)
            else:
                raise ValueError(f"[ClassificationCropDataset] Expected multilabel list, got {type(lbl_raw)} at idx={idx}, fname={stem}")
            if len(lbl_list) != self.num_classes:
                raise ValueError(f"[ClassificationCropDataset] Multilabel length {len(lbl_list)} != num_classes {self.num_classes} at idx={idx}, fname={stem}")
            label_t = torch.tensor([float(v) for v in lbl_list], dtype=torch.float32)
        else:
            try:
                ycls = int(lbl_raw)
            except Exception as e:
                raise ValueError(f"[ClassificationCropDataset] Could not parse multiclass label at idx={idx}, fname={stem}: {lbl_raw}") from e
            label_t = torch.tensor(ycls, dtype=torch.long)

        # --- optional transform
        if self.transform:
            crop_t = self.transform(crop_t)

        locs = (int(row['cell_x']), int(row['cell_y']))
        fname = str(stem)

        # IMPORTANT: always return the 4-tuple
        return crop_t, label_t, locs, fname
