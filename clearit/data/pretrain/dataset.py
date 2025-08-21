# clearit/data/pretrain/dataset.py
import torch
from torch.utils.data import Dataset
import tifffile
import numpy as np
from pathlib import Path
from functools import lru_cache

from clearit.config import DATASETS_DIR

class SingleCellDatasetPretrain(Dataset):
    """Eager-loaded: all crops are precomputed and stored in a tensor."""
    def __init__(self, crops_tensor, df_samples, dataset_name):
        self.crops_tensor = crops_tensor
        self.df = df_samples
        self.dataset_name = dataset_name

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        crop_id = row['crop_tensor_index']
        # add channel‐axis and float
        return self.crops_tensor[crop_id].float()

    def __len__(self):
        return len(self.df)


class PretrainCropDataset(Dataset):
    """
    Lazy on‐the‐fly cropping: loads, pads, caches up to 4 images,
    and extracts one cell crop per __getitem__.
    """
    def __init__(self, df_samples, dataset_name, crop_size=64,
                 scale_col='scale', transform=None):
        # Ensure there’s always a scale column
        df = df_samples.copy()
        if scale_col not in df.columns:
            df[scale_col] = 255.0
        self.df = df.reset_index(drop=True)
        self.dataset_name = dataset_name
        self.crop_size = crop_size
        self.half = crop_size // 2
        self.scale_col = scale_col
        self.transform = transform


    def __len__(self):
        return len(self.df)

    @lru_cache(maxsize=32)
    def _load_and_pad_image(self, stem):
        images_dir = Path(DATASETS_DIR) / self.dataset_name / 'images'
        arr = None
        for ext in ('.tiff', '.tif'):
            p = images_dir / f'{stem}{ext}'
            if p.exists():
                arr = tifffile.imread(str(p))
                break
        if arr is None:
            raise FileNotFoundError(f'No image for {stem}')
        return np.pad(
            arr,
            ((0,0), (self.half,self.half), (self.half,self.half)),
            mode='reflect'
        )

    def __getitem__(self, idx):
        row  = self.df.iloc[idx]
        stem = row['fname']
        padded = self._load_and_pad_image(stem)

        x, y = int(row['cell_x']), int(row['cell_y'])
        ch   = int(row['channel'])     # <-- pick only this one

        crop_np = padded[ch,
                         y : y+self.crop_size,
                         x : x+self.crop_size]

        # normalize and add a channel‐axis
        crop_t = torch.from_numpy(crop_np / row[self.scale_col]) \
                    .unsqueeze(0)                    \
                    .float()

        if self.transform:
            crop_t = self.transform(crop_t)
        return crop_t
