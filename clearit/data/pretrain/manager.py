# clearit/data/pretrain/manager.py
from torch.utils.data import DataLoader

from .dataset import SingleCellDatasetPretrain, PretrainCropDataset
from clearit.data.utils import extract_crops
from clearit.augmentations.utils import get_crop_size_preload

class PretrainDataManager:
    # Holds any DataLoader we've built, keyed by (cache_key, mode, crop_size)
    dataset_cache = {}

    @staticmethod
    def get_dataloader(dataset_name,
                       df_samples,
                       config,
                       num_workers=0,
                       verbose=False):
        """
        Returns a PyTorch DataLoader for SimCLR pretraining.

        If `lazy_crops=True` in config, uses PretrainCropDataset
        to crop on‐the‐fly. Otherwise, eagerly pre‐extracts every
        single‐channel patch into a big tensor.

        The result is cached (per unique cache_key + mode + crop_size)
        so repeated calls don’t reload or re‐pad.
        """
        # pull core params
        batch_size = config['batch_size']
        img_size   = config['img_size']
        transforms = config['transforms']
        lazy_mode  = bool(config.get('lazy_crops', False))

        # build a unique cache key for this exact dataset + mode
        user_key = config.get('cache_key', dataset_name)
        mode_flag = 'lazy' if lazy_mode else 'eager'
        # for eager: we’ll compute an expanded crop_size; for lazy we use img_size
        crop_size = img_size if lazy_mode else get_crop_size_preload(transforms, img_size)
        cache_name = f"{user_key}|{mode_flag}|{crop_size}"

        # hit? return immediately
        if cache_name in PretrainDataManager.dataset_cache:
            if verbose:
                print(f"[cache hit] {cache_name}")
            return PretrainDataManager.dataset_cache[cache_name]

        # miss → build it
        if verbose:
            print(f"[cache miss] building DataLoader for {cache_name}")

        # 1) Lazy: on‐the‐fly cropping
        if lazy_mode:
            if verbose:
                print(f"  → using lazy PretrainCropDataset (on‐the‐fly)")

            ds = PretrainCropDataset(
                df_samples=df_samples,
                dataset_name=dataset_name,
                crop_size=img_size,
                transform=None
            )
            loader = DataLoader(
                ds,
                batch_size=batch_size,
                shuffle=True,
                num_workers=num_workers,
                pin_memory=True,
                persistent_workers=(num_workers > 0)
            )

        # 2) Eager: pre‐extract everything up front
        else:
            if verbose:
                print(f"  → pre‐extracting all crops (crop_size={crop_size})")

            crop_df, crops_tensor = extract_crops(
                df_samples,
                dataset_name=dataset_name,
                crop_size=crop_size,
                max_workers=num_workers
            )
            # merge to get the crop_tensor_index column
            df_updated = df_samples.merge(
                crop_df,
                how='left',
                left_index=True,
                right_on='original_index'
            )
            ds = SingleCellDatasetPretrain(
                crops_tensor=crops_tensor,
                df_samples=df_updated,
                dataset_name=dataset_name
            )
            loader = DataLoader(
                ds,
                batch_size=batch_size,
                shuffle=True,
                num_workers=num_workers,
                pin_memory=True,
                persistent_workers=(num_workers > 0)
            )

        # cache & return
        PretrainDataManager.dataset_cache[cache_name] = loader
        return loader
