# # clearit/data/classification/manager.py
# from random import randint
# from sklearn.model_selection import train_test_split
# from torch.utils.data import DataLoader

# from .dataset import SingleCellDatasetClassification
# from clearit.data.utils import extract_crops

# class ClassificationDataManager:
#     """
#     Builds train+val DataLoaders for head training.
#     Eagerly pre-extracts N-channel crops, keyed by (cache_key, crop_size).
#     """
#     crops_cache = {}

#     @staticmethod
#     def get_dataloader(dataset_name,
#                        df_samples,
#                        config,
#                        device='cuda',
#                        num_workers=0,
#                        test_size=0.2,
#                        random_state=None,
#                        verbose=False):
#         # core params
#         batch_size = config['batch_size']
#         img_size   = config['img_size']
#         label_mode = config['label_mode']
#         num_classes= config['num_classes']

#         # before the cache_name
#         key_cols = [c for c in ("fname","cell_x","cell_y") if c in df_samples.columns]
#         fingerprint = hash(tuple(map(tuple, df_samples[key_cols].to_numpy())))
#         cache_name = f"{user_key}|eager|{img_size}|{fingerprint}"


#         if cache_name in ClassificationDataManager.crops_cache:
#             if verbose:
#                 print(f"[cache hit] loading crops for {cache_name}")
#             crop_df, crops_tensor = ClassificationDataManager.crops_cache[cache_name]
#         else:
#             if verbose:
#                 print(f"[cache miss] extracting crops for {cache_name}")
#             crop_df, crops_tensor = extract_crops(
#                 df_samples,
#                 dataset_name=dataset_name,
#                 crop_size=img_size,
#                 mode='all_channels',
#                 max_workers=num_workers
#             )
#             ClassificationDataManager.crops_cache[cache_name] = (crop_df, crops_tensor)

#         # merge so every row knows its `crop_tensor_index`
#         df = df_samples.merge(
#             crop_df,
#             how='left',
#             left_index=True,
#             right_on='original_index'
#         )

#         # split train/val
#         if test_size > 0:
#             df_train, df_val = train_test_split(
#                 df, test_size=test_size,
#                 random_state=random_state or randint(0,2**32-1)
#             )
#         else:
#             df_train, df_val = df, None

#         # build datasets
#         ds_train = SingleCellDatasetClassification(
#             crops_tensor, df_train, label_mode, num_classes, dataset_name
#         )
#         loader_train = DataLoader(
#             ds_train,
#             batch_size=batch_size,
#             shuffle=True,
#             num_workers=num_workers,
#             pin_memory=(device=='cuda'),
#             persistent_workers=(num_workers>0)
#         )

#         if df_val is not None:
#             ds_val = SingleCellDatasetClassification(
#                 crops_tensor, df_val, label_mode, num_classes, dataset_name+"_val"
#             )
#             loader_val = DataLoader(
#                 ds_val,
#                 batch_size=batch_size,
#                 shuffle=False,
#                 num_workers=num_workers,
#                 pin_memory=(device=='cuda'),
#                 persistent_workers=(num_workers>0)
#             )
#         else:
#             loader_val = None

#         return loader_train, loader_val

# clearit/data/classification/manager.py
from random import randint
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Subset

from .dataset import SingleCellDatasetClassification, ClassificationCropDataset
from clearit.data.utils import extract_crops
import hashlib

class ClassificationDataManager:
    """
    Builds train+val DataLoaders for classification heads.

    Modes:
      • Eager (default): pre-extract all N-channel crops to a big tensor.
      • Lazy (config['lazy_crops']=True): crop on-the-fly from full images
        using an LRU-cached per-image pad, nearly as fast as eager preload.

    Caches:
      • Eager: keeps (crop_df, crops_tensor) in memory, keyed by dataset & crop size & df fingerprint.
      • Lazy: keeps a base ClassificationCropDataset (with LRU image cache), keyed similarly.
    """
    crops_cache = {}     # for eager tensors
    dataset_cache = {}   # for lazy base datasets

    @staticmethod
    def _fingerprint_df(df, key_cols):
        """
        Order-independent hash of the rows that define unique crops.
        Ensures cache re-use even if df is shuffled.
        """
        key_df = df.loc[:, key_cols].copy()
        key_df = key_df.sort_values(key_cols).reset_index(drop=True)
        s = key_df.astype(str).agg('|'.join, axis=1).str.cat(sep=';')
        return hashlib.sha1(s.encode('utf-8')).hexdigest()

    @staticmethod
    def get_dataloader(dataset_name,
                       df_samples,
                       config,
                       device='cuda',
                       num_workers=0,
                       test_size=0.2,
                       random_state=None,
                       verbose=False):

        # core params
        batch_size  = config['batch_size']
        img_size    = config['img_size']
        label_mode  = config['label_mode']
        num_classes = config['num_classes']
        lazy_mode   = bool(config.get('lazy_crops', False))

        key_cols = ["fname", "cell_x", "cell_y"]
        user_key = config.get('cache_key', dataset_name)
        df_sig   = ClassificationDataManager._fingerprint_df(df_samples, key_cols)
        mode_flag = 'lazy' if lazy_mode else 'eager'
        cache_name = f"{user_key}|{mode_flag}|{img_size}|{df_sig}"

        if lazy_mode:
            # ----- LAZY: on-the-fly cropping over a reusable base dataset -----
            if cache_name in ClassificationDataManager.dataset_cache:
                if verbose: print(f"[cache hit] {cache_name}")
                base_ds = ClassificationDataManager.dataset_cache[cache_name]
            else:
                if verbose: print(f"[cache miss] {cache_name} → building ClassificationCropDataset (lazy)")
                base_ds = ClassificationCropDataset(
                    df_samples=df_samples,
                    dataset_name=dataset_name,
                    crop_size=img_size,
                    label_mode=label_mode,
                    num_classes=num_classes,
                    scale_col='scale',
                    transform=None
                )
                ClassificationDataManager.dataset_cache[cache_name] = base_ds

            # Split into train/val with Subset wrappers (share the same LRU image cache)
            if test_size > 0:
                idx_all = list(range(len(base_ds)))
                idx_train, idx_val = train_test_split(
                    idx_all, test_size=test_size, random_state=random_state or randint(0, 2**32 - 1)
                )
                ds_train = Subset(base_ds, idx_train)
                ds_val   = Subset(base_ds, idx_val)
            else:
                ds_train = base_ds
                ds_val   = None

            loader_train = DataLoader(
                ds_train,
                batch_size=batch_size,
                shuffle=True,
                num_workers=num_workers,
                pin_memory=(device == 'cuda'),
                persistent_workers=(num_workers > 0)
            )
            loader_val = None
            if ds_val is not None:
                loader_val = DataLoader(
                    ds_val,
                    batch_size=batch_size,
                    shuffle=False,
                    num_workers=num_workers,
                    pin_memory=(device == 'cuda'),
                    persistent_workers=(num_workers > 0)
                )
            return loader_train, loader_val

        # ----- EAGER: pre-extract all crops (existing path, kept intact) -----
        if cache_name in ClassificationDataManager.crops_cache:
            if verbose:
                print(f"[cache hit] loading crops for {cache_name}")
            crop_df, crops_tensor = ClassificationDataManager.crops_cache[cache_name]
        else:
            if verbose:
                print(f"[cache miss] extracting crops for {cache_name}")
            crop_df, crops_tensor = extract_crops(
                df_samples,
                dataset_name=dataset_name,
                crop_size=img_size,
                mode='all_channels',
                max_workers=num_workers
            )
            ClassificationDataManager.crops_cache[cache_name] = (crop_df, crops_tensor)

        # Merge on keys (order-invariant), not on the DataFrame index
        df = df_samples.merge(crop_df, on=key_cols, how='left', validate='many_to_one')

        # split train/val
        if test_size > 0:
            df_train, df_val = train_test_split(
                df, test_size=test_size,
                random_state=random_state or randint(0, 2**32 - 1)
            )
        else:
            df_train, df_val = df, None

        # build datasets
        ds_train = SingleCellDatasetClassification(
            crops_tensor, df_train, label_mode, num_classes, dataset_name
        )
        loader_train = DataLoader(
            ds_train,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=(device == 'cuda'),
            persistent_workers=(num_workers > 0)
        )

        loader_val = None
        if df_val is not None:
            ds_val = SingleCellDatasetClassification(
                crops_tensor, df_val, label_mode, num_classes, dataset_name + "_val"
            )
            loader_val = DataLoader(
                ds_val,
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers,
                pin_memory=(device == 'cuda'),
                persistent_workers=(num_workers > 0)
            )

        return loader_train, loader_val
