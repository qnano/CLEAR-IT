# clearit/shap/utils.py
from pathlib import Path
from typing import Optional, Tuple
import yaml
import torch
from torch.utils.data import DataLoader
from torch.utils.data._utils.collate import default_collate

from clearit.config import MODELS_DIR, OUTPUTS_DIR
from clearit.inference.pipeline import load_encoder_head
from clearit.data.classification.dataset import ClassificationCropDataset

def load_model_from_test(test_id: str, device: Optional[torch.device] = None):
    """
    Resolve head_id & encoder_id via outputs/tests/<test_id>/conf_test.yaml
    and return (model, test_cfg, head_cfg). Model is eval() on `device`.
    """
    device = device or (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
    test_cfg_path = Path(OUTPUTS_DIR) / "tests" / test_id / "conf_test.yaml"
    if not test_cfg_path.exists():
        raise FileNotFoundError(f"Missing {test_cfg_path}")
    test_cfg = yaml.safe_load(test_cfg_path.read_text())
    head_id = test_cfg["head_id"]
    encoder_id = test_cfg.get("encoder_id")

    head_cfg_path = Path(MODELS_DIR) / "heads" / head_id / "conf_head.yaml"
    if not head_cfg_path.exists():
        raise FileNotFoundError(f"Missing {head_cfg_path}")
    head_cfg = yaml.safe_load(head_cfg_path.read_text())

    if not encoder_id:
        encoder_id = head_cfg.get("base_encoder") or head_cfg.get("encoder_id")
        if not encoder_id:
            raise KeyError("encoder_id not found in test/head configs")

    model = load_encoder_head(encoder_id, head_id, device=device)
    model.eval()
    return model, test_cfg, head_cfg

def _collate_skip_none(batch):
    batch = [b for b in batch if b is not None]
    if not batch:
        raise RuntimeError("All samples in batch were None/invalid.")
    return default_collate(batch)

def get_dataloader_for_shap(
    df_samples,
    *,
    dataset_name: str,
    patch_size: int,
    label_mode: str,
    num_classes: int,
    batch_size: int = 64,
    num_workers: int = 0,
    persistent_workers: bool = False,
) -> Tuple[DataLoader, torch.utils.data.Dataset]:
    """
    Deterministic, non-shuffling DataLoader over df_samples using lazy crops.
    Yields (img [B,C,H,W], label, (x,y), fname).
    """
    ds = ClassificationCropDataset(
        df_samples=df_samples,
        dataset_name=dataset_name,
        crop_size=patch_size,
        label_mode=label_mode,
        num_classes=num_classes,
        scale_col="scale",
        transform=None,
    )
    loader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=(num_workers > 0 and persistent_workers),
        collate_fn=_collate_skip_none,
    )
    return loader, ds
