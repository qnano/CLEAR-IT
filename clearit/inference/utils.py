# clearit/inference/utils.py
import os, random
import numpy as np

import torch
from torch.utils.data import DataLoader
import pandas as pd
from pathlib import Path
from typing import Optional, List

from clearit.config import MODELS_DIR, DATASETS_DIR
from clearit.inference.pipeline import load_encoder_head
from clearit.data.classification.manager import ClassificationDataManager
from clearit.models.resnet import ResNetEncoder
import yaml

# def load_encoder_only(
#     encoder_id: str,
#     device: Optional[torch.device] = None
# ) -> ResNetEncoder:
#     """
#     Load only the pretrained ResNetEncoder (no head) for inference/embeddings.
#     """
#     device = device or (torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'))

#     enc_dir = MODELS_DIR / 'encoders' / encoder_id
#     cfg = yaml.safe_load((enc_dir / 'conf_enc.yaml').read_text())
#     encoder = ResNetEncoder(
#         encoder_name     = cfg['encoder_name'],
#         encoder_features = cfg['encoder_features'],
#         mlp_layers       = cfg['mlp_layers'],
#         mlp_features     = cfg['mlp_features'],
#         in_channels      = cfg.get('in_channels', 1),
#     )
#     ckpt = torch.load(enc_dir / 'enc.pt', map_location='cpu')
#     encoder.load_state_dict(ckpt, strict=False)
#     return encoder.to(device).eval()

def load_encoder_only(
    encoder_id: str,
    device: Optional[torch.device] = None,
    proj_layers: int = 0,  # <- NEW: control how many projector layers to keep
) -> ResNetEncoder:
    device = device or (torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'))

    enc_dir = MODELS_DIR / 'encoders' / encoder_id
    cfg = yaml.safe_load((enc_dir / 'conf_enc.yaml').read_text())

    k = int(proj_layers)
    mlp_list = list(cfg.get('mlp_layers', []))[:k]

    encoder = ResNetEncoder(
        encoder_name     = cfg['encoder_name'],
        encoder_features = cfg['encoder_features'],
        mlp_layers       = mlp_list,
        mlp_features     = cfg['mlp_features'],
    )
    ckpt = torch.load(enc_dir / 'enc.pt', map_location='cpu')
    r = encoder.load_state_dict(ckpt, strict=False)
    # match old training behavior: no trained fc in ckpt → use Identity
    if "main_backbone.fc.weight" in r.missing_keys or "main_backbone.fc.bias" in r.missing_keys:
        import torch.nn as nn
        encoder.main_backbone.fc = nn.Identity()

    return encoder.to(device).eval()


# def _build_loader_from_df(
#     df_samples: pd.DataFrame,
#     dataset_name: str,
#     annotation_name: str,
#     config: dict,
#     device: torch.device,
#     num_workers: int = 0
# ):
#     """
#     Internal: build a single‐split (test_size=0) dataloader from a df.
#     Uses ClassificationDataManager, merges crop_df internally.
#     """
#     df_work = df_samples.reset_index(drop=True)
#     loader, _ = ClassificationDataManager.get_dataloader(
#         dataset_name = f"{dataset_name}",
#         df_samples   = df_work,
#         config       = config,
#         device       = device,
#         num_workers  = num_workers,
#         test_size    = 0,
#         random_state = None,
#         verbose      = False
#     )
#     return loader

def _build_loader_from_df(
    df_samples,
    dataset_name: str,
    annotation_name: str,     # kept for signature compatibility
    config: dict,
    device,
    num_workers: int = 0,
):
    """
    Build a single-split (test_size=0) DataLoader deterministically:
    - sort by (fname, cell_x, cell_y) and reset index,
    - disable shuffle in the returned DataLoader.
    """
    # stable row order
    cols = [c for c in ("fname", "cell_x", "cell_y") if c in df_samples.columns]
    if cols:
        df_samples = df_samples.sort_values(cols).reset_index(drop=True)
    else:
        df_samples = df_samples.reset_index(drop=True)

    # get the (train, None) pair from the manager
    loader_train, _ = ClassificationDataManager.get_dataloader(
        dataset_name = f"{dataset_name}",
        df_samples   = df_samples,
        config       = config,
        device       = device,
        num_workers  = max(0, int(num_workers)),
        test_size    = 0,              # single split
        random_state = None,
        verbose      = False,
    )

    # rebuild a *non-shuffling* loader over the same dataset
    loader = DataLoader(
        loader_train.dataset,
        batch_size         = loader_train.batch_size,
        shuffle            = False,                      # <— important
        num_workers        = max(0, int(num_workers)),
        pin_memory         = (str(device) == 'cuda'),
        persistent_workers = (num_workers > 0),
    )
    return loader

# def get_classification_predictions(
#     df_samples: pd.DataFrame,
#     dataset_name: str,
#     annotation_name: str,
#     encoder_id: str,
#     head_id:    str,
#     batch_size: int = 64,
#     num_workers: int = 0,
#     device: Optional[torch.device] = None
# ) -> pd.DataFrame:
#     """
#     Run a trained encoder+head on all rows in df_samples and return a DataFrame
#     with the original df columns plus:
#       - one column per sigmoid_{i} prediction
#       - one column per target_{i}
#       - cell_x, cell_y (as ints)
#     """
#     device = device or (torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'))
#     # load composite model
#     model = load_encoder_head(encoder_id, head_id, device=device)

#     # build loader
#     cfg = dict(
#         batch_size  = batch_size,
#         img_size    = int(df_samples.attrs.get('img_size', 64)),
#         label_mode  = df_samples.attrs.get('label_mode', 'multilabel'),
#         num_classes = df_samples.attrs.get('num_classes', 1),
#     )
#     loader = _build_loader_from_df(df_samples, dataset_name, annotation_name, cfg, device, num_workers)

#     records = []
#     with torch.no_grad():
#         for imgs, labs, locs, fnames in loader:
#             imgs = imgs.to(device)
#             out  = model(imgs)
#             if cfg['label_mode']=='multilabel':
#                 preds = torch.sigmoid(out).cpu().numpy()
#             else:
#                 preds = torch.softmax(out,1).cpu().numpy()
#             labs = labs.numpy()
#             xs, ys = locs

#             for b in range(imgs.size(0)):
#                 row = dict(
#                     fname    = Path(fnames[b]).name,
#                     cell_x   = int(xs[b]),
#                     cell_y   = int(ys[b]),
#                 )
#                 for i in range(cfg['num_classes']):
#                     if cfg['label_mode']=='multilabel':
#                         row[f"sigmoid_{i}"] = float(preds[b,i])
#                         row[f"target_{i}"]  = int(labs[b,i])
#                     else:
#                         row['prediction'] = int(preds[b].argmax())
#                         row['target']     = int(labs[b])
#                 records.append(row)

#     return pd.DataFrame.from_records(records)

def get_classification_predictions(
    df_samples,
    dataset_name: str,
    annotation_name: str,
    encoder_id: str,
    head_id: str,
    batch_size: int = 128,
    num_workers: int = 0,
    device=None,
):
    _ensure_determinism(0)
    device = device or (torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'))

    # build a deterministic loader
    cfg = dict(
        batch_size  = batch_size,
        img_size    = int(df_samples.attrs.get('img_size', 64)),
        label_mode  = df_samples.attrs.get('label_mode', 'multilabel'),
        num_classes = df_samples.attrs.get('num_classes', 1),
        lazy_crops  = df_samples.attrs.get('lazy_crops', False)
    )
    loader = _build_loader_from_df(df_samples, dataset_name, annotation_name, cfg, device, num_workers)

    # load model (already .eval() inside), but assert anyway
    model = load_encoder_head(encoder_id, head_id, device=device)
    model.eval()
    for m in model.modules():
        # sanity: no Dropout/BN in training mode
        if isinstance(m, (torch.nn.Dropout, torch.nn.BatchNorm2d, torch.nn.BatchNorm1d)):
            assert m.training is False

    # run inference
    import pandas as pd
    rows = []
    multilabel = (cfg['label_mode'] == 'multilabel')
    with torch.no_grad():
        for imgs, labels, (xs, ys), fnames in loader:
            imgs = imgs.to(device, non_blocking=True)
            logits = model(imgs)
            preds = torch.sigmoid(logits) if multilabel else torch.softmax(logits, dim=1)
            preds = preds.cpu().numpy()
            labels = labels.cpu().numpy()

            for i, f in enumerate(fnames):
                rec = {"fname": f, "cell_x": int(xs[i]), "cell_y": int(ys[i])}
                if multilabel:
                    for k in range(cfg['num_classes']):
                        rec[f"sigmoid_{k}"] = float(preds[i, k])
                        rec[f"target_{k}"]  = int(labels[i, k])
                else:
                    rec["prediction"] = int(preds[i].argmax())
                    rec["target"]     = int(labels[i])
                rows.append(rec)

    return pd.DataFrame(rows)

# def get_embeddings(
#     df_samples: pd.DataFrame,
#     dataset_name: str,
#     annotation_name: str,
#     encoder_id: str,
#     batch_size: int = 64,
#     num_workers: int = 0,
#     device: Optional[torch.device] = None,
# ) -> pd.DataFrame:
#     """
#     Run only the encoder over each channel‐patch in df_samples and return a DataFrame
#     with columns:
#       - fname, cell_x, cell_y
#       - embedding: flattened list of length C*F
#     """
#     device = device or (torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'))
#     encoder = load_encoder_only(encoder_id, device=device)

#     # reuse classification dataloader to get (N,C,H,W)
#     cfg = dict(
#         batch_size  = batch_size,
#         img_size    = int(df_samples.attrs.get('img_size', 64)),
#         label_mode  = df_samples.attrs.get('label_mode', 'multilabel'),
#         num_classes = df_samples.attrs.get('num_channels', 1),
#     )
#     loader = _build_loader_from_df(df_samples, dataset_name, annotation_name, cfg, device, num_workers)

#     records: List[dict] = []
#     with torch.no_grad():
#         for imgs, *_ in loader:
#             # imgs: [B, C, H, W]
#             B,C,H,W = imgs.shape
#             x = imgs.view(B*C,1,H,W).to(device)
#             x = x.repeat(1,3,1,1)               # fake-RGB
#             feats = encoder(x)                  # [B*C, F]
#             feats = feats.view(B, C, -1)        # [B, C, F]
#             concatenated = feats.reshape(B, -1).cpu().numpy()  # [B, C*F]

#             # retrieve locs+fnames from original loader batch attrs
#             xs, ys = loader.dataset.df.loc[loader.dataset.df.index[:B], ['cell_x','cell_y']].values.T
#             fnames = loader.dataset.df.loc[loader.dataset.df.index[:B], 'fname'].values

#             for b in range(B):
#                 records.append({
#                     'fname':    fnames[b],
#                     'cell_x':   int(xs[b]),
#                     'cell_y':   int(ys[b]),
#                     'embedding': concatenated[b].tolist(),
#                 })

#     return pd.DataFrame.from_records(records)

# def get_embeddings(
#     df_samples,
#     dataset_name: str,
#     annotation_name: str,
#     encoder_id: str,
#     batch_size: int = 128,
#     num_workers: int = 0,
#     proj_layers: int = 0,  # optional: how many SimCLR layers to consume
#     device=None,
# ):
#     _ensure_determinism(0)
#     device = device or (torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'))

#     # deterministic loader
#     cfg = dict(
#         batch_size  = batch_size,
#         img_size    = int(df_samples.attrs.get('img_size', 64)),
#         label_mode  = df_samples.attrs.get('label_mode', 'multilabel'),
#         num_classes = df_samples.attrs.get('num_classes', 1),
#         lazy_crops  = df_samples.attrs.get('lazy_crops', False)
#     )
#     loader = _build_loader_from_df(df_samples, dataset_name, annotation_name, cfg, device, num_workers)

#     # load encoder+head, but we’ll only use the encoder
#     model = load_encoder_head(encoder_id, head_id=None, device=device)  # if you have a separate loader for encoder-only
#     enc = model.encoder if hasattr(model, "encoder") else model
#     enc.eval()

#     # collect embeddings
#     outs = []
#     with torch.no_grad():
#         for imgs, *_ in loader:
#             imgs = imgs.to(device, non_blocking=True)
#             B, C, H, W = imgs.shape
#             feats = enc(imgs.view(B*C, 1, H, W).repeat(1, 3, 1, 1))
#             feats = feats.view(B, C, -1).reshape(B, -1)  # [B, C*F]
#             outs.append(feats.cpu())
#     return torch.cat(outs, dim=0)

# clearit/inference/utils.py
def get_embeddings(
    df_samples,
    dataset_name: str,
    annotation_name: str,
    encoder_id: str,
    batch_size: int = 128,
    num_workers: int = 0,
    proj_layers: int = 0,  # how many SimCLR proj layers to include in features
    device=None,
):
    _ensure_determinism(0)
    device = device or (torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'))

    cfg = dict(
        batch_size  = batch_size,
        img_size    = int(df_samples.attrs.get('img_size', 64)),
        label_mode  = df_samples.attrs.get('label_mode', 'multilabel'),
        num_classes = df_samples.attrs.get('num_classes', 1),
        lazy_crops  = df_samples.attrs.get('lazy_crops', False),
    )
    loader = _build_loader_from_df(df_samples, dataset_name, annotation_name, cfg, device, num_workers)

    # encoder-only, consistent with classifier loading
    enc = load_encoder_only(encoder_id, device=device, proj_layers=proj_layers)
    enc.eval()

    outs = []
    with torch.no_grad():
        for imgs, *_ in loader:
            imgs = imgs.to(device, non_blocking=True)  # [B,C,H,W]
            B, C, H, W = imgs.shape
            # same channel unroll as EncoderClassifier
            x = imgs.view(B*C, 1, H, W).repeat(1, 3, 1, 1)
            feats = enc(x)                     # [B*C, F_k]
            feats = feats.view(B, C, -1).reshape(B, -1)  # [B, C*F_k]
            outs.append(feats.cpu())
    return torch.cat(outs, dim=0)


def _ensure_determinism(seed: int = 0):
    """
    Soft, inference-friendly determinism:
    - seed Python / NumPy / Torch
    - disable cuDNN autotuner (benchmark)
    - use cuDNN deterministic kernels where available
    No torch.use_deterministic_algorithms(), so no CuBLAS env needed.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    # DO NOT call torch.use_deterministic_algorithms(True)
