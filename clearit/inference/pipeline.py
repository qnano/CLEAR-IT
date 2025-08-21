# clearit/inference/pipeline.py
from pathlib import Path
import yaml
import torch
import torch.nn as nn

from clearit.config           import MODELS_DIR
from clearit.models.resnet    import ResNetEncoder
from clearit.models.head      import MLPHead
from clearit.models.composite import EncoderClassifier

# def load_encoder_head(
#     encoder_id: str,
#     head_id:    str,
#     device:     torch.device = None,
# ) -> EncoderClassifier:
#     """
#     Load a trained encoder + head into a composite model for inference.
#     Respects `proj_layers` in the head config so that when proj_layers==0
#     no SimCLR MLP is applied (i.e. you get the raw encoder_features).
#     """
#     device = device or (torch.device('cuda') if torch.cuda.is_available() 
#                                      else torch.device('cpu'))

#     # 1) load encoder config
#     enc_dir = MODELS_DIR / 'encoders' / encoder_id
#     enc_cfg = yaml.safe_load((enc_dir / 'conf_enc.yaml').read_text())

#     # 2) pick how many of the SimCLR‐MLP layers to include
#     proj = enc_cfg.get('proj_layers', 0)  # or head_cfg['proj_layers']
#     # you may also want to read proj from the head config:
#     head_cfg = yaml.safe_load((MODELS_DIR / 'heads' / head_id / 'conf_head.yaml').read_text())
#     proj = head_cfg.get('proj_layers', 0)

#     # 3) build exactly the same encoder—with only the first `proj` MLP layers
#     full_mlp = enc_cfg.get('mlp_layers', [])              # e.g. [512,512,512]
#     mlp_list = full_mlp[:proj]                            # [] if proj=0
#     encoder = ResNetEncoder(
#         encoder_name     = enc_cfg['encoder_name'],
#         encoder_features = enc_cfg['encoder_features'],
#         mlp_layers       = mlp_list,                      # <= critical
#         mlp_features     = enc_cfg['mlp_features'],
#     )
#     ckpt = torch.load(enc_dir / 'enc.pt', map_location='cpu')
#     encoder.load_state_dict(ckpt, strict=False)
#     encoder.to(device).eval()

#     # 4) now build the head
#     feature_size = encoder.get_feature_size(proj) * head_cfg['num_channels']
#     head = MLPHead(
#         input_size  = feature_size,
#         num_classes = head_cfg['num_classes'],
#         dropout     = head_cfg['dropout'],
#         head_layers = head_cfg.get('head_layers', []),
#     )
#     head_ckpt = torch.load(MODELS_DIR / 'heads' / head_id / 'head.pt',
#                            map_location='cpu')
#     head.load_state_dict(head_ckpt)
#     head.to(device).eval()

#     # 5) glue and return
#     model = EncoderClassifier(encoder=encoder,
#                               classification_head=head)
#     return model.to(device).eval()

# clearit/inference/pipeline.py
from pathlib import Path
import yaml
import torch

from clearit.config           import MODELS_DIR
from clearit.models.resnet    import ResNetEncoder
from clearit.models.head      import MLPHead
from clearit.models.composite import EncoderClassifier

def load_encoder_head(
    encoder_id: str,
    head_id:    str,
    device:     torch.device = None,
) -> EncoderClassifier:
    """
    Load a trained encoder and head into a composite model for inference.
    Ensures the encoder uses exactly the same number of projection MLP layers
    that the head expects (head_cfg['proj_layers']).
    """
    device = device or (torch.device('cuda') if torch.cuda.is_available()
                         else torch.device('cpu'))

    # 1) Encoder config + checkpoint
    enc_dir = MODELS_DIR / 'encoders' / encoder_id
    enc_cfg = yaml.safe_load((enc_dir / 'conf_enc.yaml').read_text())
    enc_ckpt = torch.load(enc_dir / 'enc.pt', map_location='cpu')

    # 2) Head config + checkpoint
    head_dir = MODELS_DIR / 'heads' / head_id
    head_cfg = yaml.safe_load((head_dir / 'conf_head.yaml').read_text())
    head_ckpt = torch.load(head_dir / 'head.pt', map_location='cpu')

    # 3) Build encoder with EXACTLY k projection layers used by the head
    k = int(head_cfg.get('proj_layers', 0))
    mlp_list = list(enc_cfg.get('mlp_layers', []))[:k]  # [] if k==0
    encoder = ResNetEncoder(
        encoder_name     = enc_cfg['encoder_name'],
        encoder_features = enc_cfg['encoder_features'],
        mlp_layers       = mlp_list,                    # <-- align with head
        mlp_features     = enc_cfg['mlp_features'],
    )
    encoder.load_state_dict(enc_ckpt, strict=False)

    # <<< add this
    enc_result = encoder.load_state_dict(enc_ckpt, strict=False)
    print(f"[encoder] missing={len(enc_result.missing_keys)} unexpected={len(enc_result.unexpected_keys)}")
    if enc_result.missing_keys:
        print("  eg missing:", enc_result.missing_keys[:8])
    if enc_result.unexpected_keys:
        print("  eg unexpected:", enc_result.unexpected_keys[:8])
    # >>>

    # >>> NEW: if the checkpoint didn't have fc weights, match old behavior (no fc)
    if "main_backbone.fc.weight" in enc_result.missing_keys or "main_backbone.fc.bias" in enc_result.missing_keys:
        print("[encoder] checkpoint has no fc weights -> using Identity fc to match old training")
        encoder.main_backbone.fc = nn.Identity()
        # (Dimension remains 512 for resnet18; encoder.get_feature_size(0) stays correct.)

    encoder.to(device).eval()

    # 4) Build head with the right input dim
    feat_dim = encoder.get_feature_size(k) * head_cfg['num_channels']
    head = MLPHead(
        input_size  = feat_dim,
        num_classes = head_cfg['num_classes'],
        dropout     = head_cfg.get('dropout', 0.0),
        head_layers = head_cfg.get('head_layers', []),
    )

    # <<< use strict=True here so we *fail fast* if shapes or names don’t line up
    head_result = head.load_state_dict(head_ckpt, strict=True)
    print("[head] strict load ok.")
    # >>>

    head.load_state_dict(head_ckpt)
    head.to(device).eval()

    # 5) Compose
    model = EncoderClassifier(encoder=encoder, classification_head=head)
    model.to(device).eval()
    return model
