import torch
import torch.nn as nn
import yaml

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

    # Load the encoder config and checkpoint.
    enc_dir = MODELS_DIR / 'encoders' / encoder_id
    enc_cfg = yaml.safe_load((enc_dir / 'conf_enc.yaml').read_text())
    enc_ckpt = torch.load(enc_dir / 'enc.pt', map_location='cpu')

    # Load the head config and checkpoint.
    head_dir = MODELS_DIR / 'heads' / head_id
    head_cfg = yaml.safe_load((head_dir / 'conf_head.yaml').read_text())
    head_ckpt = torch.load(head_dir / 'head.pt', map_location='cpu')

    # Build the encoder with the projection depth expected by the head.
    k = int(head_cfg.get('proj_layers', 0))
    mlp_list = list(enc_cfg.get('mlp_layers', []))[:k]
    encoder = ResNetEncoder(
        encoder_name     = enc_cfg['encoder_name'],
        encoder_features = enc_cfg['encoder_features'],
        mlp_layers       = mlp_list,
        mlp_features     = enc_cfg['mlp_features'],
    )
    enc_result = encoder.load_state_dict(enc_ckpt, strict=False)
    print(f"[encoder] missing={len(enc_result.missing_keys)} unexpected={len(enc_result.unexpected_keys)}")
    if enc_result.missing_keys:
        print("  eg missing:", enc_result.missing_keys[:8])
    if enc_result.unexpected_keys:
        print("  eg unexpected:", enc_result.unexpected_keys[:8])

    # Some checkpoints omit the stored fc layer weights.
    if "main_backbone.fc.weight" in enc_result.missing_keys or "main_backbone.fc.bias" in enc_result.missing_keys:
        print("[encoder] checkpoint has no fc weights -> using Identity fc for this checkpoint layout")
        encoder.main_backbone.fc = nn.Identity()

    encoder.to(device).eval()

    # Build the head with the expected input dimension.
    feat_dim = encoder.get_feature_size(k) * head_cfg['num_channels']
    head = MLPHead(
        input_size  = feat_dim,
        num_classes = head_cfg['num_classes'],
        dropout     = head_cfg.get('dropout', 0.0),
        head_layers = head_cfg.get('head_layers', []),
    )

    # Validate the head checkpoint strictly before composing the model.
    head.load_state_dict(head_ckpt, strict=True)
    print("[head] strict load ok.")
    head.to(device).eval()

    # Compose the inference model.
    model = EncoderClassifier(encoder=encoder, classification_head=head)
    model.to(device).eval()
    return model
