# clearit/augmentations/factory.py
from __future__ import annotations

import math
import kornia
from .kornia_extra import ColorJitterSimple, RandomRadialTranslate, RandomRightAngleRotation
from .utils import make_transformdict


def get_augmentations(transformdict: dict = None, img_size: int = 64):
    """
    Build a Kornia augmentation pipeline for CLEAR-IT patches.

    The composed transform applies, in order:
      1) Gaussian blur (if gaussian_blur > 0)
      2) Color jitter (if any of brightness/contrast/saturation/hue > 0)
      3) Radial translation (if translate != 0)
      4) Random zoom in/out via affine scale (if zoomfactor_min/max != 1)
      5) CenterCrop back to `img_size` if (3) or (4) were applied
      6) Random right-angle rotation: uniformly samples one of {0°,90°,180°,270°}
         per image when `rotate` is enabled. (Implemented via RandomRightAngleRotation.)
      7) Random horizontal and vertical flips, each with probability 0.5 when `flip` is enabled.

    Notes
    -----
    - "Enabled" means the corresponding value in `transformdict` is non-zero.
    - Rotations are applied *before* flips to ensure a uniform distribution
      over the four orientations when both are enabled.
    - Input/Output tensors are expected as (B, C, H, W).
    - Defaults are read from `configs/pretrainer_defaults.yaml` → `transforms`
      and then updated with the provided `transformdict` (if any).
      Unknown keys in `transformdict` are ignored by this function and
      passed through harmlessly.

    Parameters
    ----------
    transformdict : dict, optional
        Per-augmentation controls. If omitted, YAML defaults are used.
        Recognized keys (all numeric):
          • 'gaussian_blur'   : kernel radius r (0 = off). We use
                                kernel_size = 2r+1, sigma ∈ (0, r].
          • 'brightness'      : factor amplitude in [0, ∞), 0 = off.
          • 'contrast'        : factor amplitude in [0, ∞), 0 = off.
          • 'saturation'      : factor amplitude in [0, ∞), 0 = off.
          • 'hue'             : factor amplitude in [0, ∞), 0 = off.
          • 'translate'       : max radial shift (pixels). 0 = no translation.
                                Direction is uniform in [0°, 360°).
          • 'zoomfactor_min'  : ≥ 1.0 → maximum zoom-in (smaller FOV).
          • 'zoomfactor_max'  : ≥ 1.0 → maximum zoom-out (larger FOV).
                                We map (zoomfactor_min, zoomfactor_max) into an
                                affine scale range s ∈ [1/√zoom_max, 1/√zoom_min].
          • 'flip'            : 0 or 1. If 1, apply independent H/V flips with
                                p = 0.5 each.
          • 'rotate'          : 0 or 1. If 1, apply a 90° rotation with p = 0.5;
                                when 'flip' == 0 we also add a 180° rotation with p = 0.5.

    img_size : int, default 64
        The final patch size after augmentations. If translation and/or zoom are
        enabled, a CenterCrop to `img_size` is appended to preserve shape.

    Returns
    -------
    kornia.augmentation.container.ImageSequential
        A callable augmentation pipeline with `same_on_batch=False`
        (randomized *per sample*). All enabled steps have p=1 unless stated
        otherwise above.

    Notes
    -----
    • Expected input: torch.Tensor of shape (B, C, H, W) with float values,
      typically in [0, 1]. Works with grayscale or multi-channel data.
    • This function does *not* resize; it only crops back to `img_size`
      after translate/zoom. Upstream code should ensure the preloaded crop is
      large enough. See `get_crop_size_preload(...)` for how to compute that.
    • Changing defaults only requires editing the YAML; no code changes needed.

    Examples
    --------
    >>> from clearit.augmentations.factory import get_augmentations
    >>> aug = get_augmentations({'translate': 4, 'zoomfactor_max': 1.2}, img_size=64)
    >>> x_aug = aug(x)  # x: (B, C, 64, 64), returns same shape
    """

    tdict = make_transformdict(transformdict or {})

    augs = []
    crop_needed = False
    crop_size = img_size

    # 1) Gaussian blur
    if tdict.get("gaussian_blur", 0) != 0:
        k = int(tdict["gaussian_blur"]) * 2 + 1
        augs.append(
            kornia.augmentation._2d.intensity.RandomGaussianBlur(
                kernel_size=(k, k),
                sigma=(0, float(tdict["gaussian_blur"])),
                separable=True,
                border_type="reflect",
                p=1.0,
                same_on_batch=False,
            )
        )

    # 2) Color jitter
    if any(float(tdict.get(k, 0)) != 0 for k in ("brightness", "contrast", "saturation", "hue")):
        augs.append(
            ColorJitterSimple(
                brightness=float(tdict.get("brightness", 0)),
                contrast=float(tdict.get("contrast", 0)),
                saturation=float(tdict.get("saturation", 0)),
                hue=float(tdict.get("hue", 0)),
                p=1.0,
                same_on_batch=False,
            )
        )

    # 3) Translate (radial)
    if float(tdict.get("translate", 0)) != 0:
        augs.append(
            RandomRadialTranslate(
                translate_rho=(0, float(tdict["translate"])),
                translate_phi=(0, 360),
                p=1.0,
                same_on_batch=False,
            )
        )
        crop_needed = True

    # 4) Zoom via RandomAffine scale range
    zmin = float(tdict.get("zoomfactor_min", 1.0))
    zmax = float(tdict.get("zoomfactor_max", 1.0))
    if (zmin != 1.0) or (zmax != 1.0):
        # Convert zoom factors to scale range (affine expects scale<1 for zoom-in)
        smin = 1 / math.sqrt(max(zmax, 1e-12))  # zoom-out factor -> smaller scale
        smax = 1 / math.sqrt(max(zmin, 1e-12))
        augs.append(
            kornia.augmentation._2d.geometric.RandomAffine(
                degrees=0.0,
                scale=(smin, smax),
                p=1.0,
                same_on_batch=False,
            )
        )
        crop_needed = True

    # 5) If translate/zoom applied, center-crop back to final size
    if crop_needed:
        augs.append(kornia.augmentation._2d.geometric.CenterCrop(crop_size))

    # 6) Right-angle rotations with uniform {0,90,180,270}
    if transformdict['rotate'] != 0:
        augs.append(
            RandomRightAngleRotation(
                p=1.0,              # always sample a k in {0,1,2,3}
                same_on_batch=False
            )
        )

    # 7) Flips after rotations (each with 0.5 prob)
    if transformdict['flip'] != 0:
        augs.append(
            kornia.augmentation._2d.geometric.RandomHorizontalFlip(
                p=0.5, same_on_batch=False
            )
        )
        augs.append(
            kornia.augmentation._2d.geometric.RandomVerticalFlip(
                p=0.5, same_on_batch=False
            )
        )


    return kornia.augmentation.container.ImageSequential(*augs, same_on_batch=False)
