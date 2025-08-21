# clearit/augmentations/kornia_extra.py
# Modified functions and classes from kornia library to enable translation augmentation in radial direction
# RandomTranslate:    https://github.com/kornia/kornia/blob/9329a5bed2aa05bbc1e3a51707335156638f9060/kornia/augmentation/_2d/geometric/translate.py#L10
# TranslateGenerator: https://github.com/kornia/kornia/blob/9329a5bed2aa05bbc1e3a51707335156638f9060/kornia/augmentation/random_generator/_2d/translate.py#L14
#
# ==============================================================================

from typing import Any, Dict, List, Optional, Tuple, Union

from kornia.augmentation import random_generator as rg
from kornia.augmentation._2d.geometric.base import GeometricAugmentationBase2D
from kornia.constants import Resample, SamplePadding
from kornia.core import Tensor, as_tensor, stack
from kornia.geometry.transform import get_translation_matrix2d, warp_affine

import torch
import torch.nn as nn
from torch.distributions import Uniform
from kornia.augmentation.random_generator.base import RandomGeneratorBase
from kornia.augmentation.utils import _adapted_rsampling, _common_param_check, _range_bound
from kornia.utils.helpers import _extract_device_dtype

from kornia.augmentation._2d.intensity.base import IntensityAugmentationBase2D
from kornia.constants import pi
from kornia.enhance import (
    adjust_brightness_accumulative,
    adjust_contrast_with_mean_subtraction,
    adjust_hue,
    adjust_saturation_with_gray_subtraction,
)

class RandomRadialTranslate(GeometricAugmentationBase2D):
    r"""Apply a random 2D radial translation to a tensor image.

    Args:
        translate_rho: tuple of minimum and maximum translation radius in pixels.
        translate_phi: tuple of minimum and maximum translation angle in degrees.
        resample: resample mode from "nearest" (0) or "bilinear" (1).
        padding_mode: padding mode from "zeros" (0), "border" (1) or "refection" (2).
        same_on_batch: apply the same transformation across the batch.
        align_corners: interpolation flag.
        p: probability of applying the transformation.
        keepdim: whether to keep the output shape the same as input (True) or broadcast it to the batch form (False).

    Shape:
        - Input: :math:`(C, H, W)` or :math:`(B, C, H, W)`
        - Output: :math:`(B, C, H, W)`

    .. note::
        This function internally uses :func:`kornia.geometry.transform.warp_affine`.

    To apply the exact augmenation again, you may take the advantage of the previous parameter state:
        >>> input = torch.randn(1, 3, 32, 32)
        >>> aug = RandomTranslate((0, 10), (0, 360), p=1.)
        >>> (aug(input) == aug(input, params=aug._params)).all()
        tensor(True)
    """

    def __init__(
        self,
        translate_rho: Optional[Union[Tensor, Tuple[float, float]]] = None,
        translate_phi: Optional[Union[Tensor, Tuple[float, float]]] = None,
        resample: Union[str, int, Resample] = Resample.BILINEAR.name,
        same_on_batch: bool = False,
        align_corners: bool = False,
        padding_mode: Union[str, int, SamplePadding] = SamplePadding.ZEROS.name,
        p: float = 0.5,
        keepdim: bool = False,
    ) -> None:
        super().__init__(p=p, same_on_batch=same_on_batch, keepdim=keepdim)
        self._param_generator: TranslateRadialGenerator = TranslateRadialGenerator(translate_rho, translate_phi)
        self.flags = dict(
            resample=Resample.get(resample), padding_mode=SamplePadding.get(padding_mode), align_corners=align_corners
        )

    def compute_transformation(self, input: Tensor, params: Dict[str, Tensor], flags: Dict[str, Any]) -> Tensor:
        translations = stack([params["translate_rho"], params["translate_phi"]], dim=-1)
        return get_translation_matrix2d(as_tensor(translations, device=input.device, dtype=input.dtype))

    def apply_transform(
        self, input: Tensor, params: Dict[str, Tensor], flags: Dict[str, Any], transform: Optional[Tensor] = None
    ) -> Tensor:
        _, _, height, width = input.shape
        if not isinstance(transform, Tensor):
            raise TypeError(f'Expected the `transform` be a Tensor. Got {type(transform)}.')
        # Compute x and y translations from rho and phi
        rho = torch.sqrt(transform[:, 0, -1]) # rho is drawn from Uniform(0,rho**2) in order to ensure uniform distribution
                                              # within the circle. We must take the square root to obtain the actual radius. 
        
        phi = torch.deg2rad(transform[:, 1, -1])
        x = rho * torch.cos(phi)
        y = rho * torch.sin(phi)
        transform[:, 0, -1] = x
        transform[:, 1, -1] = y

        return warp_affine(
            input,
            transform[:, :2, :],
            (height, width),
            flags["resample"].name.lower(),
            align_corners=flags["align_corners"],
            padding_mode=flags["padding_mode"].name.lower(),
        )

    def inverse_transform(
        self,
        input: Tensor,
        flags: Dict[str, Any],
        transform: Optional[Tensor] = None,
        size: Optional[Tuple[int, int]] = None,
    ) -> Tensor:
        if not isinstance(transform, Tensor):
            raise TypeError(f'Expected the `transform` be a Tensor. Got {type(transform)}.')
        return self.apply_transform(
            input,
            params=self._params,
            transform=as_tensor(transform, device=input.device, dtype=input.dtype),
            flags=flags,
        )


class TranslateRadialGenerator(RandomGeneratorBase):
    r"""Get parameters for ``translate`` for a random radial translate transform.
    Args:
        translate: tuple of maximum radial translation in pixels and angle in degrees.
    Returns:
        A dict of parameters to be passed for transformation.
            - translations (Tensor): element-wise translations with a shape of (B, 2).
    Note:
        The generated random numbers are not reproducible across different devices and dtypes. By default,
        the parameters will be generated on CPU in float32. This can be changed by calling
        ``self.set_rng_device_and_dtype(device="cuda", dtype=torch.float64)``.
    """

    def __init__(
        self,
        translate_rho: Optional[Union[Tensor, Tuple[float, float]]] = None,
        translate_phi: Optional[Union[Tensor, Tuple[float, float]]] = None,
    ) -> None:
        super().__init__()
        self.translate_rho = translate_rho
        self.translate_phi = translate_phi

    def __repr__(self) -> str:
        repr = f"translate_rho={self.translate_rho}, translate_phi={self.translate_phi}"
        return repr

    def make_samplers(self, device: torch.device, dtype: torch.dtype) -> None:
        self.translate_rho_sampler = None
        self.translate_phi_sampler = None

        if self.translate_rho is not None:
            _translate_rho = torch.tensor(self.translate_rho).to(device=device, dtype=dtype)
            self.translate_rho_sampler = Uniform(_translate_rho[0]**2, _translate_rho[1]**2, validate_args=False)

        if self.translate_phi is not None:
            _translate_phi = _range_bound(self.translate_phi, 'translate_phi', bounds=(0, 360), check='joint').to(
                device=device, dtype=dtype
            )

            self.translate_phi_sampler = Uniform(_translate_phi[..., 0], _translate_phi[..., 1], validate_args=False)

    def forward(self, batch_shape: torch.Size, same_on_batch: bool = False) -> Dict[str, Tensor]:
        batch_size = batch_shape[0]
        height = batch_shape[-2]
        width = batch_shape[-1]

        _device, _dtype = _extract_device_dtype([self.translate_rho, self.translate_phi])
        _common_param_check(batch_size, same_on_batch)
        if not (isinstance(width, (int,)) and isinstance(height, (int,)) and width > 0 and height > 0):
            raise AssertionError(f"`width` and `height` must be positive integers. Got {width}, {height}.")

        if self.translate_rho_sampler is not None:
            translate_rho = (
                _adapted_rsampling((batch_size,), self.translate_rho_sampler, same_on_batch).to(
                    device=_device, dtype=_dtype
                )
            )
        else:
            translate_rho = torch.zeros((batch_size,), device=_device, dtype=_dtype)

        if self.translate_phi_sampler is not None:
            translate_phi = (
                _adapted_rsampling((batch_size,), self.translate_phi_sampler, same_on_batch).to(
                    device=_device, dtype=_dtype
                )
            )
        else:
            translate_phi = torch.zeros((batch_size,), device=_device, dtype=_dtype)

        return dict(translate_rho=translate_rho, translate_phi=translate_phi)
    
    

class ColorJitterSimple(IntensityAugmentationBase2D):
    r"""Apply a random transformation to the brightness and contrast of a tensor image (omits saturation and hue).

    This implementation aligns PIL. Hence, the output is close to TorchVision. However, it does not
    follow the color theory and is not be actively maintained. Prefer using
    :func:`kornia.augmentation.ColorJiggle`

    .. image:: _static/img/ColorJitter.png

    Args:
        p: probability of applying the transformation.
        brightness: The brightness factor to apply.
        contrast: The contrast factor to apply.
        saturation: The saturation factor to apply.
        hue: The hue factor to apply.
        silence_instantiation_warning: if True, silence the warning at instantiation.
        same_on_batch: apply the same transformation across the batch.
        keepdim: whether to keep the output shape the same as input (True) or broadcast it
                 to the batch form (False).
    Shape:
        - Input: :math:`(C, H, W)` or :math:`(B, C, H, W)`, Optional: :math:`(B, 3, 3)`
        - Output: :math:`(B, C, H, W)`

    .. note::
        This function internally uses :func:`kornia.enhance.adjust_brightness_accumulative`,
        :func:`kornia.enhance.adjust_contrast_with_mean_subtraction`,
        :func:`kornia.enhance.adjust_saturation_with_gray_subtraction`,
        :func:`kornia.enhance.adjust_hue`.

    Examples:
        >>> rng = torch.manual_seed(0)
        >>> inputs = torch.ones(1, 3, 3, 3)
        >>> aug = ColorJitter(0.1, 0.1, 0.1, 0.1, p=1.)
        >>> aug(inputs)
        tensor([[[[0.9993, 0.9993, 0.9993],
                  [0.9993, 0.9993, 0.9993],
                  [0.9993, 0.9993, 0.9993]],
        <BLANKLINE>
                 [[0.9993, 0.9993, 0.9993],
                  [0.9993, 0.9993, 0.9993],
                  [0.9993, 0.9993, 0.9993]],
        <BLANKLINE>
                 [[0.9993, 0.9993, 0.9993],
                  [0.9993, 0.9993, 0.9993],
                  [0.9993, 0.9993, 0.9993]]]])

    To apply the exact augmenation again, you may take the advantage of the previous parameter state:

        >>> input = torch.randn(1, 3, 32, 32)
        >>> aug = ColorJitter(0.1, 0.1, 0.1, 0.1, p=1.)
        >>> (aug(input) == aug(input, params=aug._params)).all()
        tensor(True)
    """

    def __init__(
        self,
        brightness: Union[Tensor, float, Tuple[float, float], List[float]] = 0.0,
        contrast: Union[Tensor, float, Tuple[float, float], List[float]] = 0.0,
        saturation: Union[Tensor, float, Tuple[float, float], List[float]] = 0.0,
        hue: Union[Tensor, float, Tuple[float, float], List[float]] = 0.0,
        same_on_batch: bool = False,
        p: float = 1.0,
        keepdim: bool = False,
    ) -> None:
        super().__init__(p=p, same_on_batch=same_on_batch, keepdim=keepdim)

        self.brightness = brightness
        self.contrast = contrast
        self.saturation = saturation
        self.hue = hue
        self._param_generator = rg.ColorJitterGenerator(brightness, contrast, saturation, hue)

    def apply_transform(
        self, input: Tensor, params: Dict[str, Tensor], flags: Dict[str, Any], transform: Optional[Tensor] = None
    ) -> Tensor:
        transforms = [
            lambda img: adjust_brightness_accumulative(img, params["brightness_factor"]),
            lambda img: adjust_contrast_with_mean_subtraction(img, params["contrast_factor"]),
            lambda img: img,
            lambda img: img,
            #lambda img: adjust_saturation_with_gray_subtraction(img, params["saturation_factor"]),
            #lambda img: adjust_hue(img, params["hue_factor"] * 2 * pi),
        ]

        jittered = input
        for idx in params["order"].tolist():
            t = transforms[idx]
            jittered = t(jittered)

        return jittered

class RandomRightAngleRotation(nn.Module):
    """
    Rotate each image by k * 90°, where k is sampled uniformly from {0,1,2,3}.

    Args:
        p (float): probability to apply a (possibly 0°) rotation per sample.
                   If not applied, k=0 for that sample.
        same_on_batch (bool): if True, use the same k for the whole batch.

    Notes:
        - Input shape must be (B, C, H, W).
        - Rotation is done with torch.rot90 over the last two dims.
        - Setting p=1.0 gives a uniform distribution over 0/90/180/270 for each item.
    """
    def __init__(self, p: float = 1.0, same_on_batch: bool = False):
        super().__init__()
        self.p = float(p)
        self.same_on_batch = bool(same_on_batch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training or self.p <= 0.0:
            return x

        B = x.size(0)
        device = x.device

        if self.same_on_batch:
            # coin flip once for the whole batch
            apply = torch.rand((), device=device) < self.p
            k = torch.randint(0, 4, (1,), device=device).item() if apply else 0
            return x if k == 0 else torch.rot90(x, k, dims=(-2, -1))

        # per-sample coin flips
        apply_mask = (torch.rand((B,), device=device) < self.p)
        # sample k for everyone, then zero out k where we don't apply
        ks = torch.randint(0, 4, (B,), device=device)
        ks = ks * apply_mask.to(torch.long)

        # rotate per group of k
        out = x
        for k in (1, 2, 3):
            sel = (ks == k)
            if sel.any():
                out = out.clone()  # avoid in-place on a view
                out[sel] = torch.rot90(x[sel], k, dims=(-2, -1))
        return out
