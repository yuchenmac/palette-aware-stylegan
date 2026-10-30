"""Data standardisation and patch construction from Sections 2.2-2.3.1."""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Dict, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np
from PIL import Image, ImageCms, ImageOps

from .image_arrays import normalise_rgb_array

if TYPE_CHECKING:
    from torch import Tensor


PathLike = Union[str, Path]


@dataclass(frozen=True)
class PatchRecord:
    """A retained patch and its auditable source-level coordinates."""

    rgb: np.ndarray
    lab: np.ndarray
    source_id: str
    top: int
    left: int
    greyscale_std: float
    mean_saturation: float


def load_image(path: PathLike) -> Image.Image:
    """Read an already standardised image as an oriented 8-bit RGB working copy.

    This Pillow path does not preserve a 16-bit acquisition/calibration pipeline.
    Convert high-bit-depth originals to standardised sRGB before using this path.
    """

    image_path = Path(path).expanduser()
    if image_path.suffix.lower() not in {".png", ".jpg", ".jpeg", ".tif", ".tiff"}:
        raise ValueError("Expected a PNG, JPEG, or TIFF image.")
    with Image.open(image_path) as opened:
        if opened.mode.startswith("I") or opened.mode == "F":
            raise ValueError(
                "High-bit-depth/integer or floating-point Pillow sources require "
                "explicit colour standardisation before conversion to an 8-bit sRGB working copy."
            )
        return ImageOps.exif_transpose(opened).convert("RGB").copy()


def apply_icc_profile(
    image: Image.Image,
    profile_path: Optional[PathLike] = None,
) -> Image.Image:
    """Apply a dataset-specific input ICC profile and convert to sRGB.

    This hook operates on Pillow's 8-bit RGB working representation.
    """

    if profile_path is None:
        warnings.warn(
            "No input ICC profile was supplied; ICC conversion was skipped.",
            RuntimeWarning,
            stacklevel=2,
        )
        return image.convert("RGB")
    path = Path(profile_path).expanduser()
    if not path.is_file():
        raise FileNotFoundError("The requested calibration profile does not exist.")
    input_profile = ImageCms.getOpenProfile(str(path))
    output_profile = ImageCms.createProfile("sRGB")
    return ImageCms.profileToProfile(
        image.convert("RGB"), input_profile, output_profile, outputMode="RGB"
    )


def apply_white_balance(
    image: Image.Image,
    matrix: Optional[np.ndarray] = None,
) -> Image.Image:
    """Apply a caller-supplied 3 x 3 white-balance matrix in linear RGB.

    The caller supplies the acquisition matrix. The result is an 8-bit RGB image.
    """

    if matrix is None:
        warnings.warn(
            "No white-balance matrix was supplied; white balancing was skipped.",
            RuntimeWarning,
            stacklevel=2,
        )
        return image.convert("RGB")
    transform = np.asarray(matrix, dtype=np.float32)
    if transform.shape != (3, 3):
        raise ValueError("A white-balance matrix must have shape (3, 3).")
    srgb = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
    linear = srgb_to_linear_numpy(srgb)
    balanced = np.einsum("...c,dc->...d", linear, transform)
    encoded = linear_to_srgb_numpy(np.clip(balanced, 0.0, 1.0))
    return Image.fromarray(np.round(encoded * 255.0).astype(np.uint8), mode="RGB")


def normalise_geometry(
    image: Image.Image,
    content_box: Optional[Tuple[int, int, int, int]] = None,
    output_size: Optional[Tuple[int, int]] = None,
) -> Image.Image:
    """Crop a caller-supplied central-panel box and optionally standardise size."""

    result = image.crop(content_box) if content_box is not None else image.copy()
    if output_size is not None:
        result = result.resize(output_size, Image.Resampling.BICUBIC)
    return result


def srgb_to_linear_numpy(rgb: np.ndarray) -> np.ndarray:
    """Decode sRGB values in [0, 1] to linear RGB."""

    rgb = np.asarray(rgb, dtype=np.float32)
    return np.where(rgb <= 0.04045, rgb / 12.92, ((rgb + 0.055) / 1.055) ** 2.4)


def linear_to_srgb_numpy(rgb: np.ndarray) -> np.ndarray:
    """Encode linear RGB values in [0, 1] as sRGB."""

    rgb = np.asarray(rgb, dtype=np.float32)
    return np.where(rgb <= 0.0031308, 12.92 * rgb, 1.055 * rgb ** (1.0 / 2.4) - 0.055)


def srgb_to_lab(rgb: np.ndarray) -> np.ndarray:
    """Convert an sRGB array in [0, 1] to CIELAB under D65."""

    from skimage.color import rgb2lab

    array = np.asarray(rgb, dtype=np.float32)
    if array.ndim != 3 or array.shape[-1] != 3:
        raise ValueError("Expected an H x W x 3 sRGB array.")
    return rgb2lab(np.clip(array, 0.0, 1.0), illuminant="D65").astype(np.float32)


def srgb_to_lab_torch(rgb: Tensor) -> Tensor:
    """Differentiable sRGB-to-CIELAB(D65) conversion for the U-port.

    Implements the colour-space conversion required before Eq. (21) in the
    manuscript. Input must have shape B x 3 x H x W and values in [0, 1].
    """

    import torch

    if rgb.ndim != 4 or rgb.shape[1] != 3:
        raise ValueError("Expected a B x 3 x H x W tensor.")
    rgb = rgb.clamp(0.0, 1.0)
    linear = torch.where(
        rgb <= 0.04045,
        rgb / 12.92,
        ((rgb + 0.055) / 1.055).pow(2.4),
    )
    matrix = rgb.new_tensor(
        [
            [0.4124564, 0.3575761, 0.1804375],
            [0.2126729, 0.7151522, 0.0721750],
            [0.0193339, 0.1191920, 0.9503041],
        ]
    )
    xyz = torch.einsum("dc,bchw->bdhw", matrix, linear)
    white = rgb.new_tensor([0.95047, 1.0, 1.08883]).view(1, 3, 1, 1)
    scaled = xyz / white
    delta = 6.0 / 29.0
    f = torch.where(
        scaled > delta**3,
        scaled.clamp_min(1.0e-12).pow(1.0 / 3.0),
        scaled / (3.0 * delta**2) + 4.0 / 29.0,
    )
    l_channel = 116.0 * f[:, 1:2] - 16.0
    a_channel = 500.0 * (f[:, 0:1] - f[:, 1:2])
    b_channel = 200.0 * (f[:, 1:2] - f[:, 2:3])
    return torch.cat([l_channel, a_channel, b_channel], dim=1)


def perceptual_greyscale(rgb: np.ndarray) -> np.ndarray:
    """Compute linear-light BT.709 greyscale from Eq. (A.2)."""

    linear = srgb_to_linear_numpy(np.asarray(rgb, dtype=np.float32))
    return (
        0.2126 * linear[..., 0]
        + 0.7152 * linear[..., 1]
        + 0.0722 * linear[..., 2]
    )


def patch_quality(
    rgb: np.ndarray,
    min_greyscale_std: Optional[float] = 0.02,
    min_mean_saturation: Optional[float] = 0.05,
) -> Tuple[bool, float, float]:
    """Evaluate the grey-level and saturation patch filters in Section 2.3.1."""

    from skimage.color import rgb2hsv

    grey_std = float(perceptual_greyscale(rgb).std())
    mean_saturation = float(rgb2hsv(np.clip(rgb, 0.0, 1.0))[..., 1].mean())
    passes_texture = min_greyscale_std is None or grey_std >= min_greyscale_std
    passes_saturation = (
        min_mean_saturation is None or mean_saturation >= min_mean_saturation
    )
    return passes_texture and passes_saturation, grey_std, mean_saturation


def extract_patches(
    image: Union[Image.Image, np.ndarray],
    source_id: str,
    patch_size: int = 512,
    stride: int = 256,
    min_greyscale_std: Optional[float] = 0.02,
    min_mean_saturation: Optional[float] = 0.05,
) -> List[PatchRecord]:
    """Extract patches from RGB8 Pillow images or explicitly ranged RGB arrays.

    Arrays may be uint8, uint16, or finite floating-point values in [0, 1].
    The returned RGB and CIELAB arrays use float32.
    """

    if isinstance(image, Image.Image):
        if image.mode.startswith("I") or image.mode == "F":
            raise ValueError("Convert high-bit-depth Pillow images to standardised RGB before patching.")
        rgb = normalise_rgb_array(np.asarray(image.convert("RGB")))
    else:
        rgb = normalise_rgb_array(image)
    if patch_size < 1 or stride < 1:
        raise ValueError("Patch size and stride must be positive.")
    height, width = rgb.shape[:2]
    if height < patch_size or width < patch_size:
        raise ValueError("The standardised source image is smaller than one patch.")
    records: List[PatchRecord] = []
    tops = list(range(0, height - patch_size + 1, stride))
    lefts = list(range(0, width - patch_size + 1, stride))
    if tops[-1] != height - patch_size:
        tops.append(height - patch_size)
    if lefts[-1] != width - patch_size:
        lefts.append(width - patch_size)
    for top in tops:
        for left in lefts:
            patch = rgb[top : top + patch_size, left : left + patch_size]
            keep, grey_std, saturation = patch_quality(
                patch, min_greyscale_std, min_mean_saturation
            )
            if keep:
                records.append(
                    PatchRecord(
                        rgb=patch.copy(),
                        lab=srgb_to_lab(patch),
                        source_id=source_id,
                        top=top,
                        left=left,
                        greyscale_std=grey_std,
                        mean_saturation=saturation,
                    )
                )
    return records


def source_image_split(
    sources: Sequence[PathLike],
    ratios: Sequence[float] = (0.70, 0.15, 0.15),
    seed: int = 42,
) -> Dict[str, List[Path]]:
    """Split source images before patching to prevent cross-subset leakage.

    Implements the source-image-level 70/15/15 interface in Section 3.1.
    """

    if (
        len(ratios) != 3
        or not all(np.isfinite(value) and value >= 0.0 for value in ratios)
        or not np.isclose(sum(ratios), 1.0)
    ):
        raise ValueError("Source split ratios must contain three nonnegative values summing to one.")
    paths = [Path(item).expanduser() for item in sources]
    if len({path.resolve() for path in paths}) != len(paths):
        raise ValueError("Source images must not contain duplicate file paths.")
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(paths))
    shuffled = [paths[index] for index in order]
    n_train = int(round(len(shuffled) * ratios[0]))
    n_val = min(int(round(len(shuffled) * ratios[1])), len(shuffled) - n_train)
    return {
        "train": shuffled[:n_train],
        "val": shuffled[n_train : n_train + n_val],
        "test": shuffled[n_train + n_val :],
    }
