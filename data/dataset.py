"""Generic image dataset for authorised, privately held patches."""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Dict, List, Union

import torch
from PIL import Image, ImageOps
from torch import Tensor
from torch.utils.data import Dataset
from torchvision.transforms import functional as TF


SUPPORTED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff"}


class WoodblockPatchDataset(Dataset[Dict[str, Union[Tensor, str]]]):
    """Read PNG, JPEG, or TIFF patches without embedding private paths.

    Implements the generic data interface required by Sections 2.2, 2.3.1, and
    3.1 of the manuscript. Images are returned in the StyleGAN range [-1, 1].

    A supplied image that is not square at the configured patch resolution is
    centre-cropped and bicubically resized. Dataset owners should normally run the
    explicit preprocessing module.
    """

    def __init__(
        self,
        data_root: Union[str, Path],
        split: str = "train",
        resolution: int = 512,
        recursive: bool = True,
    ) -> None:
        self.data_root = Path(data_root).expanduser()
        split_root = self.data_root / split
        self.root = split_root
        self.split = split
        self.resolution = int(resolution)
        if not split_root.is_dir():
            warnings.warn(
                f"Expected split directory does not exist: {split_root}. "
                "The dataset is empty; root fallback is disabled to prevent leakage.",
                RuntimeWarning,
                stacklevel=2,
            )
        globber = self.root.rglob if recursive else self.root.glob
        self.files: List[Path] = sorted(
            path
            for path in globber("*")
            if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
        ) if self.root.exists() else []

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, index: int) -> Dict[str, Union[Tensor, str]]:
        path = self.files[index]
        with Image.open(path) as opened:
            image = ImageOps.exif_transpose(opened).convert("RGB")
            if image.size != (self.resolution, self.resolution):
                side = min(image.size)
                image = TF.center_crop(image, [side, side])
                image = TF.resize(
                    image,
                    [self.resolution, self.resolution],
                    interpolation=TF.InterpolationMode.BICUBIC,
                    antialias=True,
                )
            tensor = TF.to_tensor(image).mul(2.0).sub(1.0)
        return {
            "image": tensor,
            "path": str(path),
            "source_id": path.stem,
        }

    @property
    def is_empty(self) -> bool:
        """Return True when the requested private split has no supported files."""

        return len(self.files) == 0
