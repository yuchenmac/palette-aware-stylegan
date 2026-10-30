"""Generic image dataset for authorised, privately held patches."""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Dict, List, Optional, Union

import torch
from PIL import Image, ImageOps
from torch import Tensor
from torch.utils.data import Dataset
from torchvision.transforms import functional as TF

from data.manifest import load_manifest


SUPPORTED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff"}


class WoodblockPatchDataset(Dataset[Dict[str, Union[Tensor, str]]]):
    """Read prepared RGB patches and optional source identities from manifest.json.

    Implements the generic data interface required by Sections 2.2, 2.3.1, and
    3.1 of the manuscript. Images are returned in the StyleGAN range [-1, 1].

    A supplied image that is not square at the configured patch resolution is
    centre-cropped and bicubically resized. Dataset owners should normally run the
    explicit preprocessing module.

    With a manifest, every loaded patch must be recorded and source identities
    are checked across all splits. Without a manifest, source_id is an empty
    string; patch_id always identifies the patch file.
    """

    def __init__(
        self,
        data_root: Union[str, Path],
        split: str = "train",
        resolution: int = 512,
        recursive: bool = True,
        manifest_path: Optional[Union[str, Path]] = None,
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
        manifest_file = (
            Path(manifest_path).expanduser()
            if manifest_path is not None else self.data_root / "manifest.json"
        )
        if manifest_path is not None and not manifest_file.is_file():
            raise FileNotFoundError(f"Dataset manifest does not exist: {manifest_file}")
        self.manifest = load_manifest(manifest_file) if manifest_file.is_file() else None
        if self.manifest is not None:
            for path in self.files:
                relative = path.relative_to(self.data_root).as_posix()
                if relative not in self.manifest:
                    raise ValueError(f"Patch is missing from the dataset manifest: {relative}")
            for relative, record in self.manifest.items():
                if record["split"] == self.split and not (self.data_root / relative).is_file():
                    raise FileNotFoundError(f"Manifest patch does not exist: {relative}")

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, index: int) -> Dict[str, Union[Tensor, str]]:
        path = self.files[index]
        with Image.open(path) as opened:
            if opened.mode.startswith("I") or opened.mode == "F":
                raise ValueError("Dataset patches must be standardised 8-bit RGB working images.")
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
            "patch_id": path.stem,
            "source_id": (
                str(self.manifest[path.relative_to(self.data_root).as_posix()]["source_id"])
                if self.manifest is not None else ""
            ),
        }

    @property
    def is_empty(self) -> bool:
        """Return True when the requested private split has no supported files."""

        return len(self.files) == 0
