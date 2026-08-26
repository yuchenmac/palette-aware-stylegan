"""Evaluate generated and private reference images with the manuscript metrics."""

from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np
import torch
import yaml
from PIL import Image, ImageOps
from torch import Tensor
from torchvision.transforms import functional as TF

from evaluation.palette_metrics import palette_kl_from_counts, palette_soft_counts
from evaluation.structure_metrics import closure_rate, edge_map_from_image, line_continuity_index
from priors.structural_prior import StructuralPrior


EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff"}


def image_paths(root: Path) -> List[Path]:
    return sorted(path for path in root.rglob("*") if path.is_file() and path.suffix.lower() in EXTENSIONS)


def select_evaluation_images(paths: Sequence[Path], required: int) -> List[Path]:
    """Select the protocol count using a platform-independent deterministic order."""

    if required < 1:
        raise ValueError("evaluation_images_per_method must be positive.")
    ordered = sorted(
        paths,
        key=lambda path: (path.as_posix().casefold(), path.as_posix()),
    )
    if len(ordered) < required:
        warnings.warn(
            f"Only {len(ordered)} generated images were found; the manuscript "
            f"evaluation protocol uses {required} images per method.",
            RuntimeWarning,
            stacklevel=2,
        )
    return ordered[:required]


def load_batch(paths: Sequence[Path], resolution: int) -> Tensor:
    images = []
    for path in paths:
        with Image.open(path) as opened:
            image = ImageOps.exif_transpose(opened).convert("RGB")
            if image.size != (resolution, resolution):
                image = TF.resize(
                    TF.center_crop(image, min(image.size)),
                    [resolution, resolution],
                    interpolation=TF.InterpolationMode.BICUBIC,
                    antialias=True,
                )
            images.append(TF.to_tensor(image))
    return torch.stack(images)


def batched(items: Sequence[Path], batch_size: int) -> Iterable[Sequence[Path]]:
    for start in range(0, len(items), batch_size):
        yield items[start : start + batch_size]


def build_structural(config: Mapping[str, object]) -> StructuralPrior:
    structural = config["structural_prior"]
    assert isinstance(structural, Mapping)
    return StructuralPrior(
        tau_edge=float(structural["tau_edge"]),
        gamma=float(structural["gamma"]),
        line_threshold=float(structural["line_threshold"]),
        dilation_radius=int(structural["dilation_radius"]),
        buffer_radius=int(structural["buffer_radius"]),
        max_distance=int(structural["max_distance"]),
        gaussian_sigma=float(structural["gaussian_sigma"]),
        sobel_normalisation=float(structural["sobel_normalisation"]),
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate Palette-aware StyleGAN images.")
    parser.add_argument("--generated", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Path("config.yaml"))
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    if not args.checkpoint.is_file():
        print("Trained checkpoints from the study are not distributed with this code.")
        return 2
    if not args.generated.is_dir() or not args.reference.is_dir():
        print("Generated and authorised reference image directories are required.")
        return 2
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    if "config" in checkpoint:
        config = checkpoint["config"]
    else:
        if not args.config.is_file():
            print("Configuration file not found.")
            return 2
        with args.config.open("r", encoding="utf-8") as handle:
            config = yaml.safe_load(handle)
    if not isinstance(config, Mapping):
        raise ValueError("Checkpoint configuration must be a mapping.")

    generated_paths = image_paths(args.generated)
    reference_paths = image_paths(args.reference)
    if not generated_paths or not reference_paths:
        print("No supported evaluation images were found.")
        return 2
    resolution = int(config["image"]["resolution"])
    evaluation = config["evaluation"]
    palette_loss = config["palette_loss"]
    assert isinstance(evaluation, Mapping) and isinstance(palette_loss, Mapping)
    generated_paths = select_evaluation_images(
        generated_paths, int(evaluation["evaluation_images_per_method"])
    )
    device = torch.device(args.device)
    centroids = torch.as_tensor(checkpoint["palette_centroids"], dtype=torch.float32, device=device)
    target = torch.as_tensor(checkpoint["palette_proportions"], dtype=torch.float32, device=device)
    target = target / target.sum()
    structural = build_structural(config)
    from torchmetrics.image.fid import FrechetInceptionDistance
    from torchmetrics.image.kid import KernelInceptionDistance

    subset_size = max(
        2,
        min(int(evaluation["kid_subset_size"]), len(generated_paths), len(reference_paths)),
    )
    fid = FrechetInceptionDistance(normalize=True).to(device)
    kid = KernelInceptionDistance(subset_size=subset_size, normalize=True).to(device)

    for paths in batched(reference_paths, args.batch_size):
        real = load_batch(paths, resolution).to(device)
        fid.update(real, real=True)
        kid.update(real, real=True)

    soft_counts = torch.zeros_like(target)
    soft_total = torch.zeros((), device=device)
    flat_numerator = 0.0
    flat_denominator = 0
    closure_values: List[float] = []
    generated_continuity: List[float] = []
    for paths in batched(generated_paths, args.batch_size):
        generated_01 = load_batch(paths, resolution).to(device)
        fid.update(generated_01, real=False)
        kid.update(generated_01, real=False)
        generated = generated_01.mul(2.0).sub(1.0)
        counts, count = palette_soft_counts(
            generated, centroids, float(palette_loss["temperature_end"]), structural
        )
        soft_counts += counts
        soft_total += count
        prior_output = structural(generated)
        flat_mask = prior_output.flat_fill_mask
        violations = flat_mask & (
            prior_output.gradient_magnitude > float(evaluation["flatness_gradient_threshold"])
        )
        flat_numerator += float(violations.sum())
        flat_denominator += int(flat_mask.sum())
        for image in generated_01:
            edge = edge_map_from_image(image, structural)
            closure_values.append(closure_rate(edge))
            generated_continuity.append(line_continuity_index(edge))

    reference_continuity: List[float] = []
    for paths in batched(reference_paths, args.batch_size):
        for image in load_batch(paths, resolution):
            reference_continuity.append(line_continuity_index(edge_map_from_image(image, structural)))

    palette_kl_value = float(
        palette_kl_from_counts(soft_counts, soft_total, target)
    )
    kid_mean, kid_std = kid.compute()
    results = {
        "fid": float(fid.compute().cpu()),
        "kid_mean": float(kid_mean.cpu()),
        "kid_std": float(kid_std.cpu()),
        "palette_kl": palette_kl_value,
        "flatness": flat_numerator / max(flat_denominator, 1),
        "closure_rate": float(np.mean(closure_values)),
        "line_continuity_normalised": float(np.mean(generated_continuity))
        / max(float(np.mean(reference_continuity)), 1.0e-8),
        "generated_images": len(generated_paths),
        "reference_images": len(reference_paths),
    }
    print(json.dumps(results, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
