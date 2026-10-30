"""Evaluate generated and private reference images with the manuscript metrics."""

from __future__ import annotations

import argparse
import random
from importlib.metadata import version
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
from evaluation.protocol import (
    SCHEMA_VERSION, check_reference_count, experiment_config, file_digest,
    image_set_digest, select_evaluation_images, training_seed, value_digest, write_result,
)
from priors.structural_prior import StructuralPrior


EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff"}


def image_paths(root: Path) -> List[Path]:
    return sorted(path for path in root.rglob("*") if path.is_file() and path.suffix.lower() in EXTENSIONS)


def seed_evaluation(seed: int) -> None:
    """Fix Python, NumPy and Torch random states, including KID subset sampling."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


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
    parser.add_argument("--seed", type=int, default=42, help="Random seed for evaluation and KID subset sampling.")
    parser.add_argument("--output", type=Path, help="Save evaluation metrics and protocol metadata as JSON.")
    parser.add_argument("--allow-partial", action="store_true", help="Explicit exploratory mode; incomplete counts are allowed and the output is marked non-protocol.")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    if not args.checkpoint.is_file():
        print(f"Checkpoint file not found: {args.checkpoint}")
        return 2
    if not args.generated.is_dir() or not args.reference.is_dir():
        print("Generated and reference image directories must both exist.")
        return 2
    if args.batch_size < 1:
        parser.error("--batch-size must be positive.")
    if not 0 <= args.seed < 2**32:
        parser.error("--seed must be an integer between 0 and 2**32 - 1.")
    if args.output is not None and args.output.resolve() in {args.checkpoint.resolve(), args.config.resolve()}:
        parser.error("--output must not overwrite the checkpoint or configuration.")
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    if "config" in checkpoint:
        config = checkpoint["config"]
        config_source = "checkpoint"
    else:
        if not args.config.is_file():
            print("Configuration file not found.")
            return 2
        with args.config.open("r", encoding="utf-8") as handle:
            config = yaml.safe_load(handle)
        config_source = str(args.config.resolve())
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
    required_generated = int(evaluation["evaluation_images_per_method"])
    required_reference = evaluation.get("reference_images")
    required_reference = int(required_reference) if required_reference is not None else None
    try:
        generated_paths = select_evaluation_images(generated_paths, required_generated, allow_partial=args.allow_partial)
        check_reference_count(len(reference_paths), required_reference, allow_partial=args.allow_partial)
        run_training_seed = training_seed(config)
    except ValueError as exc:
        parser.error(str(exc))
    if min(len(generated_paths), len(reference_paths)) < 2:
        parser.error("FID/KID require at least two generated images and two reference images.")
    requested_subset_size = int(evaluation["kid_subset_size"])
    if requested_subset_size < 2:
        parser.error("evaluation.kid_subset_size must be at least two.")
    if min(len(generated_paths), len(reference_paths)) < requested_subset_size and not args.allow_partial:
        parser.error("The image counts are smaller than kid_subset_size; use the full protocol or explicitly use --allow-partial.")
    seed_evaluation(args.seed)
    device = torch.device(args.device)
    centroids = torch.as_tensor(checkpoint["palette_centroids"], dtype=torch.float32, device=device)
    target = torch.as_tensor(checkpoint["palette_proportions"], dtype=torch.float32, device=device)
    target = target / target.sum()
    structural = build_structural(config)
    from torchmetrics.image.fid import FrechetInceptionDistance
    from torchmetrics.image.kid import KernelInceptionDistance

    subset_size = max(
        2,
        min(requested_subset_size, len(generated_paths), len(reference_paths)),
    )
    fid = FrechetInceptionDistance(feature=2048, normalize=True).to(device)
    kid = KernelInceptionDistance(feature=2048, subsets=100, subset_size=subset_size, degree=3, normalize=True).to(device)

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
    # Reset immediately before KID sampling so model construction and batch size
    # cannot consume its random stream.
    seed_evaluation(args.seed)
    kid_mean, kid_subset_std = kid.compute()
    palette_identity = value_digest({
        "centroids": centroids.detach().cpu().tolist(),
        "proportions": target.detach().cpu().tolist(),
    })
    protocol = {
        "name": "palette-aware-objective-evaluation-v1",
        "complete": not args.allow_partial,
        "required_generated_images": required_generated,
        "generated_images": len(generated_paths),
        "required_reference_images": required_reference,
        "reference_images": len(reference_paths),
        "reference_images_sha256": image_set_digest(reference_paths),
        "resolution": resolution,
        "evaluation_seed": args.seed,
        "checkpoint_step": checkpoint.get("step"),
        "palette_k": int(centroids.shape[0]),
        "palette_temperature": float(palette_loss["temperature_end"]),
        "palette_codebook_sha256": palette_identity,
        "palette_kl": "KL(target || pooled generated soft counts)",
        "flatness": "total violating pixels / total valid flat-fill pixels",
        "flatness_gradient_threshold": float(evaluation["flatness_gradient_threshold"]),
        "structural_prior": dict(config["structural_prior"]),
        "structural_aggregation": "image means; continuity divided by reference-image mean",
        "fid_feature": 2048,
        "kid_feature": 2048,
        "kid_degree": 3,
        "kid_subsets": 100,
        "kid_subset_size": subset_size,
        "torchmetrics_version": version("torchmetrics"),
        "torch_fidelity_version": version("torch-fidelity"),
        "torch_version": torch.__version__,
        "image_processing": "EXIF transpose; RGB; centre-crop and bicubic resize only when resolution differs",
    }
    results = {
        "schema_version": SCHEMA_VERSION,
        "fid": float(fid.compute().cpu()),
        "kid_mean": float(kid_mean.cpu()),
        "kid_subset_std": float(kid_subset_std.cpu()),
        "palette_kl": palette_kl_value,
        "flatness": flat_numerator / flat_denominator if flat_denominator > 0 else float("nan"),
        "palette_valid_pixels": float(soft_total),
        "flatness_valid_pixels": flat_denominator,
        "closure_rate": float(np.mean(closure_values)),
        "line_continuity_normalised": float(np.mean(generated_continuity))
        / max(float(np.mean(reference_continuity)), 1.0e-8),
        "generated_images": len(generated_paths),
        "reference_images": len(reference_paths),
        "metadata": {
            "training_seed": run_training_seed,
            "evaluation_seed": args.seed,
            "checkpoint": {
                "path": str(args.checkpoint.resolve()),
                "sha256": file_digest(args.checkpoint),
                "step": checkpoint.get("step"),
            },
            "config": {
                "source": config_source,
                "values": dict(config),
                "sha256": value_digest(config),
                "experiment_sha256": value_digest(experiment_config(config)),
            },
            "generated_directory": str(args.generated.resolve()),
            "reference_directory": str(args.reference.resolve()),
            "protocol": protocol,
            "kid_subset_std_description": "Standard deviation over KID subsets within this evaluation, not across training seeds.",
        },
    }
    print(write_result(results, args.output))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
