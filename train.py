"""Command-line training entry point. Importing this module never starts training."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Iterator, Mapping, MutableMapping

import torch
import yaml
from torch import Tensor
from torch.utils.data import DataLoader

from data.dataset import WoodblockPatchDataset
from losses.palette_aware import PaletteAwareRegularisation
from models.discriminator import DualDiscriminator
from models.generator import Generator
from priors.palette_prior import PalettePrior
from priors.structural_prior import StructuralPrior
from training.trainer import Trainer, set_random_seed


def load_config(path: Path) -> MutableMapping[str, object]:
    with path.open("r", encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle)
    if not isinstance(loaded, dict):
        raise ValueError("Configuration root must be a mapping.")
    return loaded


def build_structural_prior(config: Mapping[str, object]) -> StructuralPrior:
    item = config["structural_prior"]
    assert isinstance(item, Mapping)
    return StructuralPrior(
        tau_edge=float(item["tau_edge"]),
        gamma=float(item["gamma"]),
        line_threshold=float(item["line_threshold"]),
        dilation_radius=int(item["dilation_radius"]),
        buffer_radius=int(item["buffer_radius"]),
        max_distance=int(item["max_distance"]),
        gaussian_sigma=float(item["gaussian_sigma"]),
        sobel_normalisation=float(item["sobel_normalisation"]),
    )


def dataset_images(dataset: WoodblockPatchDataset) -> Iterator[Tensor]:
    for index in range(len(dataset)):
        sample = dataset[index]
        image = sample["image"]
        if isinstance(image, Tensor):
            yield image


def main() -> int:
    parser = argparse.ArgumentParser(description="Train Palette-aware StyleGAN.")
    parser.add_argument("--config", type=Path, default=Path("config.yaml"))
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--checkpoint-dir", type=Path, default=Path("checkpoints"))
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    if not args.config.is_file():
        print("Configuration file not found.")
        return 2
    if not args.data_root.is_dir():
        print("Private dataset directory not found. No training was started.")
        return 2
    config = load_config(args.config)
    image_config = config["image"]
    training_config = config["training"]
    generator_config = config["generator"]
    discriminator_config = config["discriminator"]
    palette_config = config["palette"]
    palette_loss_config = config["palette_loss"]
    ablation = config["ablation"]
    structural_config = config["structural_prior"]
    assert all(
        isinstance(item, Mapping)
        for item in (
            image_config, training_config, generator_config, discriminator_config,
            palette_config, palette_loss_config, ablation, structural_config,
        )
    )

    set_random_seed(int(training_config["seed"]))
    resolution = int(image_config["resolution"])
    train_dataset = WoodblockPatchDataset(args.data_root, "train", resolution)
    val_dataset = WoodblockPatchDataset(args.data_root, "val", resolution)
    if train_dataset.is_empty:
        print("No supported images were found in the private training split. No training was started.")
        return 2
    if len(train_dataset) < int(training_config["batch_size"]):
        print("The private training split is smaller than the configured batch size. No training was started.")
        return 2

    train_loader = DataLoader(
        train_dataset,
        batch_size=int(training_config["batch_size"]),
        shuffle=True,
        num_workers=int(training_config["num_workers"]),
        drop_last=True,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = None if val_dataset.is_empty else DataLoader(
        val_dataset,
        batch_size=int(training_config["batch_size"]),
        shuffle=False,
        num_workers=int(training_config["num_workers"]),
        drop_last=False,
        pin_memory=torch.cuda.is_available(),
    )

    device = torch.device(args.device)
    structural_prior = build_structural_prior(config)
    if args.resume is not None:
        if not args.resume.is_file():
            print("Resume checkpoint not found. No training was started.")
            return 2
        resume_data = torch.load(args.resume, map_location="cpu")
        centroids = torch.as_tensor(resume_data["palette_centroids"], dtype=torch.float32)
        proportions = torch.as_tensor(resume_data["palette_proportions"], dtype=torch.float32)
    else:
        print("Learning the finite palette codebook from the private training split.")
        palette_prior = PalettePrior(
            k=int(palette_config["k"]),
            max_samples_per_patch=int(palette_config["max_samples_per_patch"]),
            random_seed=int(palette_config["random_seed"]),
            max_iter=int(palette_config["kmeans_max_iter"]),
            tolerance=float(palette_config["kmeans_tolerance"]),
            epsilon=float(palette_config["numerical_epsilon"]),
        )
        codebook = palette_prior.fit_rgb_patches(dataset_images(train_dataset), structural_prior)
        centroids, proportions = codebook.tensors("cpu")

    generator = Generator(
        latent_dim=int(generator_config["latent_dim"]),
        style_dim=int(generator_config["style_dim"]),
        mapping_layers=int(generator_config["mapping_layers"]),
        resolution=int(generator_config["resolution"]),
        channel_base=int(generator_config["channel_base"]),
        channel_max=int(generator_config["channel_max"]),
        leaky_relu_slope=float(generator_config["leaky_relu_slope"]),
        demodulation_epsilon=float(generator_config["demodulation_epsilon"]),
    )
    discriminator = DualDiscriminator(
        resolution=resolution,
        channel_base=int(discriminator_config["channel_base"]),
        channel_max=int(discriminator_config["channel_max"]),
        head_channels=int(discriminator_config["head_channels"]),
        head_features=int(discriminator_config["head_features"]),
        structural_fusion_weight=float(discriminator_config["structural_fusion_weight"]),
        leaky_relu_slope=float(discriminator_config["leaky_relu_slope"]),
        use_structural_branch=bool(ablation["use_structural_branch"]),
        tau_edge=float(structural_config["tau_edge"]),
        gamma=float(structural_config["gamma"]),
        gaussian_sigma=float(structural_config["gaussian_sigma"]),
        sobel_normalisation=float(structural_config["sobel_normalisation"]),
    )
    palette_regulariser = PaletteAwareRegularisation(
        centroids,
        proportions,
        structural_prior,
        epsilon=float(palette_config["numerical_epsilon"]),
    )
    trainer = Trainer(
        generator, discriminator, palette_regulariser,
        train_loader, val_loader, config, device, args.checkpoint_dir,
    )
    if args.resume is not None:
        trainer.resume(args.resume)
    trainer.train()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
