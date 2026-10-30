"""Generate images from a specified checkpoint."""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Mapping

import torch
import yaml
from torchvision.utils import save_image

from models.generator import Generator
from training.trainer import set_random_seed


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate with Palette-aware StyleGAN.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Path("config.yaml"))
    parser.add_argument("--num-images", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, default=Path("outputs"))
    parser.add_argument("--overwrite", action="store_true", help="Replace existing sample-NNNNN.png files in the output directory.")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    if not args.checkpoint.is_file():
        print(f"Checkpoint file not found: {args.checkpoint}")
        return 2
    if args.num_images < 1 or args.batch_size < 1:
        print("num-images and batch-size must be positive.")
        return 2
    if not 0 <= args.seed < 2**32:
        parser.error("--seed must be an integer between 0 and 2**32 - 1.")
    if args.output.exists() and not args.output.is_dir():
        parser.error("--output must be a directory.")
    existing_images = [
        path for path in args.output.rglob("*")
        if path.is_file() and path.suffix.lower() in {".png", ".jpg", ".jpeg", ".tif", ".tiff"}
    ] if args.output.exists() else []
    owned_samples = [
        path for path in existing_images
        if path.parent == args.output and re.fullmatch(r"sample-\d{5,}\.png", path.name)
    ]
    if existing_images and not args.overwrite:
        parser.error("The output directory already contains images. Use an empty directory or --overwrite for existing generated samples.")
    if args.overwrite and len(owned_samples) != len(existing_images):
        parser.error("The output directory contains other image files; choose an empty directory. --overwrite only replaces sample-NNNNN.png files.")
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
    generator_config = config["generator"]
    assert isinstance(generator_config, Mapping)
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
    state = checkpoint.get("ema_generator", checkpoint.get("generator"))
    if state is None:
        raise ValueError("Checkpoint does not contain generator weights.")
    generator.load_state_dict(state)
    device = torch.device(args.device)
    generator.to(device).eval()
    args.output.mkdir(parents=True, exist_ok=True)
    for path in owned_samples:
        path.unlink()
    set_random_seed(args.seed)
    written = 0
    with torch.no_grad():
        while written < args.num_images:
            current = min(args.batch_size, args.num_images - written)
            latent = torch.randn(current, generator.latent_dim, device=device)
            images = generator(latent, noise_mode="const")
            assert isinstance(images, torch.Tensor)
            images = images.add(1.0).mul(0.5).clamp(0.0, 1.0)
            for offset, image in enumerate(images):
                save_image(image, args.output / f"sample-{written + offset:05d}.png")
            written += current
    print(f"Generated {written} images in {args.output}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
