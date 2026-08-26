"""Generate images from a future user-supplied checkpoint."""

from __future__ import annotations

import argparse
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
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    if not args.checkpoint.is_file():
        print("Trained checkpoints from the study are not distributed with this code.")
        return 2
    if args.num_images < 1 or args.batch_size < 1:
        print("num-images and batch-size must be positive.")
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
