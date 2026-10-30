"""Palette-aware StyleGAN model components."""

from .discriminator import DualDiscriminator
from .generator import Generator
from .lineart_mapping import DifferentiableLineArtMapping

__all__ = ["DualDiscriminator", "Generator", "DifferentiableLineArtMapping"]
