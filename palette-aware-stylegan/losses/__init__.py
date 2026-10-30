"""Adversarial and Palette-aware objectives."""

from .adversarial import (
    discriminator_adversarial_loss,
    generator_adversarial_loss,
    path_length_regularisation,
    r1_regularisation,
)
from .palette_aware import PaletteAwareRegularisation

__all__ = [
    "PaletteAwareRegularisation",
    "discriminator_adversarial_loss",
    "generator_adversarial_loss",
    "path_length_regularisation",
    "r1_regularisation",
]
