"""Non-saturating adversarial and lazy regularisation objectives."""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn.functional as F
from torch import Tensor


def generator_adversarial_loss(fake_score: Tensor) -> Tensor:
    """Non-saturating generator objective in Eq. (19)."""

    return F.softplus(-fake_score).mean()


def discriminator_adversarial_loss(real_score: Tensor, fake_score: Tensor) -> Tensor:
    """Non-saturating discriminator objective in Eq. (20)."""

    return F.softplus(-real_score).mean() + F.softplus(fake_score).mean()


def r1_regularisation(
    appearance_real_score: Tensor,
    structural_real_score: Tensor,
    augmented_real_image: Tensor,
    real_line_art: Tensor,
    appearance_weight: float = 10.0,
    structural_weight: float = 6.0,
) -> Tensor:
    """Two-branch separable R1 penalty defined in Eq. (D.1)."""

    appearance_gradient = torch.autograd.grad(
        appearance_real_score.sum(), augmented_real_image,
        create_graph=True, retain_graph=True, only_inputs=True,
    )[0]
    appearance_penalty = appearance_gradient.square().flatten(1).sum(dim=1).mean()
    if structural_real_score.requires_grad:
        structural_gradient = torch.autograd.grad(
            structural_real_score.sum(), real_line_art,
            create_graph=True, retain_graph=True, only_inputs=True,
        )[0]
        structural_penalty = structural_gradient.square().flatten(1).sum(dim=1).mean()
    else:
        structural_penalty = appearance_penalty.new_zeros(())
    return (
        0.5 * appearance_weight * appearance_penalty
        + 0.5 * structural_weight * structural_penalty
    )


def path_length_regularisation(
    image: Tensor,
    style: Tensor,
    mean_path_length: Tensor,
    decay: float = 0.01,
) -> Tuple[Tensor, Tensor, Tensor]:
    """Style-to-image path-length penalty defined in Eq. (D.2)."""

    noise = torch.randn_like(image) / (image.shape[2] * image.shape[3]) ** 0.5
    gradient = torch.autograd.grad(
        (image * noise).sum(), style, create_graph=True, retain_graph=True,
        only_inputs=True,
    )[0]
    lengths = torch.sqrt(gradient.square().sum(dim=1) + 1.0e-8)
    updated_mean = mean_path_length + decay * (lengths.detach().mean() - mean_path_length)
    penalty = (lengths - updated_mean.detach()).square().mean()
    return penalty, updated_mean.detach(), lengths.detach()
