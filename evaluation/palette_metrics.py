"""Palette-KL and flatness violation metrics from Section 3.2."""

from __future__ import annotations

from typing import Tuple

import torch
from torch import Tensor

from preprocessing.preprocess import srgb_to_lab_torch
from priors.structural_prior import StructuralPrior


def palette_soft_counts(
    images: Tensor,
    centroids: Tensor,
    temperature: float,
    structural_prior: StructuralPrior,
) -> Tuple[Tensor, Tensor]:
    """Return soft codebook counts and the selected-pixel count from Eq. (23)."""

    if images.ndim != 4:
        raise ValueError("Expected a B x 3 x H x W image tensor.")
    rgb = images.add(1.0).mul(0.5).clamp(0.0, 1.0)
    lab = srgb_to_lab_torch(rgb)
    squared = (
        lab.unsqueeze(1) - centroids.to(lab).view(1, -1, 3, 1, 1)
    ).square().sum(dim=2)
    assignments = torch.softmax(-squared / temperature, dim=1)
    mask = structural_prior(images).flat_fill_mask.to(assignments.dtype)
    counts = (assignments * mask).sum(dim=(0, 2, 3))
    return counts, mask.sum()


def palette_kl(
    generated_images: Tensor,
    centroids: Tensor,
    target_proportions: Tensor,
    temperature: float,
    structural_prior: StructuralPrior,
    epsilon: float = 1.0e-8,
) -> Tensor:
    """Compute Palette-KL, KL(target || generated), exactly as Eq. (27)."""

    counts, total = palette_soft_counts(
        generated_images, centroids, temperature, structural_prior
    )
    return palette_kl_from_counts(counts, total, target_proportions, epsilon)


def palette_kl_from_counts(
    counts: Tensor,
    total: Tensor,
    target_proportions: Tensor,
    epsilon: float = 1.0e-8,
) -> Tensor:
    """Compute Eq. (27) once from dataset-level accumulated soft counts."""

    predicted = counts / total.clamp_min(1.0)
    predicted = predicted.clamp_min(epsilon)
    predicted = predicted / predicted.sum()
    target = target_proportions.to(predicted).clamp_min(epsilon)
    target = target / target.sum()
    return (target * (target.log() - predicted.log())).sum()


def flatness_violation(
    images: Tensor,
    structural_prior: StructuralPrior,
    gradient_threshold: float = 0.04,
) -> Tensor:
    """Compute the flat-fill gradient violation ratio in Eq. (28)."""

    output = structural_prior(images)
    mask = output.flat_fill_mask
    violations = mask & (output.gradient_magnitude > gradient_threshold)
    return violations.sum().to(images.dtype) / mask.sum().clamp_min(1).to(images.dtype)
