"""Palette-aware U-port regularisation from Eqs. (21)-(25)."""

from __future__ import annotations

from typing import Dict

import torch
from torch import Tensor, nn

from preprocessing.preprocess import srgb_to_lab_torch
from priors.structural_prior import StructuralPrior


class PaletteAwareRegularisation(nn.Module):
    """Soft assignment, reconstruction, ratio alignment, and entropy terms."""

    def __init__(
        self,
        centroids: Tensor,
        target_proportions: Tensor,
        structural_prior: StructuralPrior,
        epsilon: float = 1.0e-8,
    ) -> None:
        super().__init__()
        if centroids.ndim != 2 or centroids.shape[1] != 3:
            raise ValueError("Palette centroids must have shape K x 3.")
        if target_proportions.shape != (centroids.shape[0],):
            raise ValueError("Palette proportions must have shape K.")
        self.register_buffer("centroids", centroids.float())
        proportions = target_proportions.float().clamp_min(epsilon)
        self.register_buffer("target_proportions", proportions / proportions.sum())
        self.structural_prior = structural_prior
        self.epsilon = float(epsilon)

    def soft_assign(self, lab: Tensor, temperature: float) -> Tensor:
        """Compute the temperature-controlled Gaussian assignment in Eq. (21)."""

        if temperature <= 0.0:
            raise ValueError("Palette temperature must be positive.")
        difference = lab.unsqueeze(1) - self.centroids.view(1, -1, 3, 1, 1)
        squared_distance = difference.square().sum(dim=2)
        return torch.softmax(-squared_distance / temperature, dim=1)

    def forward(
        self,
        unaugmented_image: Tensor,
        temperature: float,
        reconstruction_weight: float = 1.0,
        proportion_weight: float = 0.25,
        entropy_weight: float = 0.10,
    ) -> Dict[str, Tensor]:
        """Evaluate the complete Palette-aware objective in Eq. (25)."""

        rgb = unaugmented_image.add(1.0).mul(0.5).clamp(0.0, 1.0)
        lab = srgb_to_lab_torch(rgb)
        structural = self.structural_prior(unaugmented_image)
        mask = structural.flat_fill_mask[:, 0].to(lab.dtype)
        selected_pixels = mask.sum(dim=(1, 2))
        valid = selected_pixels > 0.0
        denominator = selected_pixels.clamp_min(1.0)
        assignment = self.soft_assign(lab, temperature)

        reconstruction = torch.einsum("bkhw,kc->bchw", assignment, self.centroids)
        reconstruction_error = (lab - reconstruction).abs().sum(dim=1)
        rec_per_image = (reconstruction_error * mask).sum(dim=(1, 2)) / denominator
        reconstruction_loss = (
            rec_per_image[valid].mean() if bool(valid.any()) else lab.sum() * 0.0
        )

        predicted = (assignment * mask.unsqueeze(1)).sum(dim=(2, 3))
        predicted = predicted / denominator.unsqueeze(1)
        target = self.target_proportions.view(1, -1)
        predicted = torch.where(valid.unsqueeze(1), predicted, target.expand_as(predicted))
        predicted = predicted.clamp_min(self.epsilon)
        predicted = predicted / predicted.sum(dim=1, keepdim=True)
        proportion_per_image = (
            predicted * (predicted.log() - target.log())
        ).sum(dim=1)
        proportion_loss = (
            proportion_per_image[valid].mean()
            if bool(valid.any()) else lab.sum() * 0.0
        )

        entropy_map = -(assignment.clamp_min(self.epsilon).log() * assignment).sum(dim=1)
        entropy_per_image = (entropy_map * mask).sum(dim=(1, 2)) / denominator
        entropy_loss = (
            entropy_per_image[valid].mean() if bool(valid.any()) else lab.sum() * 0.0
        )
        total = (
            reconstruction_weight * reconstruction_loss
            + proportion_weight * proportion_loss
            + entropy_weight * entropy_loss
        )
        return {
            "loss": total,
            "reconstruction": reconstruction_loss,
            "proportion": proportion_loss,
            "entropy": entropy_loss,
            "predicted_proportions": (
                predicted[valid].mean(dim=0) if bool(valid.any()) else self.target_proportions
            ),
            "flat_fill_mask": structural.flat_fill_mask,
        }
