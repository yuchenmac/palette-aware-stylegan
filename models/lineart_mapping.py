"""Differentiable structural mapping used by the discriminator A-port."""

from __future__ import annotations

from torch import Tensor, nn

from priors.structural_prior import StructuralPrior


class DifferentiableLineArtMapping(nn.Module):
    """Implements the structural mapping in Eq. (C.1) of the manuscript."""

    def __init__(
        self,
        tau_edge: float = 0.08,
        gamma: float = 0.02,
        gaussian_sigma: float = 1.0,
        sobel_normalisation: float = 8.0,
    ) -> None:
        super().__init__()
        self.prior = StructuralPrior(
            tau_edge=tau_edge,
            gamma=gamma,
            gaussian_sigma=gaussian_sigma,
            sobel_normalisation=sobel_normalisation,
            input_range="minus_one_one",
        )

    def forward(self, image: Tensor) -> Tensor:
        line_art, _ = self.prior.line_art(image)
        return line_art
