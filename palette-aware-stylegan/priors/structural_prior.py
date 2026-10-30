"""Differentiable line mapping and region construction from Section 2.3.2."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

import numpy as np
import torch
import torch.nn.functional as F
from scipy.ndimage import distance_transform_edt
from torch import Tensor, nn


@dataclass
class StructuralPriorOutput:
    """Auditable outputs of the structural-prior pipeline."""

    line_art: Tensor
    line_mask: Tensor
    flat_fill_mask: Tensor
    distance_weight: Tensor
    gradient_magnitude: Tensor


def _gaussian_kernel(
    sigma: float,
    dtype: torch.dtype,
    device: torch.device,
) -> Tensor:
    if sigma <= 0.0:
        return torch.ones(1, 1, 1, 1, dtype=dtype, device=device)
    radius = max(1, int(math.ceil(3.0 * sigma)))
    coordinate = torch.arange(-radius, radius + 1, dtype=dtype, device=device)
    kernel_1d = torch.exp(-(coordinate.square()) / (2.0 * sigma**2))
    kernel_1d = kernel_1d / kernel_1d.sum()
    kernel_2d = torch.outer(kernel_1d, kernel_1d)
    return kernel_2d.view(1, 1, *kernel_2d.shape)


def _to_zero_one(image: Tensor, input_range: Literal["zero_one", "minus_one_one"]) -> Tensor:
    if input_range == "minus_one_one":
        return image.add(1.0).mul(0.5).clamp(0.0, 1.0)
    if input_range == "zero_one":
        return image.clamp(0.0, 1.0)
    raise ValueError("input_range must be zero_one or minus_one_one.")


def perceptual_luminance(image: Tensor, input_range: str = "minus_one_one") -> Tensor:
    """Compute linear-light BT.709 luminance from Eq. (A.2)."""

    if image.ndim != 4 or image.shape[1] != 3:
        raise ValueError("Expected a B x 3 x H x W RGB tensor.")
    rgb = _to_zero_one(image, input_range)  # type: ignore[arg-type]
    linear = torch.where(
        rgb <= 0.04045,
        rgb / 12.92,
        ((rgb + 0.055) / 1.055).pow(2.4),
    )
    coefficients = linear.new_tensor([0.2126, 0.7152, 0.0722]).view(1, 3, 1, 1)
    return (linear * coefficients).sum(dim=1, keepdim=True)


def gradient_magnitude(
    luminance: Tensor,
    gaussian_sigma: float,
    sobel_normalisation: float = 8.0,
) -> Tensor:
    """Gaussian pre-smoothing followed by Eq. (B.1) Sobel magnitude."""

    kernel = _gaussian_kernel(gaussian_sigma, luminance.dtype, luminance.device)
    padding = kernel.shape[-1] // 2
    smoothed = F.conv2d(luminance, kernel, padding=padding)
    sobel_x = luminance.new_tensor(
        [[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]]
    ).view(1, 1, 3, 3) / sobel_normalisation
    sobel_y = sobel_x.transpose(-1, -2)
    gx = F.conv2d(smoothed, sobel_x, padding=1)
    gy = F.conv2d(smoothed, sobel_y, padding=1)
    return torch.sqrt(gx.square() + gy.square() + 1.0e-12)


def continuous_line_art(magnitude: Tensor, tau_edge: float, gamma: float) -> Tensor:
    """Apply the differentiable continuous line-art mapping in Eq. (1)."""

    if gamma <= 0.0:
        raise ValueError("gamma must be positive.")
    return 0.5 * (1.0 + torch.tanh((magnitude - tau_edge) / gamma))


class StructuralPrior(nn.Module):
    """Build the structural prior from Eqs. (1)-(2) and (B.1)-(B.3).

    The line-art field remains differentiable. Thresholding, dilation, and the
    Euclidean distance transform construct masks and weights used as fixed supports.
    """

    def __init__(
        self,
        tau_edge: float = 0.08,
        gamma: float = 0.02,
        line_threshold: float = 0.30,
        dilation_radius: int = 2,
        buffer_radius: int = 3,
        max_distance: int = 12,
        gaussian_sigma: float = 1.0,
        sobel_normalisation: float = 8.0,
        input_range: Literal["zero_one", "minus_one_one"] = "minus_one_one",
    ) -> None:
        super().__init__()
        self.tau_edge = float(tau_edge)
        self.gamma = float(gamma)
        self.line_threshold = float(line_threshold)
        self.dilation_radius = int(dilation_radius)
        self.buffer_radius = int(buffer_radius)
        self.max_distance = int(max_distance)
        self.gaussian_sigma = float(gaussian_sigma)
        self.sobel_normalisation = float(sobel_normalisation)
        self.input_range = input_range

    def line_art(self, image: Tensor) -> tuple[Tensor, Tensor]:
        """Return the continuous field in Eq. (1) and magnitude in Eq. (B.1)."""

        luminance = perceptual_luminance(image, self.input_range)
        magnitude = gradient_magnitude(
            luminance, self.gaussian_sigma, self.sobel_normalisation
        )
        return continuous_line_art(magnitude, self.tau_edge, self.gamma), magnitude

    def forward(self, image: Tensor) -> StructuralPriorOutput:
        """Return line art, dilated line mask, flat-fill mask, and distance weight."""

        line_art, magnitude = self.line_art(image)
        raw_mask = line_art > self.line_threshold
        if self.dilation_radius > 0:
            kernel = 2 * self.dilation_radius + 1
            line_mask = F.max_pool2d(
                raw_mask.to(line_art.dtype), kernel_size=kernel, stride=1,
                padding=self.dilation_radius,
            ) > 0.0
        else:
            line_mask = raw_mask

        distance_arrays = []
        for item in line_mask.detach().cpu().numpy()[:, 0].astype(bool):
            if item.any():
                distance = distance_transform_edt(~item).astype(np.float32)
            else:
                distance = np.full(item.shape, self.max_distance, dtype=np.float32)
            distance_arrays.append(distance)
        distance = torch.from_numpy(np.stack(distance_arrays, axis=0)).to(
            device=image.device, dtype=line_art.dtype
        ).unsqueeze(1)
        flat_fill_mask = distance >= float(self.buffer_radius)
        distance_weight = (distance / float(self.max_distance)).clamp(0.0, 1.0)
        return StructuralPriorOutput(
            line_art=line_art,
            line_mask=line_mask,
            flat_fill_mask=flat_fill_mask,
            distance_weight=distance_weight,
            gradient_magnitude=magnitude,
        )
