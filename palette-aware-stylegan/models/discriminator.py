"""Independent appearance/structure discriminator branches from Section 2.4.2."""

from __future__ import annotations

import math
from typing import Dict, Iterable, Optional

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from models.generator import EqualLinear
from models.lineart_mapping import DifferentiableLineArtMapping


class ResidualDownBlock(nn.Module):
    """Residual main and skip paths corresponding to Eqs. (14)-(16)."""

    def __init__(self, in_channels: int, out_channels: int, slope: float = 0.2) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels, 3, padding=1)
        self.conv2 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.skip = nn.Conv2d(in_channels, out_channels, 1)
        self.slope = slope

    def forward(self, value: Tensor) -> Tensor:
        main = F.leaky_relu(self.conv1(value), negative_slope=self.slope)
        main = F.leaky_relu(self.conv2(main), negative_slope=self.slope)
        main = F.avg_pool2d(main, 2)
        skip = F.avg_pool2d(self.skip(value), 2)
        return main + skip


def minibatch_standard_deviation(value: Tensor) -> Tensor:
    """Append the replicated within-batch statistic used in Eq. (17)."""

    if value.shape[0] > 1:
        statistic = torch.sqrt(value.var(dim=0, unbiased=False) + 1.0e-8).mean()
    else:
        statistic = value.new_zeros(())
    channel = statistic.view(1, 1, 1, 1).expand(value.shape[0], 1, 4, 4)
    return torch.cat([value, channel], dim=1)


class DiscriminatorBranch(nn.Module):
    """One independent multi-scale residual pyramid and scalar head."""

    def __init__(
        self,
        input_channels: int,
        resolution: int,
        channel_base: int,
        channel_max: int,
        head_channels: int,
        head_features: int,
        slope: float,
    ) -> None:
        super().__init__()
        if resolution < 4 or resolution & (resolution - 1):
            raise ValueError("Discriminator resolution must be a power of two at least 4.")
        self.resolution = resolution
        self.resolutions = [
            2**power for power in range(int(math.log2(resolution)), 1, -1)
        ]
        channels: Dict[int, int] = {
            item: min(channel_base // item, channel_max) for item in self.resolutions
        }
        self.from_input = nn.Conv2d(input_channels, channels[resolution], 1)
        self.stages = nn.ModuleDict()
        for item in self.resolutions[:-1]:
            self.stages[str(item)] = ResidualDownBlock(
                channels[item], channels[item // 2], slope
            )
        self.head_conv = nn.Conv2d(channels[4] + 1, head_channels, 3, padding=1)
        self.head_fc = EqualLinear(head_channels * 4 * 4, head_features)
        self.output_fc = EqualLinear(head_features, 1)
        self.slope = slope

    def forward(self, value: Tensor) -> Tensor:
        feature = self.from_input(value)
        for item in self.resolutions[:-1]:
            feature = self.stages[str(item)](feature)
        feature = minibatch_standard_deviation(feature)
        feature = F.leaky_relu(self.head_conv(feature), negative_slope=self.slope)
        feature = feature.flatten(1)
        feature = F.leaky_relu(self.head_fc(feature), negative_slope=self.slope)
        return self.output_fc(feature).squeeze(1)

    def set_frozen_resolutions(self, resolutions: Iterable[int]) -> None:
        """Freeze only the selected high-resolution input-side stages."""

        frozen = {int(item) for item in resolutions}
        for parameter in self.from_input.parameters():
            parameter.requires_grad = self.resolution not in frozen
        for name, stage in self.stages.items():
            requires_grad = int(name) not in frozen
            for parameter in stage.parameters():
                parameter.requires_grad = requires_grad


class DualDiscriminator(nn.Module):
    """Parallel, non-sharing appearance and structural branches, Eqs. (12)-(18)."""

    def __init__(
        self,
        resolution: int = 512,
        channel_base: int = 32768,
        channel_max: int = 512,
        head_channels: int = 512,
        head_features: int = 512,
        structural_fusion_weight: float = 0.75,
        leaky_relu_slope: float = 0.2,
        use_structural_branch: bool = True,
        tau_edge: float = 0.08,
        gamma: float = 0.02,
        gaussian_sigma: float = 1.0,
        sobel_normalisation: float = 8.0,
    ) -> None:
        super().__init__()
        branch_args = (
            resolution, channel_base, channel_max, head_channels,
            head_features, leaky_relu_slope,
        )
        self.appearance = DiscriminatorBranch(3, *branch_args)
        self.structure = DiscriminatorBranch(1, *branch_args)
        self.lineart_mapping = DifferentiableLineArtMapping(
            tau_edge, gamma, gaussian_sigma, sobel_normalisation
        )
        self.structural_fusion_weight = float(structural_fusion_weight)
        self.use_structural_branch = bool(use_structural_branch)

    def score_augmented(
        self,
        augmented_rgb: Tensor,
        line_art: Optional[Tensor] = None,
    ) -> Dict[str, Tensor]:
        """Score already augmented A-port signals and fuse them using Eq. (18)."""

        appearance_score = self.appearance(augmented_rgb)
        mapped = self.lineart_mapping(augmented_rgb) if line_art is None else line_art
        if self.use_structural_branch:
            structural_score = self.structure(mapped)
            fused = appearance_score + self.structural_fusion_weight * structural_score
        else:
            structural_score = torch.zeros_like(appearance_score)
            fused = appearance_score
        return {
            "appearance": appearance_score,
            "structural": structural_score,
            "fused": fused,
            "line_art": mapped,
        }

    def forward(self, augmented_rgb: Tensor) -> Dict[str, Tensor]:
        return self.score_augmented(augmented_rgb)

    def set_frozen_resolutions(self, resolutions: Iterable[int]) -> None:
        """Apply the symmetric FreezeD setting to both independent branches."""

        self.appearance.set_frozen_resolutions(resolutions)
        self.structure.set_frozen_resolutions(resolutions)
