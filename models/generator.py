"""Runnable StyleGAN2-style generator corresponding to Section 2.4.1."""

from __future__ import annotations

import math
from typing import Dict, Literal, Optional, Tuple, Union

import torch
import torch.nn.functional as F
from torch import Tensor, nn


class PixelNorm(nn.Module):
    def forward(self, value: Tensor) -> Tensor:
        return value * torch.rsqrt(value.square().mean(dim=1, keepdim=True) + 1.0e-8)


class EqualLinear(nn.Module):
    """Baseline StyleGAN2 equalised-learning-rate affine or dense layer."""

    def __init__(self, in_features: int, out_features: int, bias: bool = True) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.randn(out_features, in_features))
        self.bias = nn.Parameter(torch.zeros(out_features)) if bias else None
        self.scale = 1.0 / math.sqrt(in_features)

    def forward(self, value: Tensor) -> Tensor:
        return F.linear(value, self.weight * self.scale, self.bias)


class MappingNetwork(nn.Module):
    """Eight-layer mapping network in Eq. (6)."""

    def __init__(self, latent_dim: int, style_dim: int, layers: int = 8) -> None:
        super().__init__()
        modules = [PixelNorm()]
        in_features = latent_dim
        for _ in range(layers):
            modules.extend([EqualLinear(in_features, style_dim), nn.LeakyReLU(0.2)])
            in_features = style_dim
        self.network = nn.Sequential(*modules)

    def forward(self, latent: Tensor) -> Tensor:
        return self.network(latent)


class ModulatedConv2d(nn.Module):
    """Style modulation and optional demodulation from Eqs. (7)-(8)."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        style_dim: int,
        upsample: bool = False,
        demodulate: bool = True,
        epsilon: float = 1.0e-8,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.upsample = upsample
        self.demodulate = demodulate
        self.epsilon = epsilon
        self.weight = nn.Parameter(
            torch.randn(1, out_channels, in_channels, kernel_size, kernel_size)
        )
        self.affine = EqualLinear(style_dim, in_channels)
        nn.init.ones_(self.affine.bias)
        self.weight_scale = 1.0 / math.sqrt(in_channels * kernel_size * kernel_size)

    def forward(self, value: Tensor, style: Tensor) -> Tensor:
        batch, _, height, width = value.shape
        if self.upsample:
            value = F.interpolate(value, scale_factor=2.0, mode="nearest")
            height, width = value.shape[-2:]
        modulation = self.affine(style).view(batch, 1, self.in_channels, 1, 1)
        weight = self.weight * self.weight_scale * modulation
        if self.demodulate:
            demodulation = torch.rsqrt(weight.square().sum((2, 3, 4)) + self.epsilon)
            weight = weight * demodulation.view(batch, self.out_channels, 1, 1, 1)
        grouped_weight = weight.reshape(
            batch * self.out_channels,
            self.in_channels,
            self.kernel_size,
            self.kernel_size,
        )
        grouped_input = value.reshape(1, batch * self.in_channels, height, width)
        output = F.conv2d(
            grouped_input,
            grouped_weight,
            padding=self.kernel_size // 2,
            groups=batch,
        )
        return output.reshape(batch, self.out_channels, output.shape[-2], output.shape[-1])


class NoiseInjection(nn.Module):
    """Single-channel Gaussian noise injection from Eq. (9)."""

    def __init__(self, resolution: int) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(1))
        self.register_buffer("constant_noise", torch.randn(1, 1, resolution, resolution))

    def forward(
        self,
        value: Tensor,
        mode: Literal["random", "const", "none"] = "random",
    ) -> Tensor:
        if mode == "none":
            return value
        if mode == "const":
            noise = self.constant_noise.expand(value.shape[0], -1, -1, -1)
        elif mode == "random":
            noise = torch.randn(
                value.shape[0], 1, value.shape[2], value.shape[3],
                device=value.device, dtype=value.dtype,
            )
        else:
            raise ValueError("noise mode must be random, const, or none.")
        return value + self.weight.view(1, 1, 1, 1) * noise


class StyledConv(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        style_dim: int,
        resolution: int,
        upsample: bool,
        slope: float,
        epsilon: float,
    ) -> None:
        super().__init__()
        self.conv = ModulatedConv2d(
            in_channels, out_channels, 3, style_dim, upsample, True, epsilon
        )
        self.noise = NoiseInjection(resolution)
        self.bias = nn.Parameter(torch.zeros(out_channels))
        self.slope = slope

    def forward(self, value: Tensor, style: Tensor, noise_mode: str) -> Tensor:
        output = self.conv(value, style)
        output = self.noise(output, noise_mode)  # type: ignore[arg-type]
        output = output + self.bias.view(1, -1, 1, 1)
        return F.leaky_relu(output, negative_slope=self.slope)


class ToRGB(nn.Module):
    """StyleGAN2 baseline implementation of the 1 x 1 mapping in Eq. (10)."""

    def __init__(self, in_channels: int, style_dim: int, epsilon: float) -> None:
        super().__init__()
        self.conv = ModulatedConv2d(
            in_channels, 3, 1, style_dim, demodulate=False, epsilon=epsilon
        )
        self.bias = nn.Parameter(torch.zeros(3))

    def forward(self, value: Tensor, style: Tensor) -> Tensor:
        return self.conv(value, style) + self.bias.view(1, 3, 1, 1)


class Generator(nn.Module):
    """Progressive 4-to-512 StyleGAN2-style generator from Eqs. (6)-(11).

    Uses nearest-neighbour feature/image upsampling and compact channel allocation.
    """

    def __init__(
        self,
        latent_dim: int = 512,
        style_dim: int = 512,
        mapping_layers: int = 8,
        resolution: int = 512,
        channel_base: int = 32768,
        channel_max: int = 512,
        leaky_relu_slope: float = 0.2,
        demodulation_epsilon: float = 1.0e-8,
    ) -> None:
        super().__init__()
        if resolution < 4 or resolution & (resolution - 1):
            raise ValueError("Generator resolution must be a power of two at least 4.")
        self.latent_dim = latent_dim
        self.style_dim = style_dim
        self.resolution = resolution
        self.mapping = MappingNetwork(latent_dim, style_dim, mapping_layers)
        self.resolutions = [2**power for power in range(2, int(math.log2(resolution)) + 1)]
        self.channels: Dict[int, int] = {
            item: min(channel_base // item, channel_max) for item in self.resolutions
        }
        self.constant = nn.Parameter(torch.randn(1, self.channels[4], 4, 4))
        self.blocks = nn.ModuleDict()
        self.to_rgbs = nn.ModuleDict()
        for item in self.resolutions:
            in_channels = self.channels[4] if item == 4 else self.channels[item // 2]
            self.blocks[str(item)] = nn.ModuleList(
                [
                    StyledConv(
                        in_channels, self.channels[item], style_dim, item,
                        upsample=item > 4, slope=leaky_relu_slope,
                        epsilon=demodulation_epsilon,
                    ),
                    StyledConv(
                        self.channels[item], self.channels[item], style_dim, item,
                        upsample=False, slope=leaky_relu_slope,
                        epsilon=demodulation_epsilon,
                    ),
                ]
            )
            self.to_rgbs[str(item)] = ToRGB(
                self.channels[item], style_dim, demodulation_epsilon
            )

    def forward(
        self,
        latent: Tensor,
        noise_mode: Literal["random", "const", "none"] = "random",
        return_latents: bool = False,
    ) -> Union[Tensor, Tuple[Tensor, Tensor]]:
        """Map z to w, synthesise RGB, and route the unaugmented U-port output."""

        style = self.mapping(latent)
        feature = self.constant.expand(latent.shape[0], -1, -1, -1)
        image: Optional[Tensor] = None
        for resolution in self.resolutions:
            block_a, block_b = self.blocks[str(resolution)]
            feature = block_a(feature, style, noise_mode)
            feature = block_b(feature, style, noise_mode)
            partial = self.to_rgbs[str(resolution)](feature, style)
            image = partial if image is None else F.interpolate(
                image, scale_factor=2.0, mode="nearest"
            ) + partial
        if image is None:
            raise RuntimeError("Generator did not produce an image.")
        return (image, style) if return_latents else image
