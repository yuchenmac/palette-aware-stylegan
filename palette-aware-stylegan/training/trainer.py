"""Complete discriminator-first U-port/A-port training loop from Section 2.5."""

from __future__ import annotations

import copy
import json
import math
import random
import warnings
from contextlib import contextmanager
from pathlib import Path
from typing import Dict, Iterator, List, Mapping, MutableMapping, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.optim import Adam
from torch.utils.data import DataLoader
from tqdm import tqdm

from evaluation.palette_metrics import (
    flatness_counts,
    palette_kl_from_counts,
    palette_soft_counts,
)
from evaluation.structure_metrics import (
    closure_rate,
    edge_map_from_image,
    line_continuity_index,
)
from losses.adversarial import (
    discriminator_adversarial_loss,
    generator_adversarial_loss,
    path_length_regularisation,
    r1_regularisation,
)
from losses.palette_aware import PaletteAwareRegularisation
from models.discriminator import DualDiscriminator
from models.generator import Generator
from priors.structural_prior import StructuralPrior
from training.loss_logging import LossLogger


def set_random_seed(seed: int) -> None:
    """Set Python, NumPy, and PyTorch seeds for auditable runs."""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def warn_metric_failure(metric_name: str, exc: Exception) -> None:
    """Expose validation metric failures without substituting a numeric result."""

    warnings.warn(
        f"{metric_name} computation could not be completed during validation: {exc}",
        RuntimeWarning,
        stacklevel=2,
    )


def _gaussian_kernel(sigma: float, dtype: torch.dtype, device: torch.device) -> Tensor:
    radius = max(1, int(math.ceil(3.0 * sigma)))
    coordinate = torch.arange(-radius, radius + 1, dtype=dtype, device=device)
    kernel = torch.exp(-coordinate.square() / (2.0 * sigma**2))
    kernel = kernel / kernel.sum()
    return torch.outer(kernel, kernel)


class UnifiedAugmentation(nn.Module):
    """Apply Eq. (12) in the fixed flip-affine-mask-colour-noise/blur order.

    Each primitive is activated by Bernoulli(p). forward_pair samples one state and
    applies it to both real and fake samples, preserving the manuscript's unified
    A-port observation domain.
    """

    def __init__(self, config: Mapping[str, object]) -> None:
        super().__init__()
        self.probability = float(config["initial_probability"])
        self.translate_fraction = float(config["translate_fraction"])
        self.rotate_degrees = float(config["rotate_degrees"])
        self.scale_log2_std = float(config["scale_log2_std"])
        self.cutout_fraction = float(config["cutout_fraction"])
        self.brightness = float(config["brightness"])
        self.contrast = float(config["contrast"])
        self.saturation = float(config["saturation"])
        self.noise_std = float(config["noise_std"])
        self.blur_sigma = float(config["blur_sigma"])

    def _activation(self, batch: int, device: torch.device) -> Tensor:
        return (torch.rand(batch, 1, 1, 1, device=device) < self.probability)

    def _sample_state(self, value: Tensor) -> Dict[str, Tensor]:
        batch, _, height, width = value.shape
        device, dtype = value.device, value.dtype
        flip_on = self._activation(batch, device)
        flip_flag = (torch.rand(batch, 1, 1, 1, device=device) < 0.5) & flip_on

        affine_on = self._activation(batch, device).view(batch)
        angle = (torch.rand(batch, device=device, dtype=dtype) * 2.0 - 1.0)
        angle = angle * math.radians(self.rotate_degrees)
        scale = torch.pow(
            value.new_tensor(2.0),
            torch.randn(batch, device=device, dtype=dtype) * self.scale_log2_std,
        )
        tx = (torch.rand(batch, device=device, dtype=dtype) * 2.0 - 1.0) * self.translate_fraction
        ty = (torch.rand(batch, device=device, dtype=dtype) * 2.0 - 1.0) * self.translate_fraction
        angle = torch.where(affine_on, angle, torch.zeros_like(angle))
        scale = torch.where(affine_on, scale, torch.ones_like(scale))
        tx = torch.where(affine_on, tx, torch.zeros_like(tx))
        ty = torch.where(affine_on, ty, torch.zeros_like(ty))
        cosine, sine = torch.cos(angle) * scale, torch.sin(angle) * scale
        theta = torch.stack([cosine, -sine, tx, sine, cosine, ty], dim=1).view(batch, 2, 3)

        mask_on = self._activation(batch, device)
        centre_x = torch.randint(0, width, (batch,), device=device)
        centre_y = torch.randint(0, height, (batch,), device=device)
        cutout_h = max(1, int(round(height * self.cutout_fraction)))
        cutout_w = max(1, int(round(width * self.cutout_fraction)))
        yy = torch.arange(height, device=device).view(1, height, 1)
        xx = torch.arange(width, device=device).view(1, 1, width)
        inside_y = (yy >= (centre_y - cutout_h // 2).view(-1, 1, 1)) & (
            yy < (centre_y + (cutout_h + 1) // 2).view(-1, 1, 1)
        )
        inside_x = (xx >= (centre_x - cutout_w // 2).view(-1, 1, 1)) & (
            xx < (centre_x + (cutout_w + 1) // 2).view(-1, 1, 1)
        )
        cutout = ~(inside_y & inside_x & mask_on.view(batch, 1, 1))

        colour_on = self._activation(batch, device).to(dtype)
        brightness = (
            (torch.rand(batch, 1, 1, 1, device=device, dtype=dtype) * 2.0 - 1.0)
            * self.brightness * colour_on
        )
        contrast = 1.0 + (
            (torch.rand(batch, 1, 1, 1, device=device, dtype=dtype) * 2.0 - 1.0)
            * self.contrast * colour_on
        )
        saturation = 1.0 + (
            (torch.rand(batch, 1, 1, 1, device=device, dtype=dtype) * 2.0 - 1.0)
            * self.saturation * colour_on
        )

        final_on = self._activation(batch, device).to(dtype)
        noise = torch.randn_like(value) * self.noise_std * final_on
        return {
            "flip": flip_flag,
            "theta": theta,
            "cutout": cutout.unsqueeze(1),
            "brightness": brightness,
            "contrast": contrast,
            "saturation": saturation,
            "noise": noise,
            "blur_on": final_on,
        }

    def _apply_state(self, value: Tensor, state: Mapping[str, Tensor]) -> Tensor:
        flipped = torch.flip(value, dims=(3,))
        output = torch.where(state["flip"], flipped, value)
        grid = F.affine_grid(state["theta"], output.shape, align_corners=False)
        output = F.grid_sample(
            output, grid, mode="bilinear", padding_mode="reflection", align_corners=False
        )
        output = output * state["cutout"].to(output.dtype)
        channel_mean = output.mean(dim=1, keepdim=True)
        output = channel_mean + state["saturation"] * (output - channel_mean)
        image_mean = output.mean(dim=(1, 2, 3), keepdim=True)
        output = image_mean + state["contrast"] * (output - image_mean)
        output = output + state["brightness"] + state["noise"]
        if self.blur_sigma > 0.0:
            kernel = _gaussian_kernel(self.blur_sigma, output.dtype, output.device)
            kernel = kernel.view(1, 1, *kernel.shape).repeat(output.shape[1], 1, 1, 1)
            blurred = F.conv2d(
                output, kernel, padding=kernel.shape[-1] // 2, groups=output.shape[1]
            )
            output = torch.where(state["blur_on"].bool(), blurred, output)
        return output

    def forward(self, value: Tensor) -> Tensor:
        if self.probability <= 0.0:
            return value
        return self._apply_state(value, self._sample_state(value))

    def forward_pair(self, real: Tensor, fake: Tensor) -> Tuple[Tensor, Tensor]:
        if real.shape != fake.shape:
            raise ValueError("Paired A-port augmentation requires equal real/fake shapes.")
        if self.probability <= 0.0:
            return real, fake
        state = self._sample_state(real)
        return self._apply_state(real, state), self._apply_state(fake, state)


@contextmanager
def _temporarily_disable_gradients(module: nn.Module) -> Iterator[None]:
    flags = [parameter.requires_grad for parameter in module.parameters()]
    try:
        for parameter in module.parameters():
            parameter.requires_grad_(False)
        yield
    finally:
        for parameter, flag in zip(module.parameters(), flags):
            parameter.requires_grad_(flag)


class Trainer:
    """Coordinate U-port, A-port, ADA, FreezeD, lazy penalties, EMA, and validation."""

    def __init__(
        self,
        generator: Generator,
        discriminator: DualDiscriminator,
        palette_regulariser: PaletteAwareRegularisation,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader],
        config: MutableMapping[str, object],
        device: torch.device,
        checkpoint_dir: Path,
        *,
        prior_feedback_path: Optional[Path] = None,
    ) -> None:
        self.generator = generator.to(device)
        self.discriminator = discriminator.to(device)
        self.palette_regulariser = palette_regulariser.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.device = device
        self.checkpoint_dir = Path(checkpoint_dir)
        self.prior_feedback_path = Path(prior_feedback_path) if prior_feedback_path else None
        self.last_prior_feedback_id: Optional[str] = None

        training = config["training"]  # type: ignore[index]
        regularisation = config["regularisation"]  # type: ignore[index]
        ada = config["ada"]  # type: ignore[index]
        freezed = config["freezed"]  # type: ignore[index]
        ablation = config["ablation"]  # type: ignore[index]
        assert isinstance(training, Mapping)
        assert isinstance(regularisation, Mapping)
        assert isinstance(ada, Mapping)
        assert isinstance(freezed, Mapping)
        assert isinstance(ablation, Mapping)

        betas = tuple(float(item) for item in training["adam_betas"])
        self.optimiser_g = Adam(
            self.generator.parameters(), lr=float(training["learning_rate_g"]), betas=betas
        )
        self.optimiser_d = Adam(
            self.discriminator.parameters(), lr=float(training["learning_rate_d"]), betas=betas
        )
        self.ema_generator = copy.deepcopy(self.generator).eval()
        for parameter in self.ema_generator.parameters():
            parameter.requires_grad_(False)

        self.use_palette = bool(ablation["use_palette_aware"])
        self.use_ada = bool(ada["enabled"]) and bool(ablation["use_ada"])
        self.use_freezed = bool(freezed["enabled"]) and bool(ablation["use_freezed"])
        self.validation_metric_name = str(freezed.get("validation_metric", "fid"))
        if self.validation_metric_name not in {"fid", "kid", "palette_kl"}:
            raise ValueError("freezed.validation_metric must be fid, kid, or palette_kl.")
        self.augment = UnifiedAugmentation(ada).to(device)
        if not self.use_ada:
            self.augment.probability = 0.0
        self.ada_step_size = float(ada["step_size"])
        self.ada_max_probability = float(ada["max_probability"])

        self.frozen_resolutions = [
            int(item) for item in freezed["initially_frozen_resolutions"]
        ] if self.use_freezed else []
        self.discriminator.set_frozen_resolutions(self.frozen_resolutions)

        self.step = 0
        self.mean_path_length = torch.zeros((), device=device)
        self.palette_weight_multiplier = 1.0
        self.annealing_advance = 0.0
        self.unfreeze_postponements = 0
        self.real_hit_history: List[float] = []
        self.fake_score_history: List[float] = []
        self.palette_kl_history: List[float] = []
        self.validation_history: List[float] = []
        self.gradient_ratio_history: List[float] = []
        self.last_processed_monitoring_window = -1
        self._train_iterator: Optional[Iterator[Mapping[str, object]]] = None
        self.loss_logger = LossLogger(
            self.checkpoint_dir / "logs",
            log_interval=int(training["log_interval"]),
            effective_batch_size=int(training["batch_size"]) * int(training["gradient_accumulation"]),
            ema_half_life_steps=training.get("loss_ema_half_life_steps"),
        )
        self.last_loss_summary: Optional[Dict[str, object]] = None

    def _next_batch(self) -> Mapping[str, object]:
        if self._train_iterator is None:
            self._train_iterator = iter(self.train_loader)
        try:
            return next(self._train_iterator)
        except StopIteration:
            self._train_iterator = iter(self.train_loader)
            return next(self._train_iterator)

    def _palette_schedule(self) -> Tuple[float, float, float]:
        palette = self.config["palette_loss"]
        training = self.config["training"]
        assert isinstance(palette, Mapping) and isinstance(training, Mapping)
        total_steps = float(training["total_steps"])
        anneal_steps = max(1.0, total_steps * float(palette["temperature_anneal_fraction"]))
        progress = min(1.0, self.step / anneal_steps + self.annealing_advance)
        temperature = float(palette["temperature_start"]) + progress * (
            float(palette["temperature_end"]) - float(palette["temperature_start"])
        )
        entropy_weight = float(palette["entropy_weight_start"]) + progress * (
            float(palette["entropy_weight_end"]) - float(palette["entropy_weight_start"])
        )
        warmup_steps = max(1.0, total_steps * float(palette["total_weight_warmup_fraction"]))
        warmup = min(1.0, (self.step + 1) / warmup_steps)
        total_weight = (
            float(palette["total_weight"]) * warmup * self.palette_weight_multiplier
        )
        return temperature, entropy_weight, total_weight

    @staticmethod
    def _gradient_norm(loss: Tensor, parameters: List[nn.Parameter]) -> Tensor:
        gradients = torch.autograd.grad(
            loss, parameters, retain_graph=True, allow_unused=True, create_graph=False
        )
        squares = [gradient.detach().square().sum() for gradient in gradients if gradient is not None]
        return torch.sqrt(torch.stack(squares).sum()) if squares else loss.new_zeros(())

    def _discriminator_update(self) -> Dict[str, float]:
        training = self.config["training"]
        regularisation = self.config["regularisation"]
        assert isinstance(training, Mapping) and isinstance(regularisation, Mapping)
        accumulation = int(training["gradient_accumulation"])
        use_r1 = self.step % int(regularisation["r1_interval"]) == 0
        self.optimiser_d.zero_grad(set_to_none=True)
        totals = {"d_total": 0.0, "d_adv": 0.0, "r1": 0.0, "real_hit": 0.0, "fake_score": 0.0}
        for _ in range(accumulation):
            batch = self._next_batch()
            real = batch["image"]
            if not isinstance(real, Tensor):
                raise TypeError("Dataset batches must contain an image tensor.")
            real = real.to(self.device)
            latent = torch.randn(real.shape[0], self.generator.latent_dim, device=self.device)
            with torch.no_grad():
                fake = self.generator(latent)
            assert isinstance(fake, Tensor)
            augmented_real, augmented_fake = self.augment.forward_pair(real, fake.detach())
            augmented_real = augmented_real.detach().requires_grad_(use_r1)
            real_line = self.discriminator.lineart_mapping(augmented_real)
            if use_r1:
                real_line = real_line.detach().requires_grad_(True)
            fake_line = self.discriminator.lineart_mapping(augmented_fake).detach()
            real_scores = self.discriminator.score_augmented(augmented_real, real_line)
            fake_scores = self.discriminator.score_augmented(augmented_fake, fake_line)
            adversarial = discriminator_adversarial_loss(
                real_scores["fused"], fake_scores["fused"]
            )
            r1 = adversarial.new_zeros(())
            if use_r1:
                r1 = r1_regularisation(
                    real_scores["appearance"], real_scores["structural"],
                    augmented_real, real_line,
                    float(regularisation["r1_appearance_weight"]),
                    float(regularisation["r1_structural_weight"]),
                )
            ((adversarial + r1) / accumulation).backward()
            totals["d_total"] += float((adversarial + r1).detach()) / accumulation
            totals["d_adv"] += float(adversarial.detach()) / accumulation
            totals["r1"] += float(r1.detach()) / accumulation
            totals["real_hit"] += float((real_scores["fused"] > 0).float().mean()) / accumulation
            totals["fake_score"] += float(fake_scores["fused"].detach().mean()) / accumulation
        self.optimiser_d.step()
        return totals

    def _generator_update(self) -> Dict[str, float]:
        training = self.config["training"]
        regularisation = self.config["regularisation"]
        palette = self.config["palette_loss"]
        ablation = self.config["ablation"]
        assert all(isinstance(item, Mapping) for item in (training, regularisation, palette, ablation))
        accumulation = int(training["gradient_accumulation"])
        batch_size = int(training["batch_size"])
        use_path = self.step % int(regularisation["path_length_interval"]) == 0
        monitor = (self.step + 1) % int(training["validation_interval"]) == 0
        temperature, entropy_weight, palette_total_weight = self._palette_schedule()
        rec_weight = float(palette["reconstruction_weight"]) if bool(ablation["use_reconstruction"]) else 0.0
        prop_weight = float(palette["proportion_weight"]) if bool(ablation["use_proportion_alignment"]) else 0.0
        ent_weight = entropy_weight if bool(ablation["use_entropy"]) else 0.0
        if not bool(ablation["use_entropy"]) and bool(ablation["fixed_temperature_when_entropy_disabled"]):
            temperature = float(palette["temperature_start"])

        self.optimiser_g.zero_grad(set_to_none=True)
        totals = {
            "g_total": 0.0, "g_adv": 0.0, "palette": 0.0, "palette_rec": 0.0,
            "palette_prop": 0.0, "palette_ent": 0.0, "path": 0.0,
            "palette_weighted": 0.0, "path_weighted": 0.0,
            "gradient_ratio": float("nan"), "temperature": temperature,
            "palette_total_weight": palette_total_weight,
            "palette_valid_images": 0.0, "palette_selected_pixels": 0.0,
        }
        for micro_step in range(accumulation):
            latent = torch.randn(batch_size, self.generator.latent_dim, device=self.device)
            generated, style = self.generator(latent, return_latents=True)
            assert isinstance(generated, Tensor) and isinstance(style, Tensor)
            with _temporarily_disable_gradients(self.discriminator):
                augmented_fake = self.augment(generated)
                fake_scores = self.discriminator.score_augmented(augmented_fake)
                adversarial = generator_adversarial_loss(fake_scores["fused"])
            palette_output = self.palette_regulariser(
                generated, temperature, rec_weight, prop_weight, ent_weight
            )
            palette_loss = palette_output["loss"] if self.use_palette else generated.sum() * 0.0
            path = generated.new_zeros(())
            if use_path:
                path, self.mean_path_length, _ = path_length_regularisation(
                    generated, style, self.mean_path_length,
                    float(regularisation["path_length_decay"]),
                )
            total = (
                adversarial
                + palette_total_weight * palette_loss
                + float(regularisation["path_length_weight"]) * path
            )
            if (
                monitor and micro_step == 0 and self.use_palette
                and int(palette_output["valid_images"]) > 0
            ):
                parameters = [parameter for parameter in self.generator.parameters() if parameter.requires_grad]
                palette_norm = self._gradient_norm(
                    palette_total_weight * palette_loss, parameters
                )
                adversarial_norm = self._gradient_norm(adversarial, parameters)
                if (
                    bool(torch.isfinite(palette_norm))
                    and bool(torch.isfinite(adversarial_norm))
                    and float(adversarial_norm) > 1.0e-12
                ):
                    totals["gradient_ratio"] = float(
                        (palette_norm / adversarial_norm).cpu()
                    )
            (total / accumulation).backward()
            totals["g_total"] += float(total.detach()) / accumulation
            totals["g_adv"] += float(adversarial.detach()) / accumulation
            totals["palette"] += float(palette_loss.detach()) / accumulation
            totals["palette_weighted"] += float((palette_total_weight * palette_loss).detach()) / accumulation
            totals["path_weighted"] += float((float(regularisation["path_length_weight"]) * path).detach()) / accumulation
            totals["palette_rec"] += float(palette_output["reconstruction"].detach()) / accumulation
            totals["palette_prop"] += float(palette_output["proportion"].detach()) / accumulation
            totals["palette_ent"] += float(palette_output["entropy"].detach()) / accumulation
            totals["path"] += float(path.detach()) / accumulation
            totals["palette_valid_images"] += float(palette_output["valid_images"])
            totals["palette_selected_pixels"] += float(palette_output["selected_pixels"])
        self.optimiser_g.step()
        self._update_ema(batch_size * accumulation)
        return totals

    def _update_ema(self, effective_batch: int) -> None:
        training = self.config["training"]
        assert isinstance(training, Mapping)
        half_life = float(training["ema_half_life_kimg"]) * 1000.0
        decay = 0.5 ** (effective_batch / max(half_life, 1.0))
        with torch.no_grad():
            for ema_parameter, parameter in zip(
                self.ema_generator.parameters(), self.generator.parameters()
            ):
                ema_parameter.lerp_(parameter, 1.0 - decay)
            for ema_buffer, buffer in zip(self.ema_generator.buffers(), self.generator.buffers()):
                ema_buffer.copy_(buffer)

    def _update_ada(self, real_hit_rate: float) -> None:
        ada = self.config["ada"]
        assert isinstance(ada, Mapping)
        if not self.use_ada or self.step % int(ada["update_interval"]) != 0:
            return
        update = self.ada_step_size * (real_hit_rate - float(ada["target_hit_rate"]))
        self.augment.probability = float(
            np.clip(self.augment.probability + update, 0.0, self.ada_max_probability)
        )

    def _monitor_overfitting(self) -> bool:
        ada = self.config["ada"]
        freezed = self.config["freezed"]
        assert isinstance(ada, Mapping) and isinstance(freezed, Mapping)
        window = int(freezed["monitor_window"])
        consecutive = int(freezed["consecutive_windows"])
        span = window * consecutive
        if len(self.real_hit_history) < span or len(self.fake_score_history) < span:
            return False
        if len(self.real_hit_history) % window != 0:
            return False
        monitoring_window = len(self.real_hit_history) // window
        if monitoring_window <= self.last_processed_monitoring_window:
            return False
        self.last_processed_monitoring_window = monitoring_window
        hits = np.asarray(self.real_hit_history[-span:])
        fake = np.asarray(self.fake_score_history[-span:])
        fake_increase = (fake[-1] - fake[0]) / max(abs(fake[0]), 1.0e-8)
        overfit = bool(np.all(hits > float(ada["high_hit_rate"])) and fake_increase < 0.05)
        if overfit:
            self.ada_step_size += float(ada["step_adjustment"])
            self.ada_max_probability = float(ada["max_probability"])
            self.unfreeze_postponements = max(self.unfreeze_postponements, 1)
        elif bool(np.all(hits < float(ada["low_hit_rate"]))):
            self.ada_step_size = max(
                0.0, self.ada_step_size - float(ada["step_adjustment"])
            )
            self.ada_max_probability = float(ada["tightened_max_probability"])
        return overfit

    def _maybe_unfreeze(self, *, validation_available: bool = True) -> None:
        if not self.use_freezed or not self.frozen_resolutions:
            return
        freezed = self.config["freezed"]
        assert isinstance(freezed, Mapping)
        if self.unfreeze_postponements > 0:
            self.unfreeze_postponements -= 1
            return
        window = int(freezed["monitor_window"])
        hit_trigger = False
        if len(self.real_hit_history) >= 2 * window:
            previous = np.asarray(self.real_hit_history[-2 * window : -window])
            current = np.asarray(self.real_hit_history[-window:])
            hit_trigger = bool(
                current.mean() < previous.mean()
                and current.std() <= float(freezed["hit_rate_stability_tolerance"])
            )
        validation_trigger = False
        if validation_available and len(self.validation_history) >= window:
            recent_validation = self.validation_history[-window:]
            improvement = recent_validation[0] - recent_validation[-1]
            validation_trigger = improvement < float(freezed["validation_improvement_epsilon"])
        if hit_trigger or validation_trigger:
            self.frozen_resolutions.pop(0)
            self.discriminator.set_frozen_resolutions(self.frozen_resolutions)

    def _monitor_palette(
        self, palette_kl_value: float, gradient_ratio: float,
        *, prior_too_strong: bool = False,
    ) -> None:
        """Schedule from valid measurements; manual review has first priority."""
        monitoring = self.config["adaptive_monitoring"]
        assert isinstance(monitoring, Mapping)
        self.palette_kl_history.append(palette_kl_value)
        if not math.isfinite(palette_kl_value):
            self.gradient_ratio_history.clear()
            return
        if prior_too_strong:
            self.report_prior_too_strong()
            self.gradient_ratio_history.clear()
            return
        insufficient = palette_kl_value > float(monitoring["palette_kl_absolute_threshold"])
        if len(self.palette_kl_history) >= 2:
            previous = self.palette_kl_history[-2]
            if math.isfinite(previous):
                relative_drop = (previous - palette_kl_value) / max(abs(previous), 1.0e-8)
                insufficient = insufficient or relative_drop < float(
                    monitoring["minimum_relative_improvement"]
                )
        if insufficient:
            self.palette_weight_multiplier *= 1.0 + float(
                monitoring["palette_weight_increase"]
            )
            self.annealing_advance = min(
                1.0,
                self.annealing_advance + float(monitoring["annealing_advance"]),
            )
        if math.isfinite(gradient_ratio):
            self.gradient_ratio_history.append(gradient_ratio)
            consecutive = int(monitoring["gradient_consecutive_measurements"])
            recent_ratios = self.gradient_ratio_history[-consecutive:]
            if len(recent_ratios) == consecutive and all(
                value < float(monitoring["gradient_ratio_low"])
                for value in recent_ratios
            ):
                self.palette_weight_multiplier *= 1.0 + float(
                    monitoring["gradient_weight_adjustment"]
                )
                self.gradient_ratio_history.clear()
            elif len(recent_ratios) == consecutive and all(
                value > float(monitoring["gradient_ratio_reverse_threshold"])
                for value in recent_ratios
            ):
                self.palette_weight_multiplier /= 1.0 + float(
                    monitoring["gradient_weight_adjustment"]
                )
                self.gradient_ratio_history.clear()
        else:
            # Missing support or a zero/invalid adversarial norm breaks a run of
            # consecutive gradient measurements instead of counting as weakness.
            self.gradient_ratio_history.clear()
        self.palette_weight_multiplier = float(
            np.clip(self.palette_weight_multiplier, 0.1, 5.0)
        )

    def report_prior_too_strong(self, flag: bool = True) -> None:
        """Apply the explicit reverse of the insufficient-prior adjustment.

        Visually apparent banding or block breakup is reported through this manual
        trigger; no automatic detector is used.
        """

        if not flag:
            return
        monitoring = self.config["adaptive_monitoring"]
        assert isinstance(monitoring, Mapping)
        self.palette_weight_multiplier /= 1.0 + float(
            monitoring["palette_weight_increase"]
        )
        self.annealing_advance = max(
            0.0,
            self.annealing_advance - float(monitoring["annealing_advance"]),
        )
        self.palette_weight_multiplier = float(
            np.clip(self.palette_weight_multiplier, 0.1, 5.0)
        )

    def _read_prior_feedback(self, palette_kl_value: float) -> Optional[str]:
        """Read an explicit manual report once, only while valid KL decreases.

        A caller may atomically write
        {"request_id": "review-001", "prior_too_strong": true} to the optional
        feedback file. The request stays pending until the KL condition holds.
        """

        if self.prior_feedback_path is None or not self.prior_feedback_path.is_file():
            return None
        try:
            report = json.loads(self.prior_feedback_path.read_text(encoding="utf-8"))
            if not isinstance(report, dict):
                raise ValueError("The feedback must be a JSON object.")
            request_id = report.get("request_id")
            if not isinstance(request_id, str) or not request_id.strip():
                raise ValueError("The feedback requires a nonempty string request_id.")
            if request_id == self.last_prior_feedback_id:
                return None
            if report.get("prior_too_strong") is not True:
                return None
        except (OSError, ValueError) as exc:
            warnings.warn(f"Manual prior feedback was not applied: {exc}", RuntimeWarning)
            return None
        previous = self.palette_kl_history[-1] if self.palette_kl_history else float("nan")
        if (
            math.isfinite(previous) and math.isfinite(palette_kl_value)
            and palette_kl_value < previous
        ):
            return request_id
        return None

    def _process_validation(self, validation: Mapping[str, float], gradient_ratio: float) -> None:
        """Keep FreezeD's metric fixed and apply only supported palette feedback."""

        monitor_value = validation.get(self.validation_metric_name)
        metric_valid = monitor_value is not None and math.isfinite(float(monitor_value))
        if metric_valid:
            self.validation_history.append(float(monitor_value))
        self._maybe_unfreeze(validation_available=metric_valid)
        if self.use_palette:
            palette_value = float(validation.get("palette_kl", float("nan")))
            feedback_id = self._read_prior_feedback(palette_value)
            self._monitor_palette(
                palette_value, gradient_ratio, prior_too_strong=feedback_id is not None
            )
            if feedback_id is not None:
                self.last_prior_feedback_id = feedback_id

    def train_step(self) -> Dict[str, float]:
        """Run one effective-batch update in the fixed D-then-G order."""

        self.generator.train()
        self.discriminator.train()
        d_metrics = self._discriminator_update()
        g_metrics = self._generator_update()
        self._update_ada(d_metrics["real_hit"])
        self.real_hit_history.append(d_metrics["real_hit"])
        self.fake_score_history.append(d_metrics["fake_score"])
        self._monitor_overfitting()
        self.step += 1
        metrics = {
            **d_metrics,
            **g_metrics,
            "ada_probability": self.augment.probability,
            "frozen_stages": float(len(self.frozen_resolutions)),
        }
        self.last_loss_summary = self.loss_logger.record(self.step, metrics)
        metrics["loss_ema_d"] = float(self.loss_logger.loss_ema_d)
        metrics["loss_ema_g"] = float(self.loss_logger.loss_ema_g)
        return metrics

    @torch.no_grad()
    def validate(self) -> Dict[str, float]:
        """Validate the EMA generator with perceptual and craft-specific metrics."""

        if self.val_loader is None:
            return {}
        evaluation = self.config["evaluation"]
        assert isinstance(evaluation, Mapping)
        try:
            from torchmetrics.image.fid import FrechetInceptionDistance
            from torchmetrics.image.kid import KernelInceptionDistance

            fid_metric = FrechetInceptionDistance(normalize=True).to(self.device)
            kid_metric = KernelInceptionDistance(
                subset_size=int(evaluation["kid_subset_size"]), normalize=True
            ).to(self.device)
        except (ImportError, ModuleNotFoundError) as exc:
            warnings.warn(
                f"FID/KID metrics are unavailable during validation: {exc}",
                RuntimeWarning,
                stacklevel=2,
            )
            fid_metric = None
            kid_metric = None

        self.ema_generator.eval()
        palette_counts = torch.zeros_like(self.palette_regulariser.target_proportions)
        palette_total = torch.zeros((), device=self.device)
        flatness_numerator = 0.0
        flatness_denominator = 0.0
        closure_values: List[float] = []
        generated_continuity: List[float] = []
        reference_continuity: List[float] = []
        image_count = 0
        max_batches = int(evaluation["validation_max_batches"])
        for batch_index, batch in enumerate(self.val_loader):
            if max_batches > 0 and batch_index >= max_batches:
                break
            real = batch["image"]
            if not isinstance(real, Tensor):
                continue
            real = real.to(self.device)
            latent = torch.randn(real.shape[0], self.generator.latent_dim, device=self.device)
            fake = self.ema_generator(latent, noise_mode="const")
            assert isinstance(fake, Tensor)
            real_01 = real.add(1.0).mul(0.5).clamp(0.0, 1.0)
            fake_01 = fake.add(1.0).mul(0.5).clamp(0.0, 1.0)
            if fid_metric is not None and kid_metric is not None:
                fid_metric.update(real_01, real=True)
                fid_metric.update(fake_01, real=False)
                kid_metric.update(real_01, real=True)
                kid_metric.update(fake_01, real=False)
            batch_counts, batch_total = palette_soft_counts(
                fake,
                self.palette_regulariser.centroids,
                float(self._palette_schedule()[0]),
                self.palette_regulariser.structural_prior,
            )
            palette_counts += batch_counts
            palette_total += batch_total
            batch_violations, batch_support = flatness_counts(
                fake, self.palette_regulariser.structural_prior,
                float(evaluation["flatness_gradient_threshold"]),
            )
            flatness_numerator += float(batch_violations)
            flatness_denominator += float(batch_support)
            for fake_item, real_item in zip(fake_01, real_01):
                fake_edge = edge_map_from_image(fake_item, self.palette_regulariser.structural_prior)
                real_edge = edge_map_from_image(real_item, self.palette_regulariser.structural_prior)
                closure_values.append(closure_rate(fake_edge))
                generated_continuity.append(line_continuity_index(fake_edge))
                reference_continuity.append(line_continuity_index(real_edge))
            image_count += real.shape[0]

        validation_palette_kl = (
            float(
                palette_kl_from_counts(
                    palette_counts,
                    palette_total,
                    self.palette_regulariser.target_proportions,
                )
            )
            if image_count > 0
            else float("nan")
        )
        results = {
            "palette_kl": validation_palette_kl,
            "flatness": (
                flatness_numerator / flatness_denominator
                if flatness_denominator > 0.0 else float("nan")
            ),
            "palette_valid_pixels": float(palette_total),
            "flatness_valid_pixels": flatness_denominator,
            "closure": float(np.mean(closure_values)) if closure_values else float("nan"),
            "line_continuity_normalised": (
                float(np.mean(generated_continuity)) / max(float(np.mean(reference_continuity)), 1.0e-8)
                if generated_continuity and reference_continuity else float("nan")
            ),
        }
        if fid_metric is not None and kid_metric is not None and image_count >= 2:
            try:
                results["fid"] = float(fid_metric.compute().cpu())
            except RuntimeError as exc:
                warn_metric_failure("FID", exc)
            if image_count >= int(evaluation["kid_subset_size"]):
                try:
                    kid_mean, _ = kid_metric.compute()
                    results["kid"] = float(kid_mean.cpu())
                except RuntimeError as exc:
                    warn_metric_failure("KID", exc)
        self.generator.train()
        self.discriminator.train()
        return results

    def save_checkpoint(self, path: Optional[Path] = None) -> Path:
        """Save all model, optimiser, prior, and adaptive scheduling state."""

        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        destination = path or self.checkpoint_dir / f"step-{self.step:07d}.pt"
        torch.save(
            {
                "step": self.step,
                "config": self.config,
                "generator": self.generator.state_dict(),
                "discriminator": self.discriminator.state_dict(),
                "ema_generator": self.ema_generator.state_dict(),
                "optimiser_g": self.optimiser_g.state_dict(),
                "optimiser_d": self.optimiser_d.state_dict(),
                "palette_centroids": self.palette_regulariser.centroids.detach().cpu(),
                "palette_proportions": self.palette_regulariser.target_proportions.detach().cpu(),
                "mean_path_length": self.mean_path_length.detach().cpu(),
                "ada_probability": self.augment.probability,
                "ada_step_size": self.ada_step_size,
                "ada_max_probability": self.ada_max_probability,
                "frozen_resolutions": self.frozen_resolutions,
                "unfreeze_postponements": self.unfreeze_postponements,
                "validation_metric_name": self.validation_metric_name,
                "last_prior_feedback_id": self.last_prior_feedback_id,
                "palette_weight_multiplier": self.palette_weight_multiplier,
                "annealing_advance": self.annealing_advance,
                "last_processed_monitoring_window": self.last_processed_monitoring_window,
                "loss_logging": self.loss_logger.state_dict(),
                "histories": {
                    "real_hit": self.real_hit_history,
                    "fake_score": self.fake_score_history,
                    "palette_kl": self.palette_kl_history,
                    "validation": self.validation_history,
                    "gradient_ratio": self.gradient_ratio_history,
                },
            },
            destination,
        )
        return destination

    def resume(self, path: Path) -> None:
        """Restore model, optimiser, adaptive-schedule and loss-logging state."""

        checkpoint = torch.load(Path(path), map_location=self.device)
        self.generator.load_state_dict(checkpoint["generator"])
        self.discriminator.load_state_dict(checkpoint["discriminator"])
        self.ema_generator.load_state_dict(checkpoint["ema_generator"])
        self.optimiser_g.load_state_dict(checkpoint["optimiser_g"])
        self.optimiser_d.load_state_dict(checkpoint["optimiser_d"])
        self.step = int(checkpoint["step"])
        self.mean_path_length = checkpoint["mean_path_length"].to(self.device)
        self.augment.probability = float(checkpoint.get("ada_probability", 0.0))
        self.ada_step_size = float(checkpoint.get("ada_step_size", self.ada_step_size))
        self.ada_max_probability = float(
            checkpoint.get("ada_max_probability", self.ada_max_probability)
        )
        self.frozen_resolutions = list(checkpoint.get("frozen_resolutions", []))
        self.unfreeze_postponements = int(checkpoint.get("unfreeze_postponements", 0))
        self.last_prior_feedback_id = checkpoint.get("last_prior_feedback_id")
        self.discriminator.set_frozen_resolutions(self.frozen_resolutions)
        self.palette_weight_multiplier = float(
            checkpoint.get("palette_weight_multiplier", 1.0)
        )
        self.annealing_advance = float(checkpoint.get("annealing_advance", 0.0))
        histories = checkpoint.get("histories", {})
        self.real_hit_history = list(histories.get("real_hit", []))
        self.fake_score_history = list(histories.get("fake_score", []))
        self.palette_kl_history = list(histories.get("palette_kl", []))
        if checkpoint.get("validation_metric_name") == self.validation_metric_name:
            self.validation_history = list(histories.get("validation", []))
        else:
            self.validation_history = []
            if histories.get("validation"):
                warnings.warn(
                    "Validation-trigger history reset because its metric identity changed or was unspecified.",
                    RuntimeWarning,
                )
        self.gradient_ratio_history = list(histories.get("gradient_ratio", []))
        freezed = self.config["freezed"]
        assert isinstance(freezed, Mapping)
        completed_window = len(self.real_hit_history) // int(freezed["monitor_window"])
        self.last_processed_monitoring_window = int(
            checkpoint.get("last_processed_monitoring_window", completed_window)
        )
        logging_state = checkpoint.get("loss_logging")
        if logging_state is None:
            self.loss_logger.reset(completed_step=self.step)
            warnings.warn(
                "Checkpoint has no loss-logging state; loss EMA and interval statistics "
                f"start with the next observed update, step {self.step + 1}.",
                RuntimeWarning,
            )
        else:
            self.loss_logger.load_state_dict(logging_state, completed_step=self.step)
        self.last_loss_summary = None

    def train(self) -> None:
        """Run training for the configured number of main steps."""

        training = self.config["training"]
        assert isinstance(training, Mapping)
        total_steps = int(training["total_steps"])
        progress = tqdm(total=total_steps, initial=self.step, desc="training")
        while self.step < total_steps:
            metrics = self.train_step()
            if self.step % int(training["log_interval"]) == 0:
                printable = {key: round(value, 6) for key, value in metrics.items()}
                print(json.dumps({
                    "step": self.step,
                    "kimg": self.step * self.loss_logger.effective_batch_size / 1000.0,
                    **printable,
                    "loss_interval": self.last_loss_summary,
                }, sort_keys=True))
            if self.step % int(training["validation_interval"]) == 0:
                validation = self.validate()
                if validation:
                    print(json.dumps({"step": self.step, "validation": validation}, sort_keys=True))
                    self._process_validation(
                        validation, metrics.get("gradient_ratio", float("nan"))
                    )
            if self.step % int(training["checkpoint_interval"]) == 0:
                self.save_checkpoint()
            progress.update(1)
        progress.close()
        final_interval = self.loss_logger.flush()
        if final_interval is not None:
            print(json.dumps({"loss_interval": final_interval}, sort_keys=True))
        self.save_checkpoint(self.checkpoint_dir / "final.pt")
