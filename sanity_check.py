"""Lightweight synthetic execution check. This file never trains a model."""

from __future__ import annotations

import tempfile
import warnings
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from data.dataset import WoodblockPatchDataset
from evaluation.evaluate import select_evaluation_images
from evaluation.palette_metrics import (
    flatness_violation,
    palette_kl,
    palette_kl_from_counts,
)
from evaluation.structure_metrics import closure_rate, edge_map_from_image, line_continuity_index
from losses.adversarial import (
    discriminator_adversarial_loss,
    generator_adversarial_loss,
    path_length_regularisation,
    r1_regularisation,
)
from losses.palette_aware import PaletteAwareRegularisation
from models.discriminator import DualDiscriminator
from models.generator import Generator
from priors.palette_prior import PalettePrior
from priors.structural_prior import StructuralPrior
from training.trainer import Trainer, set_random_seed, warn_metric_failure


class RecordingStructuralPrior:
    """Record the range received through PalettePrior.fit_rgb_patches."""

    def __init__(self, delegate: StructuralPrior) -> None:
        self.delegate = delegate
        self.inputs: list[torch.Tensor] = []

    def cpu(self) -> "RecordingStructuralPrior":
        self.delegate.cpu()
        return self

    def eval(self) -> "RecordingStructuralPrior":
        self.delegate.eval()
        return self

    def __call__(self, image: torch.Tensor):
        self.inputs.append(image.detach().clone())
        return self.delegate(image)


class DummyFreezer:
    def __init__(self) -> None:
        self.last_resolutions: list[int] = []

    def set_frozen_resolutions(self, resolutions: list[int]) -> None:
        self.last_resolutions = list(resolutions)


def monitoring_stub(monitor_window: int = 2) -> Trainer:
    trainer = Trainer.__new__(Trainer)
    trainer.config = {
        "ada": {
            "high_hit_rate": 0.80,
            "low_hit_rate": 0.45,
            "step_adjustment": 0.005,
            "max_probability": 0.70,
            "tightened_max_probability": 0.60,
        },
        "freezed": {
            "monitor_window": monitor_window,
            "consecutive_windows": 2,
            "hit_rate_stability_tolerance": 0.02,
            "validation_improvement_epsilon": 1.0e-3,
        },
        "adaptive_monitoring": {
            "palette_kl_absolute_threshold": 0.03,
            "minimum_relative_improvement": 0.05,
            "palette_weight_increase": 0.15,
            "annealing_advance": 0.10,
            "gradient_ratio_low": 0.20,
            "gradient_ratio_reverse_threshold": 0.60,
            "gradient_weight_adjustment": 0.10,
            "gradient_consecutive_measurements": 2,
        },
    }
    trainer.use_freezed = True
    trainer.ada_step_size = 0.05
    trainer.ada_max_probability = 0.70
    trainer.unfreeze_postponements = 0
    trainer.last_processed_monitoring_window = -1
    trainer.real_hit_history = []
    trainer.fake_score_history = []
    trainer.palette_kl_history = []
    trainer.validation_history = []
    trainer.gradient_ratio_history = []
    trainer.palette_weight_multiplier = 1.0
    trainer.annealing_advance = 0.0
    trainer.frozen_resolutions = [512, 256]
    trainer.discriminator = DummyFreezer()
    return trainer


def synthetic_flat_colour_batch(size: int = 32) -> torch.Tensor:
    colours = torch.tensor(
        [
            [0.86, 0.10, 0.08], [0.10, 0.55, 0.18], [0.95, 0.75, 0.10],
            [0.05, 0.05, 0.05], [0.92, 0.90, 0.78], [0.12, 0.30, 0.75],
            [0.70, 0.15, 0.55], [0.95, 0.45, 0.08], [0.35, 0.70, 0.65],
        ],
        dtype=torch.float32,
    )
    image = torch.zeros(3, size, size)
    boundaries = torch.linspace(0, size, len(colours) + 1).round().to(torch.int64)
    for index, colour in enumerate(colours):
        image[:, :, boundaries[index] : boundaries[index + 1]] = colour.view(3, 1, 1)
    image[:, 4, 4:-4] = 0.0
    image[:, -5, 4:-4] = 0.0
    image[:, 4:-4, 4] = 0.0
    image[:, 4:-4, -5] = 0.0
    second = torch.flip(image, dims=(2,))
    return torch.stack([image, second], dim=0).mul(2.0).sub(1.0)


def main() -> int:
    print("SANITY CHECK ONLY.")
    print("Synthetic inputs are used only to verify code execution.")
    print("No results reported in the manuscript are reproduced here.")
    set_random_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    palette_source = synthetic_flat_colour_batch(128).to(device)
    real = F.interpolate(palette_source, size=(32, 32), mode="nearest")
    structural = StructuralPrior(
        tau_edge=0.08, gamma=0.02, line_threshold=0.30,
        dilation_radius=2, buffer_radius=3, max_distance=12,
        gaussian_sigma=1.0, sobel_normalisation=8.0,
    ).to(device)
    structural_output = structural(palette_source)
    assert structural_output.line_art.shape == (2, 1, 128, 128)
    assert structural_output.flat_fill_mask.shape == (2, 1, 128, 128)

    palette_builder = PalettePrior(
        k=9, max_samples_per_patch=512, random_seed=42, max_iter=50, tolerance=1.0e-5
    )
    recording_structural = RecordingStructuralPrior(structural.cpu())
    codebook = palette_builder.fit_rgb_patches(
        [item.detach().cpu() for item in palette_source], recording_structural
    )
    assert recording_structural.inputs
    assert torch.allclose(
        recording_structural.inputs[0], palette_source[0:1].detach().cpu(), atol=1.0e-6
    )
    centroids, proportions = codebook.tensors(device)
    structural.to(device)

    generator = Generator(
        latent_dim=64, style_dim=64, mapping_layers=8, resolution=32,
        channel_base=512, channel_max=64,
    ).to(device)
    discriminator = DualDiscriminator(
        resolution=32, channel_base=512, channel_max=64,
        head_channels=64, head_features=64, structural_fusion_weight=0.75,
        tau_edge=0.08, gamma=0.02, gaussian_sigma=1.0,
    ).to(device)
    palette_loss = PaletteAwareRegularisation(
        centroids, proportions, structural.to(device)
    ).to(device)

    latent = torch.randn(2, 64, device=device)
    generated, style = generator(latent, return_latents=True)
    assert generated.shape == (2, 3, 32, 32)
    fake_scores = discriminator(generated)
    assert fake_scores["appearance"].shape == (2,)
    assert fake_scores["structural"].shape == (2,)
    palette_values = palette_loss(generated, temperature=0.25)
    g_adv = generator_adversarial_loss(fake_scores["fused"])
    mean_path = torch.zeros((), device=device)
    path_penalty, _, _ = path_length_regularisation(generated, style, mean_path)
    (g_adv + palette_values["loss"] + 0.5 * path_penalty).backward()

    discriminator.zero_grad(set_to_none=True)
    real_leaf = real.detach().requires_grad_(True)
    real_line = discriminator.lineart_mapping(real_leaf).detach().requires_grad_(True)
    real_scores = discriminator.score_augmented(real_leaf, real_line)
    detached_fake_scores = discriminator.score_augmented(generated.detach())
    d_adv = discriminator_adversarial_loss(
        real_scores["fused"], detached_fake_scores["fused"]
    )
    r1 = r1_regularisation(
        real_scores["appearance"], real_scores["structural"],
        real_leaf, real_line, 10.0, 6.0,
    )
    (d_adv + r1).backward()

    palette_metric = palette_kl(palette_source, centroids, proportions, 0.25, structural)
    flatness_metric = flatness_violation(palette_source, structural, 0.04)
    edge = edge_map_from_image(
        palette_source[0].detach().add(1.0).mul(0.5), structural
    )
    expected_raw_edge = (
        structural_output.line_art[0, 0] > structural.line_threshold
    ).detach().cpu().numpy()
    dilated_edge = structural_output.line_mask[0, 0].detach().cpu().numpy()
    assert np.array_equal(edge, expected_raw_edge)
    assert not np.array_equal(edge, dilated_edge)
    closure_metric = closure_rate(edge)
    continuity_metric = line_continuity_index(edge)

    first_counts = torch.tensor([90.0, 10.0], device=device)
    second_counts = torch.tensor([0.0, 10.0], device=device)
    target_ratio = torch.tensor([0.5, 0.5], device=device)
    global_kl = palette_kl_from_counts(
        first_counts + second_counts,
        torch.tensor(110.0, device=device),
        target_ratio,
    )
    mean_batch_kl = torch.stack(
        [
            palette_kl_from_counts(
                first_counts, torch.tensor(100.0, device=device), target_ratio
            ),
            palette_kl_from_counts(
                second_counts, torch.tensor(10.0, device=device), target_ratio
            ),
        ]
    ).mean()
    assert not torch.isclose(global_kl, mean_batch_kl)

    overfit_trainer = monitoring_stub(monitor_window=2)
    overfit_trainer.real_hit_history = [0.90, 0.91, 0.92, 0.93]
    overfit_trainer.fake_score_history = [1.00, 1.01, 1.02, 1.03]
    assert overfit_trainer._monitor_overfitting()
    adjusted_step_size = overfit_trainer.ada_step_size
    assert overfit_trainer.unfreeze_postponements == 1
    assert not overfit_trainer._monitor_overfitting()
    assert overfit_trainer.ada_step_size == adjusted_step_size
    assert overfit_trainer.unfreeze_postponements == 1

    gradient_trainer = monitoring_stub()
    gradient_trainer._monitor_palette(0.02, 0.10)
    assert np.isclose(gradient_trainer.palette_weight_multiplier, 1.0)
    gradient_trainer._monitor_palette(0.01, 0.10)
    assert np.isclose(gradient_trainer.palette_weight_multiplier, 1.10)

    freezed_trainer = monitoring_stub(monitor_window=3)
    freezed_trainer.validation_history = [1.0, 1.0]
    freezed_trainer._maybe_unfreeze()
    assert freezed_trainer.frozen_resolutions == [512, 256]
    freezed_trainer.validation_history.append(1.0)
    freezed_trainer._maybe_unfreeze()
    assert freezed_trainer.frozen_resolutions == [256]

    synthetic_paths = [Path(f"image-{index:04d}.png") for index in reversed(range(1005))]
    selected_paths = select_evaluation_images(synthetic_paths, 1000)
    assert len(selected_paths) == 1000
    assert selected_paths[0].name == "image-0000.png"
    assert selected_paths[-1].name == "image-0999.png"
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        assert len(select_evaluation_images(synthetic_paths[:3], 1000)) == 3
        assert caught and "uses 1000 images per method" in str(caught[0].message)

    with tempfile.TemporaryDirectory() as temporary:
        data_root = Path(temporary)
        Image.new("RGB", (8, 8), color=(255, 0, 0)).save(data_root / "root-image.png")
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            missing_split = WoodblockPatchDataset(data_root, split="val", resolution=8)
        assert missing_split.root == data_root / "val"
        assert len(missing_split) == 0
        assert caught and "root fallback is disabled" in str(caught[0].message)

    manual_trainer = monitoring_stub()
    manual_trainer.palette_weight_multiplier = 1.15
    manual_trainer.annealing_advance = 0.10
    manual_trainer.report_prior_too_strong()
    assert np.isclose(manual_trainer.palette_weight_multiplier, 1.0)
    assert np.isclose(manual_trainer.annealing_advance, 0.0)
    manual_trainer.report_prior_too_strong(False)
    assert np.isclose(manual_trainer.palette_weight_multiplier, 1.0)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        warn_metric_failure("FID", RuntimeError("synthetic metric failure"))
        warn_metric_failure("KID", RuntimeError("synthetic metric failure"))
    assert len(caught) == 2
    assert "FID computation could not be completed" in str(caught[0].message)
    assert "KID computation could not be completed" in str(caught[1].message)
    values = [
        float(palette_metric), float(flatness_metric),
        float(closure_metric), float(continuity_metric),
    ]
    if not all(np.isfinite(value) for value in values):
        raise RuntimeError("A synthetic metric returned a non-finite value.")
    print("structural prior: OK")
    print("weighted palette prior: OK")
    print("palette input range and raw E(p): OK")
    print("global validation Palette-KL aggregation: OK")
    print("ADA/FreezeD monitoring events: OK")
    print("evaluation image limit and split isolation: OK")
    print("manual prior-too-strong trigger: OK")
    print("FID/KID warning path: OK")
    print("generator/discriminator forward and backward: OK")
    print("Palette-aware and adversarial losses: OK")
    print("custom metric interfaces: OK")
    print("SANITY CHECK PASSED.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
