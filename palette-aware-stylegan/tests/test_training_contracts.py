"""CPU regression tests for training behaviour using synthetic inputs."""

from __future__ import annotations

import json
import math
import subprocess
import sys
import tempfile
import unittest
import warnings
from pathlib import Path
from unittest import mock

import numpy as np
import torch
import yaml
from torch import nn
from torch.utils.data import DataLoader
from PIL import Image

from evaluation.palette_metrics import flatness_violation, palette_kl_from_counts
from losses.adversarial import path_length_regularisation, r1_regularisation
from losses.palette_aware import PaletteAwareRegularisation
from models.discriminator import DualDiscriminator
from models.generator import Generator
from preprocessing.preprocess import srgb_to_lab, srgb_to_lab_torch
from priors.palette_prior import PalettePrior
from priors.structural_prior import StructuralPrior, StructuralPriorOutput
from training.trainer import Trainer, UnifiedAugmentation, set_random_seed


ROOT = Path(__file__).resolve().parents[1]
torch.set_num_threads(2)


class FixedSupport(nn.Module):
    """Controlled observation support, independent of image colours."""

    def __init__(self, valid_images: tuple[bool, ...]) -> None:
        super().__init__()
        self.valid_images = valid_images

    def forward(self, image: torch.Tensor) -> StructuralPriorOutput:
        mask = torch.tensor(self.valid_images, device=image.device).view(-1, 1, 1, 1)
        mask = mask.expand(image.shape[0], 1, *image.shape[-2:])
        zero = image.new_zeros(mask.shape)
        return StructuralPriorOutput(zero, ~mask, mask, mask.to(image), zero)


def local_palette(structural: nn.Module) -> PaletteAwareRegularisation:
    centre = srgb_to_lab_torch(torch.full((1, 3, 1, 1), 0.6))[0, :, 0, 0]
    centroids = centre + torch.tensor([[-0.4, 0.1, 0.0], [0.6, 0.0, 0.1]])
    return PaletteAwareRegularisation(centroids, torch.tensor([0.8, 0.2]), structural)


def mini_trainer(directory: Path, *, empty: bool = False) -> Trainer:
    """Construct the actual Trainer with reduced synthetic model dimensions."""

    set_random_seed(42)
    config = yaml.safe_load((ROOT / "config.yaml").read_text())
    config["image"]["resolution"] = 32
    config["generator"].update(
        latent_dim=8, style_dim=8, resolution=32, channel_base=256, channel_max=16
    )
    config["discriminator"].update(
        channel_base=256, channel_max=16, head_channels=16, head_features=16
    )
    config["training"].update(batch_size=2, gradient_accumulation=2, num_workers=0)
    config["freezed"]["initially_frozen_resolutions"] = [32, 16]
    structural = StructuralPrior(buffer_radius=100000 if empty else 3)
    generator = Generator(
        latent_dim=8, style_dim=8, resolution=32, channel_base=256, channel_max=16
    )
    discriminator = DualDiscriminator(
        resolution=32, channel_base=256, channel_max=16,
        head_channels=16, head_features=16,
    )
    samples = [{"image": torch.rand(3, 32, 32) * 2.0 - 1.0} for _ in range(4)]
    loader = DataLoader(samples, batch_size=2)
    return Trainer(
        generator, discriminator, local_palette(structural), loader, loader,
        config, torch.device("cpu"), directory,
    )


class ColourAndSupportTests(unittest.TestCase):
    def test_lab_torch_and_numpy_use_same_original_units(self) -> None:
        rgb = np.random.default_rng(42).uniform(size=(8, 8, 3)).astype(np.float32)
        reference = srgb_to_lab(rgb)
        tensor = torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0).requires_grad_()
        result = srgb_to_lab_torch(tensor)
        np.testing.assert_allclose(
            result.detach()[0].permute(1, 2, 0).numpy(), reference, atol=0.006, rtol=0.0
        )
        result.square().mean().backward()
        self.assertTrue(torch.isfinite(tensor.grad).all())
        self.assertGreater(float(tensor.grad.abs().sum()), 0.0)

    def test_all_bright_patches_restore_explicit_input_range(self) -> None:
        # sRGB .75 becomes dataset value .50, which has no negative samples.
        dataset_patch = torch.full((3, 32, 32), 0.50)
        builder = PalettePrior(k=1, max_samples_per_patch=64, max_iter=3)
        codebook = builder.fit_rgb_patches(
            [dataset_patch], StructuralPrior(), input_range="minus_one_one"
        )
        reference = srgb_to_lab(np.full((1, 1, 3), 0.75, np.float32))[0, 0]
        np.testing.assert_allclose(codebook.centroids[0], reference, atol=0.001)
        zero_one = builder.fit_rgb_patches(
            [np.full((32, 32, 3), 0.75, np.float32)],
            StructuralPrior(), input_range="zero_one",
        )
        np.testing.assert_allclose(zero_one.centroids, codebook.centroids, atol=0.001)

    def test_hard_masks_keep_each_colour_subterm_differentiable(self) -> None:
        loss = local_palette(FixedSupport((True, False)))
        image = torch.full((2, 3, 8, 8), 0.20, requires_grad=True)
        output = loss(image, 0.25)
        self.assertEqual(int(output["valid_images"]), 1)
        self.assertFalse(output["flat_fill_mask"].requires_grad)
        for key in ("reconstruction", "proportion", "entropy"):
            with self.subTest(key=key):
                gradient = torch.autograd.grad(output[key], image, retain_graph=True)[0]
                self.assertTrue(torch.isfinite(gradient).all())
                self.assertGreater(float(gradient[0].abs().sum()), 0.0)
                self.assertEqual(float(gradient[1].abs().sum()), 0.0)
        single = local_palette(FixedSupport((True,)))(image[:1], 0.25)
        torch.testing.assert_close(output["loss"], single["loss"])

    def test_empty_training_support_is_zero_but_metrics_are_undefined(self) -> None:
        structural = FixedSupport((False, False))
        image = torch.full((2, 3, 8, 8), 0.20, requires_grad=True)
        output = local_palette(structural)(image, 0.25)
        self.assertEqual(int(output["valid_images"]), 0)
        self.assertEqual(float(output["loss"]), 0.0)
        output["loss"].backward()
        self.assertTrue(torch.isfinite(image.grad).all())
        self.assertEqual(float(image.grad.abs().sum()), 0.0)
        self.assertTrue(torch.isnan(flatness_violation(image, structural)))
        self.assertTrue(torch.isnan(palette_kl_from_counts(
            torch.zeros(2), torch.tensor(0.0), torch.tensor([0.8, 0.2])
        )))

    def test_soft_assignments_stay_finite_for_large_distances(self) -> None:
        loss = local_palette(FixedSupport((True,)))
        lab = torch.full((1, 3, 2, 2), 100.0, requires_grad=True)
        for temperature in (0.25, 0.05):
            assignments = loss.soft_assign(lab, temperature)
            self.assertTrue(torch.isfinite(assignments).all())
            torch.testing.assert_close(assignments.sum(1), torch.ones(1, 2, 2))


class RegularisationAndAugmentationTests(unittest.TestCase):
    def test_pl_random_projection_detects_cancelling_image_channels(self) -> None:
        style = torch.ones(2, 1, requires_grad=True)
        scalar = style[:, 0].square()
        image = torch.stack((scalar, -scalar), dim=1).view(2, 2, 1, 1)
        projection = torch.zeros_like(image)
        projection[:, 0] = 1.0
        with mock.patch("torch.randn_like", return_value=projection):
            penalty, target, lengths = path_length_regularisation(
                image, style, torch.zeros(())
            )
        self.assertTrue((lengths > 0).all())
        self.assertTrue(torch.isfinite(target))
        penalty.backward()
        self.assertGreater(float(style.grad.abs().sum()), 0.0)

    def test_r1_uses_separate_unfused_branch_gradients(self) -> None:
        rgb = torch.ones(2, 1, 1, 1, requires_grad=True)
        line = torch.full((2, 1, 1, 1), 2.0, requires_grad=True)
        app_score = rgb.square().flatten(1).sum(1)
        edge_score = line.square().flatten(1).sum(1)
        penalty = r1_regularisation(app_score, edge_score, rgb, line, 10.0, 6.0)
        self.assertAlmostEqual(float(penalty), 68.0)
        penalty.backward()
        self.assertTrue(torch.isfinite(rgb.grad).all())
        self.assertTrue(torch.isfinite(line.grad).all())

    def test_augmentation_device_conversion_and_paired_state(self) -> None:
        config = yaml.safe_load((ROOT / "config.yaml").read_text())
        augmentation = UnifiedAugmentation(config["ada"]).to("cpu")
        augmentation.probability = 1.0
        image = torch.rand(2, 3, 16, 16, requires_grad=True)
        first, second = augmentation.forward_pair(image, image)
        torch.testing.assert_close(first, second)
        first.sum().backward()
        self.assertTrue(torch.isfinite(image.grad).all())


class TrainerIntegrationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.directory = Path(self.temporary.name)

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def test_full_discriminator_then_generator_updates_and_lazy_intervals(self) -> None:
        trainer = mini_trainer(self.directory)
        frozen = {
            name: p.detach().clone()
            for name, p in trainer.discriminator.named_parameters() if not p.requires_grad
        }
        generator_before = [p.detach().clone() for p in trainer.generator.parameters()]
        first = trainer.train_step()  # Index zero triggers both lazy penalties.
        second = trainer.train_step()  # Index one triggers neither penalty.
        self.assertEqual(trainer.step, 2)
        for key in ("d_adv", "g_adv", "palette", "r1", "path"):
            self.assertTrue(math.isfinite(first[key]), key)
        self.assertGreater(first["r1"], 0.0)
        self.assertGreater(first["path"], 0.0)
        for metrics in (first, second):
            self.assertAlmostEqual(metrics["d_total"], metrics["d_adv"] + metrics["r1"], places=5)
            self.assertAlmostEqual(
                metrics["g_total"],
                metrics["g_adv"] + metrics["palette_weighted"] + metrics["path_weighted"],
                places=5,
            )
        self.assertEqual(second["r1"], 0.0)
        self.assertEqual(second["path"], 0.0)
        self.assertTrue(any(
            not torch.equal(before, after)
            for before, after in zip(generator_before, trainer.generator.parameters())
        ))
        for name, parameter in trainer.discriminator.named_parameters():
            if name in frozen:
                torch.testing.assert_close(parameter, frozen[name], rtol=0.0, atol=0.0)
        self.assertTrue(all(torch.isfinite(p).all() for p in trainer.generator.parameters()))

    def test_training_cli_completes_one_synthetic_cpu_update(self) -> None:
        config = mini_trainer(self.directory).config
        config["training"].update(
            total_steps=1, log_interval=1, validation_interval=2, checkpoint_interval=1
        )
        config_path = self.directory / "synthetic-config.yaml"
        config_path.write_text(yaml.safe_dump(config))
        train_directory = self.directory / "data" / "train"
        train_directory.mkdir(parents=True)
        coordinate = np.linspace(0.60, 0.90, 32, dtype=np.float32)
        xx, yy = np.meshgrid(coordinate, coordinate)
        rgb = np.stack((xx, yy, (xx + yy) * 0.5), axis=-1)
        for index in range(2):
            Image.fromarray(np.round(rgb * 255).astype(np.uint8)).save(
                train_directory / f"synthetic-{index}.png"
            )
        destination = self.directory / "cli-output"
        completed = subprocess.run(
            [
                sys.executable, "train.py", "--config", str(config_path),
                "--data-root", str(self.directory / "data"),
                "--checkpoint-dir", str(destination), "--device", "cpu",
                "--prior-feedback-file", str(self.directory / "feedback.json"),
            ],
            cwd=ROOT, capture_output=True, text=True, timeout=60,
        )
        self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)
        self.assertTrue((destination / "final.pt").is_file())
        self.assertIn('"step": 1', completed.stdout)

    def test_weighted_gradient_diagnostic_and_empty_support_guard(self) -> None:
        trainer = mini_trainer(self.directory, empty=True)
        trainer.config["training"]["validation_interval"] = 1
        metrics = trainer.train_step()
        self.assertEqual(metrics["palette_valid_images"], 0.0)
        self.assertTrue(math.isnan(metrics["gradient_ratio"]))
        trainer._process_validation({"palette_kl": float("nan")}, metrics["gradient_ratio"])
        self.assertEqual(trainer.palette_weight_multiplier, 1.0)
        # Independently verify that the measured norm includes the outer weight.
        parameter = nn.Parameter(torch.tensor(2.0))
        palette = parameter.square()
        adversarial = parameter * 3.0
        ratio = trainer._gradient_norm(0.25 * palette, [parameter]) / trainer._gradient_norm(
            adversarial, [parameter]
        )
        self.assertAlmostEqual(float(ratio), 1.0 / 3.0, places=6)

    def test_empty_validation_has_no_false_good_or_uniform_metrics(self) -> None:
        trainer = mini_trainer(self.directory, empty=True)
        with mock.patch(
            "torchmetrics.image.fid.FrechetInceptionDistance",
            side_effect=ImportError("Synthetic test disables perceptual-weight downloads"),
        ), warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            values = trainer.validate()
        self.assertTrue(math.isnan(values["palette_kl"]))
        self.assertTrue(math.isnan(values["flatness"]))
        self.assertEqual(values["palette_valid_pixels"], 0.0)
        self.assertEqual(values["flatness_valid_pixels"], 0.0)

    def test_ada_updates_use_positive_rate_and_main_step_interval(self) -> None:
        trainer = mini_trainer(self.directory)
        trainer.augment.probability = 0.20
        trainer._update_ada(0.80)
        self.assertAlmostEqual(trainer.augment.probability, 0.21)
        trainer.step = 1
        trainer._update_ada(0.80)
        self.assertAlmostEqual(trainer.augment.probability, 0.21)
        trainer.step = 4
        trainer._update_ada(0.80)
        self.assertAlmostEqual(trainer.augment.probability, 0.22)

    def test_freezed_or_trigger_and_fixed_validation_identity(self) -> None:
        trainer = mini_trainer(self.directory)
        trainer.validation_history = [5.0] * 5
        trainer._process_validation({"palette_kl": 0.02}, float("nan"))
        self.assertEqual(trainer.frozen_resolutions, [32, 16])
        self.assertEqual(trainer.validation_history, [5.0] * 5)
        trainer._process_validation({"fid": 5.0, "palette_kl": 0.01}, float("nan"))
        self.assertEqual(trainer.frozen_resolutions, [16])  # Validation alone suffices.

    def test_missing_gradient_measurement_breaks_consecutive_trigger(self) -> None:
        trainer = mini_trainer(self.directory)
        for kl, ratio in ((0.02, 0.1), (0.01, float("nan")), (0.005, 0.1)):
            trainer._monitor_palette(kl, ratio)
        self.assertEqual(trainer.palette_weight_multiplier, 1.0)
        trainer._monitor_palette(0.002, 0.1)
        self.assertAlmostEqual(trainer.palette_weight_multiplier, 1.1)

    def test_manual_feedback_is_integrated_prioritised_and_consumed_once(self) -> None:
        trainer = mini_trainer(self.directory)
        trainer.prior_feedback_path = self.directory / "feedback.json"
        trainer.prior_feedback_path.write_text(json.dumps({
            "request_id": "review-001", "prior_too_strong": True
        }))
        trainer.palette_kl_history = [0.02]
        trainer.palette_weight_multiplier = 1.15
        trainer.annealing_advance = 0.1
        trainer.gradient_ratio_history = [0.1]
        trainer._process_validation({"palette_kl": 0.01}, 0.1)
        self.assertAlmostEqual(trainer.palette_weight_multiplier, 1.0)
        self.assertAlmostEqual(trainer.annealing_advance, 0.0)
        self.assertEqual(trainer.last_prior_feedback_id, "review-001")
        trainer._process_validation({"palette_kl": 0.005}, 0.1)
        self.assertAlmostEqual(trainer.palette_weight_multiplier, 1.0)
        trainer.prior_feedback_path.write_text(json.dumps({
            "request_id": "review-002", "prior_too_strong": True
        }))
        trainer._process_validation({"palette_kl": 0.02}, float("nan"))
        self.assertEqual(trainer.last_prior_feedback_id, "review-001")

    def test_pending_freezed_and_feedback_state_round_trip(self) -> None:
        trainer = mini_trainer(self.directory)
        trainer.unfreeze_postponements = 1
        trainer.validation_history = [3.0, 2.0, 1.0]
        trainer.last_prior_feedback_id = "review-001"
        destination = trainer.save_checkpoint()
        restored = mini_trainer(self.directory)
        restored.resume(destination)
        self.assertEqual(restored.unfreeze_postponements, 1)
        self.assertEqual(restored.validation_history, [3.0, 2.0, 1.0])
        self.assertEqual(restored.last_prior_feedback_id, "review-001")
        restored.validation_metric_name = "palette_kl"
        with self.assertWarnsRegex(RuntimeWarning, "metric identity"):
            restored.resume(destination)
        self.assertEqual(restored.validation_history, [])

    def test_loss_logging_continues_from_checkpoint_and_supports_older_checkpoints(self) -> None:
        trainer = mini_trainer(self.directory)
        trainer.train_step()
        destination = trainer.save_checkpoint()
        restored = mini_trainer(self.directory)
        restored.resume(destination)
        self.assertEqual(restored.loss_logger.state_dict(), trainer.loss_logger.state_dict())
        restored.train_step()
        self.assertEqual(restored.loss_logger.last_step, 2)
        self.assertEqual(restored.loss_logger.interval_steps, 2)
        checkpoint = torch.load(destination, map_location="cpu")
        checkpoint.pop("loss_logging")
        older = self.directory / "older.pt"
        torch.save(checkpoint, older)
        restored_old = mini_trainer(self.directory)
        with self.assertWarnsRegex(RuntimeWarning, "no loss-logging state"):
            restored_old.resume(older)
        metrics = restored_old.train_step()
        self.assertEqual(restored_old.loss_logger.history_start_step, 2)
        self.assertEqual(restored_old.loss_logger.interval_steps, 1)
        self.assertEqual(metrics["loss_ema_d"], metrics["d_total"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
