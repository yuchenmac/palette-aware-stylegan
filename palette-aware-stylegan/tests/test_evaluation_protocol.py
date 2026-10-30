"""Protocol checks that run without PyTorch or image-processing packages."""

from __future__ import annotations

import copy
import importlib.util
import io
import json
import contextlib
import sys
import tempfile
import unittest
import warnings
from pathlib import Path
from unittest import mock

from evaluation.protocol import (
    check_reference_count, experiment_config, image_set_digest,
    select_evaluation_images, training_seed, value_digest, write_result,
)


class EvaluationProtocolTests(unittest.TestCase):
    def test_formal_selection_is_deterministic_and_exact(self) -> None:
        paths = [Path(f"sample-{index:05d}.png") for index in reversed(range(1005))]
        selected = select_evaluation_images(paths, 1000)
        self.assertEqual(len(selected), 1000)
        self.assertEqual(selected[0], Path("sample-00000.png"))
        self.assertEqual(selected[-1], Path("sample-00999.png"))

    def test_short_formal_evaluation_fails(self) -> None:
        with self.assertRaisesRegex(ValueError, "requires 1000"):
            select_evaluation_images([Path("one.png")], 1000)

    def test_explicit_partial_evaluation_warns(self) -> None:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            self.assertEqual(len(select_evaluation_images([Path("one.png")], 1000, allow_partial=True)), 1)
        self.assertTrue(any("exploratory" in str(item.message) for item in caught))

    def test_reference_count_and_legacy_configuration(self) -> None:
        self.assertTrue(check_reference_count(2400, 2400))
        self.assertTrue(check_reference_count(15, None))
        with self.assertRaisesRegex(ValueError, "requires 2400"):
            check_reference_count(2399, 2400)
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            self.assertFalse(check_reference_count(2399, 2400, allow_partial=True))

    def test_extra_references_also_fail_the_fixed_protocol(self) -> None:
        with self.assertRaises(ValueError):
            check_reference_count(2401, 2400)

    def test_invalid_required_counts_fail(self) -> None:
        with self.assertRaises(ValueError):
            select_evaluation_images([], 0)
        with self.assertRaises(ValueError):
            check_reference_count(1, 0)

    def test_only_training_seed_is_ignored_in_config_comparison(self) -> None:
        base = {"training": {"seed": 42, "steps": 200000}, "palette": {"random_seed": 42}}
        changed = copy.deepcopy(base)
        changed["training"]["seed"] = 43
        self.assertEqual(experiment_config(base), experiment_config(changed))
        changed["palette"]["random_seed"] = 43
        self.assertNotEqual(experiment_config(base), experiment_config(changed))
        self.assertEqual(base["training"]["seed"], 42)

    def test_training_seed_must_be_recorded(self) -> None:
        for config in ({}, {"training": {}}, {"training": {"seed": True}}):
            with self.assertRaises(ValueError):
                training_seed(config)

    def test_reference_identity_uses_content_not_directory_names(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first, second = root / "first.bin", root / "renamed.bin"
            first.write_bytes(b"image bytes")
            second.write_bytes(b"image bytes")
            self.assertEqual(image_set_digest([first]), image_set_digest([second]))
            second.write_bytes(b"different bytes")
            self.assertNotEqual(image_set_digest([first]), image_set_digest([second]))

    def test_json_persists_undefined_metrics_as_null(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "nested" / "result.json"
            text = write_result({"palette_kl": float("nan"), "flatness": 0.1}, output)
            self.assertEqual(json.loads(text), {"palette_kl": None, "flatness": 0.1})
            self.assertEqual(json.loads(output.read_text()), json.loads(text))

    def test_config_digest_does_not_depend_on_key_order(self) -> None:
        self.assertEqual(value_digest({"a": 1, "b": 2}), value_digest({"b": 2, "a": 1}))


@unittest.skipUnless(
    all(importlib.util.find_spec(name) is not None for name in ("torch", "torchmetrics", "skimage")),
    "Image evaluation dependencies are not installed.",
)
class EvaluationIntegrationTests(unittest.TestCase):
    def test_evaluation_seed_reproduces_kid_subset_sampling(self) -> None:
        import torch
        from torchmetrics.image.kid import KernelInceptionDistance
        from evaluation.evaluate import seed_evaluation

        torch.manual_seed(123)
        real, fake = torch.randn(40, 8), torch.randn(40, 8)

        def compute(seed: int):
            metric = KernelInceptionDistance(feature=torch.nn.Identity(), subsets=10, subset_size=8)
            metric.update(real, real=True)
            metric.update(fake, real=False)
            seed_evaluation(seed)
            return metric.compute()

        first, second = compute(42), compute(42)
        self.assertTrue(all(torch.equal(a, b) for a, b in zip(first, second)))
        self.assertFalse(torch.equal(first[0], compute(43)[0]))

    def test_evaluator_metadata_aggregates_three_training_seeds(self) -> None:
        import torch
        import yaml
        from PIL import Image
        from torchmetrics.image.fid import FrechetInceptionDistance as FID
        from torchmetrics.image.kid import KernelInceptionDistance as KID
        from evaluation.evaluate import main
        from evaluation.aggregate_results import aggregate_results

        class TinyFeatures(torch.nn.Module):
            def forward(self, image):
                return image.float().mean(dim=(2, 3))

        # Small deterministic features keep this integration test offline. The
        # production evaluator still uses standard Inception features.
        def fid_factory(**kwargs):
            return FID(feature=TinyFeatures(), normalize=True, input_img_size=(3, 32, 32))

        def kid_factory(**kwargs):
            return KID(feature=TinyFeatures(), subsets=100, subset_size=kwargs["subset_size"], normalize=True)

        root = Path(__file__).resolve().parents[1]
        config = yaml.safe_load((root / "config.yaml").read_text())
        config["image"]["resolution"] = 32
        config["palette"]["k"] = 2
        config["evaluation"].update(evaluation_images_per_method=2, reference_images=2, kid_subset_size=2)
        with tempfile.TemporaryDirectory() as directory:
            work = Path(directory)
            generated, reference = work / "generated", work / "reference"
            generated.mkdir()
            reference.mkdir()
            for index, grey in enumerate((70, 180)):
                Image.new("RGB", (32, 32), (grey, grey, grey)).save(generated / f"{index}.png")
            for index, grey in enumerate((90, 160)):
                Image.new("RGB", (32, 32), (grey, grey, grey)).save(reference / f"{index}.png")
            results = []
            for seed in (42, 43, 44):
                current = copy.deepcopy(config)
                current["training"]["seed"] = seed
                checkpoint = work / f"seed-{seed}.pt"
                torch.save({
                    "step": 200000, "config": current,
                    "palette_centroids": torch.tensor([[50.0, 0.0, 0.0], [70.0, 0.0, 0.0]]),
                    "palette_proportions": torch.tensor([0.5, 0.5]),
                }, checkpoint)
                output = work / f"seed-{seed}.json"
                argv = [
                    "evaluation.evaluate", "--generated", str(generated), "--reference", str(reference),
                    "--checkpoint", str(checkpoint), "--device", "cpu", "--seed", "42", "--output", str(output),
                ]
                with mock.patch.object(sys, "argv", argv), mock.patch("torchmetrics.image.fid.FrechetInceptionDistance", side_effect=fid_factory), mock.patch("torchmetrics.image.kid.KernelInceptionDistance", side_effect=kid_factory), contextlib.redirect_stdout(io.StringIO()):
                    self.assertEqual(main(), 0)
                results.append(json.loads(output.read_text()))
            summary = aggregate_results(results)
            self.assertEqual(summary["training_seeds"], [42, 43, 44])
            self.assertEqual(summary["metrics"]["fid"]["std"], 0.0)
            self.assertEqual(results[0]["metadata"]["protocol"]["palette_temperature"], 0.05)
            self.assertIn("kid_subset_std", results[0])
            self.assertNotIn("kid_std", results[0])


if __name__ == "__main__":
    unittest.main()
