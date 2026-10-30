"""Regression tests for cross-training-seed aggregation; standard library only."""

from __future__ import annotations

import copy
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from evaluation.aggregate_results import aggregate_results
from evaluation.protocol import METRIC_NAMES, value_digest


ROOT = Path(__file__).resolve().parents[1]


def result_for_seed(seed: int, metric: float) -> dict:
    config = {
        "training": {"seed": seed, "total_steps": 200000},
        "palette": {"k": 9, "random_seed": 42},
        "evaluation": {"evaluation_images_per_method": 1000, "reference_images": 2400},
    }
    result = {name: metric for name in METRIC_NAMES}
    result.update({
        "schema_version": 1,
        "kid_subset_std": 9000.0 + seed,
        "generated_images": 1000,
        "reference_images": 2400,
        "metadata": {
            "training_seed": seed,
            "evaluation_seed": 42,
            "checkpoint": {"sha256": f"checkpoint-{seed}"},
            "config": {"values": config, "sha256": value_digest(config)},
            "protocol": {
                "name": "palette-aware-objective-evaluation-v1",
                "complete": True,
                "required_generated_images": 1000,
                "generated_images": 1000,
                "required_reference_images": 2400,
                "reference_images": 2400,
                "reference_images_sha256": "same-reference-images",
                "evaluation_seed": 42,
                "palette_k": 9,
                "palette_temperature": 0.05,
                "palette_codebook_sha256": "same-palette",
            },
        },
    })
    return result


class AggregateResultsTests(unittest.TestCase):
    def setUp(self) -> None:
        self.results = [result_for_seed(seed, value) for seed, value in ((42, 1.0), (43, 2.0), (44, 3.0))]

    def test_means_and_sample_sd_are_across_training_seeds(self) -> None:
        result = aggregate_results(self.results)
        self.assertEqual(result["training_seeds"], [42, 43, 44])
        self.assertEqual(result["standard_deviation_ddof"], 1)
        for metric in METRIC_NAMES:
            self.assertEqual(result["metrics"][metric]["mean"], 2.0)
            self.assertEqual(result["metrics"][metric]["std"], 1.0)
        self.assertNotIn("kid_subset_std", result["metrics"])

    def test_default_requires_three_results(self) -> None:
        with self.assertRaisesRegex(ValueError, "Expected 3"):
            aggregate_results(self.results[:2])
        result = aggregate_results(self.results[:2], expected_seeds=2)
        self.assertEqual(result["num_seeds"], 2)
        self.assertAlmostEqual(result["metrics"]["fid"]["std"], 2**-0.5)
        with self.assertRaises(ValueError):
            aggregate_results(self.results[:1], expected_seeds=1)

    def test_duplicate_seed_is_rejected(self) -> None:
        self.results[2] = copy.deepcopy(self.results[0])
        with self.assertRaisesRegex(ValueError, "Duplicate training seed"):
            aggregate_results(self.results)

    def test_same_checkpoint_with_different_seed_is_rejected(self) -> None:
        self.results[2]["metadata"]["checkpoint"] = copy.deepcopy(self.results[0]["metadata"]["checkpoint"])
        with self.assertRaisesRegex(ValueError, "same checkpoint"):
            aggregate_results(self.results)

    def test_changed_experiment_configuration_is_rejected(self) -> None:
        record = self.results[2]["metadata"]["config"]
        record["values"]["palette"]["random_seed"] = 100
        record["sha256"] = value_digest(record["values"])
        with self.assertRaisesRegex(ValueError, "configurations differ"):
            aggregate_results(self.results)

    def test_changed_protocol_is_rejected(self) -> None:
        for name, value in (("palette_temperature", 0.25), ("reference_images_sha256", "other-reference")):
            results = copy.deepcopy(self.results)
            results[2]["metadata"]["protocol"][name] = value
            with self.assertRaisesRegex(ValueError, "protocols differ"):
                aggregate_results(results)

    def test_partial_evaluation_is_rejected(self) -> None:
        self.results[2]["metadata"]["protocol"]["complete"] = False
        with self.assertRaisesRegex(ValueError, "Partial"):
            aggregate_results(self.results)

    def test_wrong_image_counts_are_rejected(self) -> None:
        self.results[2]["metadata"]["protocol"]["generated_images"] = 16
        with self.assertRaisesRegex(ValueError, "generated-image count"):
            aggregate_results(self.results)

    def test_wrong_reference_counts_are_rejected(self) -> None:
        self.results[2]["metadata"]["protocol"]["reference_images"] = 2399
        with self.assertRaisesRegex(ValueError, "reference-image count"):
            aggregate_results(self.results)

    def test_undefined_metric_is_not_silently_averaged(self) -> None:
        for invalid in (None, float("nan"), float("inf")):
            results = copy.deepcopy(self.results)
            results[1]["palette_kl"] = invalid
            with self.assertRaisesRegex(ValueError, "missing or undefined"):
                aggregate_results(results)

    def test_configuration_and_seed_identity_are_checked(self) -> None:
        results = copy.deepcopy(self.results)
        results[2]["metadata"]["training_seed"] = 100
        with self.assertRaisesRegex(ValueError, "training_seed does not match"):
            aggregate_results(results)
        results = copy.deepcopy(self.results)
        results[2]["metadata"]["config"]["sha256"] = "wrong"
        with self.assertRaisesRegex(ValueError, "digest mismatch"):
            aggregate_results(results)

    def test_cli_writes_summary_without_training_dependencies(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            inputs = []
            for index, result in enumerate(self.results):
                path = root / f"seed-{index}.json"
                path.write_text(json.dumps(result), encoding="utf-8")
                inputs.append(path)
            output = root / "summary.json"
            completed = subprocess.run(
                [sys.executable, "-m", "evaluation.aggregate_results", *map(str, inputs), "--output", str(output)],
                cwd=ROOT, capture_output=True, text=True,
            )
            self.assertEqual(completed.returncode, 0, completed.stderr)
            self.assertEqual(json.loads(output.read_text())["metrics"]["kid_mean"]["std"], 1.0)


if __name__ == "__main__":
    unittest.main()
