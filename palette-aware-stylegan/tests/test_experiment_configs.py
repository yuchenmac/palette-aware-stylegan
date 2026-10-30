"""Regression tests for study-factor changes and configuration provenance."""

from __future__ import annotations

import copy
import json
import tempfile
import unittest
from pathlib import Path

import yaml

from prepare_experiments import build_experiments, write_experiments


ROOT = Path(__file__).resolve().parents[1]


class ExperimentConfigurationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8"))

    def test_component_variants_change_only_named_flags_and_training_seed(self) -> None:
        expected = {
            "Base": {}, "A": {"use_palette_aware": False},
            "B": {"use_structural_branch": False}, "C": {"use_ada": False},
            "D": {"use_freezed": False}, "E": {"use_ada": False, "use_freezed": False},
        }
        original = copy.deepcopy(self.config)
        runs = build_experiments(self.config, [13, 14], "components")
        self.assertEqual(len(runs), 12)
        for run in runs:
            reference = copy.deepcopy(original)
            reference["training"]["seed"] = run["training_seed"]
            reference["ablation"].update(expected[run["variant"]])
            self.assertEqual(run["config"], reference)
        self.assertEqual(self.config, original)

    def test_palette_ablation_and_p3_fixed_temperature(self) -> None:
        self.config["ablation"]["fixed_temperature_when_entropy_disabled"] = False
        runs = {run["variant"]: run["config"] for run in build_experiments(self.config, [13], "palette")}
        self.assertEqual(set(runs), {"Base", "P1", "P2", "P3"})
        self.assertFalse(runs["P1"]["ablation"]["use_reconstruction"])
        self.assertFalse(runs["P2"]["ablation"]["use_proportion_alignment"])
        self.assertFalse(runs["P3"]["ablation"]["use_entropy"])
        self.assertTrue(runs["P3"]["ablation"]["fixed_temperature_when_entropy_disabled"])
        self.assertEqual(runs["P3"]["palette_loss"], self.config["palette_loss"])

    def test_k_sweep_uses_configured_sizes_and_fixed_palette_seed(self) -> None:
        self.config["palette"]["sensitivity_k"] = [5, 9]
        self.config["palette"]["random_seed"] = 123
        runs = build_experiments(self.config, [13, 14], "k")
        self.assertEqual({run["variant"] for run in runs}, {"K5", "K9"})
        self.assertEqual({run["palette_k"] for run in runs}, {5, 9})
        self.assertTrue(all(run["config"]["palette"]["random_seed"] == 123 for run in runs))
        self.assertTrue(all(run["config"]["ablation"] == self.config["ablation"] for run in runs))

    def test_all_suites_include_base_once_per_seed(self) -> None:
        runs = build_experiments(self.config, [13, 14, 15], "all")
        self.assertEqual(len(runs), 36)
        self.assertEqual(sum(run["variant"] == "Base" for run in runs), 3)
        self.assertEqual(len({run["run_id"] for run in runs}), 36)

    def test_rejects_invalid_or_duplicate_seeds(self) -> None:
        for seeds in ([], [13, 13], [-1], [2**32], [True], [1.5]):
            with self.subTest(seeds=seeds), self.assertRaises(ValueError):
                build_experiments(self.config, seeds, "all")

    def test_rejects_nonbaseline_instead_of_silently_resetting_flags(self) -> None:
        for section, name in (("ablation", "use_palette_aware"), ("ada", "enabled"), ("freezed", "enabled")):
            config = copy.deepcopy(self.config)
            config[section][name] = False
            with self.subTest(section=section, name=name), self.assertRaises(ValueError):
                build_experiments(config, [13], "components")

    def test_manifest_points_to_exact_saved_configurations_and_refuses_overwrite(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output_dir = Path(temporary) / "study"
            manifest = write_experiments(self.config, [13, 14, 15], "palette", output_dir)
            self.assertEqual(json.loads((output_dir / "manifest.json").read_text()), manifest)
            for run in manifest["runs"]:
                saved = yaml.safe_load((output_dir / run["config"]).read_text())
                self.assertEqual(saved["training"]["seed"], run["training_seed"])
                self.assertEqual(saved["palette"]["random_seed"], self.config["palette"]["random_seed"])
            before = (output_dir / "manifest.json").read_bytes()
            with self.assertRaises(FileExistsError):
                write_experiments(self.config, [16], "k", output_dir)
            self.assertEqual((output_dir / "manifest.json").read_bytes(), before)


if __name__ == "__main__":
    unittest.main()
