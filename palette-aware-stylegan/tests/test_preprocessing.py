"""Data-range and source-separation regressions without a PyTorch dependency."""

from __future__ import annotations

import json
import importlib.util
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from data.manifest import load_manifest, validate_manifest
from preprocessing.image_arrays import normalise_rgb_array
from preprocessing.preprocess import extract_patches, load_image, source_image_split


class RGBRangeTests(unittest.TestCase):
    def test_uint16_uses_full_scale(self) -> None:
        values = np.array([[[0, 32768, 65535]]], dtype=np.uint16)
        converted = normalise_rgb_array(values)
        self.assertEqual(converted.dtype, np.float32)
        np.testing.assert_allclose(converted[0, 0], [0.0, 32768.0 / 65535.0, 1.0])

    def test_uint8_and_bright_normalised_float_keep_their_ranges(self) -> None:
        np.testing.assert_allclose(
            normalise_rgb_array(np.array([[[0, 128, 255]]], dtype=np.uint8)),
            [[[0.0, 128.0 / 255.0, 1.0]]],
        )
        bright = np.full((2, 2, 3), 0.9, dtype=np.float64)
        np.testing.assert_allclose(normalise_rgb_array(bright), bright)

    def test_float_nonfinite_or_out_of_range_is_rejected(self) -> None:
        for value in (np.nan, np.inf, -0.01, 1.01, 255.0):
            with self.subTest(value=value), self.assertRaises(ValueError):
                normalise_rgb_array(np.full((1, 1, 3), value))
        with self.assertRaises(TypeError):
            normalise_rgb_array(np.ones((1, 1, 3), dtype=np.int16))

    def test_high_bit_depth_pillow_source_is_not_silently_clipped(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "source.tif"
            Image.fromarray(np.array([[0, 32768, 65535]], dtype=np.uint16)).save(path)
            with self.assertRaisesRegex(ValueError, "High-bit-depth"):
                load_image(path)


def sample_manifest() -> dict:
    return {
        "schema_version": 1,
        "sources": [
            {"source_id": "print-a.tif", "split": "train"},
            {"source_id": "print-b.tif", "split": "val"},
        ],
        "patches": [
            {"path": "train/a_r0_c256.png", "source_id": "print-a.tif", "split": "train", "top": 0, "left": 256},
            {"path": "val/b_r512_c0.png", "source_id": "print-b.tif", "split": "val", "top": 512, "left": 0},
        ],
    }


class SourceIdentityTests(unittest.TestCase):
    def test_manifest_round_trip_preserves_source_and_coordinates(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "manifest.json"
            path.write_text(json.dumps(sample_manifest()), encoding="utf-8")
            records = load_manifest(path)
        record = records["train/a_r0_c256.png"]
        self.assertEqual(record["source_id"], "print-a.tif")
        self.assertEqual((record["top"], record["left"]), (0, 256))

    def test_source_list_cannot_assign_one_source_to_two_splits(self) -> None:
        document = sample_manifest()
        document["sources"].append({"source_id": "print-a.tif", "split": "test"})
        with self.assertRaisesRegex(ValueError, "multiple splits"):
            validate_manifest(document)

    def test_patch_cannot_reassign_a_source_to_another_split(self) -> None:
        document = sample_manifest()
        document["patches"][1]["source_id"] = "print-a.tif"
        with self.assertRaisesRegex(ValueError, "multiple splits"):
            validate_manifest(document)

    def test_duplicate_patch_or_parent_path_is_rejected(self) -> None:
        document = sample_manifest()
        document["patches"].append(dict(document["patches"][0]))
        with self.assertRaisesRegex(ValueError, "Duplicate patch"):
            validate_manifest(document)
        document = sample_manifest()
        document["patches"][0]["path"] = "train/../val/a.png"
        with self.assertRaisesRegex(ValueError, "relative path"):
            validate_manifest(document)

    def test_source_split_is_disjoint_and_rejects_repeated_paths(self) -> None:
        sources = [Path(f"source_{index}.png") for index in range(20)]
        splits = source_image_split(sources, seed=42)
        self.assertEqual([len(splits[name]) for name in ("train", "val", "test")], [14, 3, 3])
        self.assertEqual(len(set(sum(splits.values(), []))), 20)
        with self.assertRaisesRegex(ValueError, "duplicate"):
            source_image_split([Path("a.png"), Path("./a.png")])


@unittest.skipUnless(importlib.util.find_spec("skimage"), "scikit-image is required for Lab patch extraction")
class PatchExtractionTests(unittest.TestCase):
    def test_uint16_midgrey_reaches_lab_without_white_clipping(self) -> None:
        image = np.full((4, 4, 3), 32768, dtype=np.uint16)
        records = extract_patches(
            image, "source.tif", patch_size=4, stride=4,
            min_greyscale_std=None, min_mean_saturation=None,
        )
        self.assertEqual(len(records), 1)
        self.assertAlmostEqual(float(records[0].rgb[0, 0, 0]), 32768 / 65535, places=6)
        self.assertGreater(float(records[0].lab[0, 0, 0]), 50.0)
        self.assertLess(float(records[0].lab[0, 0, 0]), 60.0)


@unittest.skipUnless(
    all(importlib.util.find_spec(name) for name in ("skimage", "torch", "torchvision", "yaml")),
    "CLI/dataset integration requires the project dependencies",
)
class PreparedDatasetIntegrationTests(unittest.TestCase):
    def test_cli_manifest_and_dataset_source_contract(self) -> None:
        from data.dataset import WoodblockPatchDataset

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            sources = root / "sources"
            sources.mkdir()
            output = root / "patches"
            y, x = np.indices((64, 64))
            for index in range(20):
                image = np.stack(((x * 4 + index) % 256, y * 4, np.full_like(x, 40)), axis=-1)
                Image.fromarray(image.astype(np.uint8)).save(sources / f"print_{index:02d}.png")
            config = root / "config.yaml"
            config.write_text(
                "patch:\n  size: 32\n  stride: 32\n  source_split: [0.70, 0.15, 0.15]\n"
                "  min_greyscale_std: 0\n  min_mean_saturation: 0\n",
                encoding="utf-8",
            )
            result = subprocess.run(
                [sys.executable, "-m", "preprocessing.prepare", "--source-root", str(sources),
                 "--output", str(output), "--config", str(config)],
                cwd=Path(__file__).resolve().parents[1], text=True, capture_output=True, check=True,
            )
            self.assertIn("train: 14 source images, 56 retained patches", result.stdout)
            self.assertIn("val: 3 source images, 12 retained patches", result.stdout)
            document = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(len(document["patches"]), 80)
            record = document["patches"][0]
            self.assertTrue((output / record["lab_path"]).is_file())
            dataset = WoodblockPatchDataset(output, split="train", resolution=32)
            sample = dataset[0]
            self.assertTrue(str(sample["source_id"]).startswith("print_"))
            self.assertNotEqual(sample["source_id"], sample["patch_id"])
            self.assertEqual(tuple(sample["image"].shape), (3, 32, 32))

            # A val patch cannot claim the identity of a source assigned to train.
            train_source = next(item["source_id"] for item in document["sources"] if item["split"] == "train")
            next(item for item in document["patches"] if item["split"] == "val")["source_id"] = train_source
            (output / "manifest.json").write_text(json.dumps(document), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "multiple splits"):
                WoodblockPatchDataset(output, split="train", resolution=32)

            # A pre-existing directory without a manifest retains a collatable string ID.
            (output / "manifest.json").unlink()
            self.assertEqual(WoodblockPatchDataset(output, "train", 32)[0]["source_id"], "")


if __name__ == "__main__":
    unittest.main()
