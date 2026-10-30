"""Prepare source-separated RGB/Lab patches and a source-coordinate manifest."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path
from typing import Dict, Optional, Sequence

import numpy as np
from PIL import Image

from data.manifest import validate_manifest
from preprocessing.preprocess import extract_patches, load_image, source_image_split


EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff"}


def prepare_dataset(
    source_root: Path,
    output_root: Path,
    *,
    patch_size: int = 512,
    stride: int = 256,
    ratios: Sequence[float] = (0.70, 0.15, 0.15),
    seed: int = 42,
    min_greyscale_std: Optional[float] = 0.02,
    min_mean_saturation: Optional[float] = 0.05,
) -> Dict[str, Dict[str, int]]:
    """Split already standardised source images, extract patches, and save metadata.

    Inputs must already have acquisition calibration, sRGB conversion, orientation
    and central-panel/content cropping applied. This operation does not infer or
    apply ICC profiles, white-balance matrices, or content boxes. Pillow supplies
    8-bit RGB working copies; each retained patch also has a float32 CIELAB file.
    """

    source_root = Path(source_root).expanduser().resolve()
    output_root = Path(output_root).expanduser().resolve()
    if not source_root.is_dir():
        raise ValueError("The standardised source-image directory does not exist.")
    if source_root == output_root or source_root in output_root.parents:
        raise ValueError("The output directory must be outside the source-image directory.")
    if output_root.exists() and (not output_root.is_dir() or any(output_root.iterdir())):
        raise ValueError("The output must be a new or empty directory; existing datasets are not overwritten.")
    if patch_size < 1 or stride < 1:
        raise ValueError("Patch size and stride must be positive.")
    for name, value in (
        ("min_greyscale_std", min_greyscale_std),
        ("min_mean_saturation", min_mean_saturation),
    ):
        if value is not None and (not np.isfinite(value) or value < 0.0 or value > 1.0):
            raise ValueError(f"{name} must be in [0, 1] or None.")

    sources = sorted(
        path for path in source_root.rglob("*")
        if path.is_file() and path.suffix.lower() in EXTENSIONS
    )
    if not sources:
        raise ValueError("No supported source images were found.")
    splits = source_image_split(sources, ratios=ratios, seed=seed)
    source_numbers = {path: index for index, path in enumerate(sources)}
    statistics = {
        split: {"sources": len(paths), "patches": 0} for split, paths in splits.items()
    }
    document: Dict[str, object] = {
        "schema_version": 1,
        "preprocessing": {
            "input_contract": "Previously calibrated and content-cropped sRGB source images",
            "rgb_working_dtype": "uint8",
            "lab_dtype": "float32",
            "lab_illuminant": "D65",
            "patch_size": patch_size,
            "stride": stride,
            "source_split": list(ratios),
            "seed": seed,
            "min_greyscale_std": min_greyscale_std,
            "min_mean_saturation": min_mean_saturation,
        },
        "sources": [],
        "patches": [],
    }
    source_records = document["sources"]
    patch_records = document["patches"]
    assert isinstance(source_records, list) and isinstance(patch_records, list)
    output_root.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".prepare-patches-", dir=output_root.parent) as temporary:
        stage = Path(temporary)
        for split, paths in splits.items():
            (stage / split).mkdir()
            (stage / "lab" / split).mkdir(parents=True)
            for path in paths:
                source_id = path.relative_to(source_root).as_posix()
                image = load_image(path)
                records = extract_patches(
                    image, source_id, patch_size=patch_size, stride=stride,
                    min_greyscale_std=min_greyscale_std,
                    min_mean_saturation=min_mean_saturation,
                )
                source_records.append({
                    "source_id": source_id, "path": source_id, "split": split,
                    "width": image.width, "height": image.height, "patch_count": len(records),
                })
                for record in records:
                    stem = f"source{source_numbers[path]:06d}_r{record.top}_c{record.left}"
                    rgb_path = f"{split}/{stem}.png"
                    lab_path = f"lab/{split}/{stem}.npy"
                    rgb8 = np.round(record.rgb * 255.0).astype(np.uint8)
                    Image.fromarray(rgb8).save(stage / rgb_path)
                    np.save(stage / lab_path, record.lab, allow_pickle=False)
                    patch_records.append({
                        "path": rgb_path, "lab_path": lab_path, "split": split,
                        "source_id": source_id, "top": record.top, "left": record.left,
                        "width": patch_size, "height": patch_size,
                        "greyscale_std": record.greyscale_std,
                        "mean_saturation": record.mean_saturation,
                    })
                statistics[split]["patches"] += len(records)
        validate_manifest(document)
        with (stage / "manifest.json").open("w", encoding="utf-8") as handle:
            json.dump(document, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
        if output_root.exists():
            output_root.rmdir()
        os.replace(stage, output_root)
    return statistics


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Prepare patches from already calibrated, content-cropped sRGB source images."
    )
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--config", type=Path, help="Optional YAML config; reads its patch settings.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--patch-size", type=int)
    parser.add_argument("--stride", type=int)
    parser.add_argument("--source-split", type=float, nargs=3, metavar=("TRAIN", "VAL", "TEST"))
    parser.add_argument("--min-greyscale-std", type=float)
    parser.add_argument("--min-mean-saturation", type=float)
    args = parser.parse_args()
    patch = {}
    try:
        if args.config is not None:
            import yaml

            with args.config.open("r", encoding="utf-8") as handle:
                config = yaml.safe_load(handle)
            if not isinstance(config, dict) or not isinstance(config.get("patch", {}), dict):
                raise ValueError("The YAML configuration must contain a patch mapping.")
            patch = config.get("patch", {})
        statistics = prepare_dataset(
            args.source_root, args.output,
            patch_size=args.patch_size if args.patch_size is not None else int(patch.get("size", 512)),
            stride=args.stride if args.stride is not None else int(patch.get("stride", 256)),
            ratios=args.source_split if args.source_split is not None else patch.get("source_split", (0.70, 0.15, 0.15)),
            seed=args.seed,
            min_greyscale_std=args.min_greyscale_std if args.min_greyscale_std is not None else patch.get("min_greyscale_std", 0.02),
            min_mean_saturation=args.min_mean_saturation if args.min_mean_saturation is not None else patch.get("min_mean_saturation", 0.05),
        )
    except (OSError, ValueError, TypeError, ImportError) as error:
        parser.exit(2, f"Data preparation failed: {error}\n")
    for split, counts in statistics.items():
        print(f"{split}: {counts['sources']} source images, {counts['patches']} retained patches")
    print(f"Prepared dataset: {args.output.resolve()}")
    print("Source identities and coordinates: manifest.json; CIELAB copies: lab/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
