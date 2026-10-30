"""Evaluation protocol checks and JSON metadata using the standard library."""

from __future__ import annotations

import copy
import hashlib
import json
import math
import warnings
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence


SCHEMA_VERSION = 1
METRIC_NAMES = (
    "fid", "kid_mean", "palette_kl", "flatness", "closure_rate",
    "line_continuity_normalised",
)


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def value_digest(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def image_set_digest(paths: Sequence[Path]) -> str:
    """Identify image bytes independently of directory names and traversal order."""
    return value_digest(sorted(file_digest(path) for path in paths))


def experiment_config(config: Mapping[str, Any]) -> Dict[str, Any]:
    """Compare experimental settings while allowing the training seed to vary."""
    result = copy.deepcopy(dict(config))
    training = result.get("training")
    if not isinstance(training, dict):
        raise ValueError("Configuration must contain a training mapping.")
    training.pop("seed", None)
    return result


def training_seed(config: Mapping[str, Any]) -> int:
    value = config.get("training", {}).get("seed")
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError("Configuration must record an integer training.seed.")
    return value


def select_evaluation_images(
    paths: Sequence[Path], required: int, *, allow_partial: bool = False,
) -> List[Path]:
    """Select a deterministic prefix; incomplete official evaluations fail."""
    if required < 1:
        raise ValueError("evaluation_images_per_method must be positive.")
    ordered = sorted(paths, key=lambda path: (path.as_posix().casefold(), path.as_posix()))
    if len(ordered) < required:
        message = f"Found {len(ordered)} generated images; the evaluation protocol requires {required}."
        if not allow_partial:
            raise ValueError(message + " Generate the full set or explicitly use --allow-partial.")
        warnings.warn(message + " Proceeding with an exploratory partial evaluation.", RuntimeWarning, stacklevel=2)
    return ordered[:required]


def check_reference_count(actual: int, required: int | None, *, allow_partial: bool = False) -> bool:
    """Use all reference images, checking the configured study count when present."""
    if required is None:
        return True
    if required < 1:
        raise ValueError("evaluation.reference_images must be positive.")
    if actual == required:
        return True
    message = f"Found {actual} reference images; the evaluation protocol requires {required}."
    if not allow_partial:
        raise ValueError(message + " Select the correct reference directory or use --allow-partial.")
    warnings.warn(message + " Proceeding with an exploratory evaluation.", RuntimeWarning, stacklevel=2)
    return False


def finite_json(value: Any) -> Any:
    """Convert non-finite observations to JSON null."""
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {key: finite_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [finite_json(item) for item in value]
    return value


def write_result(result: Mapping[str, Any], output: Path | None) -> str:
    rendered = json.dumps(finite_json(dict(result)), indent=2, sort_keys=True, allow_nan=False)
    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(rendered + "\n", encoding="utf-8")
    return rendered
