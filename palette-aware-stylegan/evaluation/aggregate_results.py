"""Aggregate independent training-seed evaluations using only the standard library."""

from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence

from evaluation.protocol import (
    METRIC_NAMES, SCHEMA_VERSION, experiment_config, training_seed,
    value_digest, write_result,
)


def aggregate_results(
    results: Sequence[Mapping[str, Any]], *, expected_seeds: int = 3,
) -> Dict[str, Any]:
    """Summarise per-run metric values, with sample SD across training seeds."""
    if expected_seeds < 2:
        raise ValueError("At least two independent seeds are required for sample standard deviation.")
    if len(results) != expected_seeds:
        raise ValueError(f"Expected {expected_seeds} seed results, received {len(results)}.")
    seeds = []
    checkpoints = []
    baseline_config = None
    baseline_protocol = None
    values: Dict[str, list[float]] = {name: [] for name in METRIC_NAMES}
    for index, result in enumerate(results, 1):
        if result.get("schema_version") != SCHEMA_VERSION:
            raise ValueError(f"Result {index}: unsupported or missing schema_version.")
        metadata = result.get("metadata")
        if not isinstance(metadata, Mapping):
            raise ValueError(f"Result {index}: metadata is required.")
        config_record = metadata.get("config")
        if not isinstance(config_record, Mapping) or not isinstance(config_record.get("values"), Mapping):
            raise ValueError(f"Result {index}: the evaluated configuration is required.")
        config = config_record["values"]
        seed = training_seed(config)
        if metadata.get("training_seed") != seed:
            raise ValueError(f"Result {index}: training_seed does not match its configuration.")
        if seed in seeds:
            raise ValueError(f"Duplicate training seed: {seed}.")
        seeds.append(seed)
        if config_record.get("sha256") != value_digest(config):
            raise ValueError(f"Result {index}: configuration digest mismatch.")
        comparison_config = experiment_config(config)
        if baseline_config is None:
            baseline_config = comparison_config
        elif comparison_config != baseline_config:
            raise ValueError("Experimental configurations differ beyond training.seed.")
        protocol = metadata.get("protocol")
        if not isinstance(protocol, Mapping):
            raise ValueError(f"Result {index}: evaluation protocol is required.")
        required_fields = {
            "name", "complete", "required_generated_images", "generated_images",
            "required_reference_images", "reference_images", "reference_images_sha256",
            "evaluation_seed", "palette_k", "palette_temperature", "palette_codebook_sha256",
        }
        if not required_fields.issubset(protocol):
            raise ValueError(f"Result {index}: incomplete evaluation-protocol metadata.")
        if protocol["name"] != "palette-aware-objective-evaluation-v1":
            raise ValueError(f"Result {index}: unsupported evaluation protocol.")
        if protocol.get("complete") is not True:
            raise ValueError("Partial or exploratory evaluations cannot be aggregated as the full protocol.")
        for count_key in ("generated_images", "reference_images"):
            count = protocol[count_key]
            if isinstance(count, bool) or not isinstance(count, int) or count < 2:
                raise ValueError(f"Result {index}: invalid {count_key}.")
        if protocol.get("generated_images") != protocol.get("required_generated_images"):
            raise ValueError(f"Result {index}: incomplete generated-image count.")
        required_reference = protocol.get("required_reference_images")
        if required_reference is not None and protocol.get("reference_images") != required_reference:
            raise ValueError(f"Result {index}: incorrect reference-image count.")
        if metadata.get("evaluation_seed") != protocol.get("evaluation_seed"):
            raise ValueError(f"Result {index}: evaluation-seed metadata mismatch.")
        if result.get("generated_images") != protocol.get("generated_images") or result.get("reference_images") != protocol.get("reference_images"):
            raise ValueError(f"Result {index}: image-count metadata mismatch.")
        if baseline_protocol is None:
            baseline_protocol = dict(protocol)
        elif dict(protocol) != baseline_protocol:
            raise ValueError("Evaluation protocols differ (counts, references, metrics, palette or evaluation seed).")
        checkpoint = metadata.get("checkpoint")
        if not isinstance(checkpoint, Mapping) or not checkpoint.get("sha256"):
            raise ValueError(f"Result {index}: checkpoint identity is required.")
        if checkpoint["sha256"] in checkpoints:
            raise ValueError("The same checkpoint was evaluated more than once.")
        checkpoints.append(checkpoint["sha256"])
        for name in METRIC_NAMES:
            value = result.get(name)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                raise ValueError(f"Result {index}: metric {name} is missing or undefined.")
            values[name].append(float(value))
    return {
        "schema_version": SCHEMA_VERSION,
        "aggregation": "independent_training_seeds",
        "num_seeds": len(seeds),
        "training_seeds": seeds,
        "standard_deviation_ddof": 1,
        "metrics": {
            name: {"mean": statistics.mean(items), "std": statistics.stdev(items), "values": items}
            for name, items in values.items()
        },
        "protocol": baseline_protocol,
        "experiment_config": baseline_config,
        "checkpoint_sha256": checkpoints,
        "kid_dispersion": "The reported KID std is across per-seed kid_mean values; kid_subset_std is not averaged.",
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Aggregate independent training-seed evaluation JSON files.")
    parser.add_argument("results", type=Path, nargs="+")
    parser.add_argument("--expected-seeds", type=int, default=3)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        if args.output.resolve() in {path.resolve() for path in args.results}:
            raise ValueError("The output path must differ from every input result.")
        results = [json.loads(path.read_text(encoding="utf-8")) for path in args.results]
        if not all(isinstance(result, dict) for result in results):
            raise ValueError("Every input file must contain a JSON object.")
        summary = aggregate_results(results, expected_seeds=args.expected_seeds)
        summary["input_files"] = [str(path.resolve()) for path in args.results]
        print(write_result(summary, args.output))
    except (OSError, ValueError, TypeError) as exc:
        parser.error(str(exc))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
