"""Generate configurations for component, palette-term, and palette-size studies."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import yaml


COMPONENT_VARIANTS = {
    "Base": {},
    "A": {"use_palette_aware": False},
    "B": {"use_structural_branch": False},
    "C": {"use_ada": False},
    "D": {"use_freezed": False},
    "E": {"use_ada": False, "use_freezed": False},
}
PALETTE_VARIANTS = {
    "Base": {},
    "P1": {"use_reconstruction": False},
    "P2": {"use_proportion_alignment": False},
    "P3": {"use_entropy": False, "fixed_temperature_when_entropy_disabled": True},
}
BASELINE_FLAGS = (
    "use_palette_aware", "use_structural_branch", "use_ada", "use_freezed",
    "use_reconstruction", "use_proportion_alignment", "use_entropy",
)


def build_experiments(
    config: Mapping[str, Any], seeds: Sequence[int], suite: str = "all"
) -> list[dict[str, Any]]:
    """Copy the baseline, changing the training seed and the named study factor.

    Palette sampling keeps the baseline's ``palette.random_seed`` in every run.
    The supplied seeds identify the new runs to be generated.
    """

    if suite not in {"components", "palette", "k", "all"}:
        raise ValueError(f"Unknown experiment suite: {suite}")
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("Supply at least one training seed, with no duplicates.")
    if any(type(seed) is not int or not 0 <= seed < 2**32 for seed in seeds):
        raise ValueError("Training seeds must be integers between 0 and 2**32 - 1.")
    for section in ("ablation", "training", "palette", "ada", "freezed"):
        if not isinstance(config.get(section), Mapping):
            raise ValueError(f"The baseline configuration requires a {section!r} mapping.")
    disabled = [name for name in BASELINE_FLAGS if config["ablation"].get(name) is not True]
    if disabled:
        raise ValueError("Use a full-model baseline with these flags enabled: " + ", ".join(disabled))
    if config["ada"].get("enabled") is not True or config["freezed"].get("enabled") is not True:
        raise ValueError("The baseline must enable both ada.enabled and freezed.enabled.")
    if "random_seed" not in config["palette"]:
        raise ValueError("The baseline must specify palette.random_seed.")

    variants: dict[str, dict[str, Any]] = {}
    if suite in {"components", "all"}:
        variants.update({name: {"ablation": changes} for name, changes in COMPONENT_VARIANTS.items()})
    if suite in {"palette", "all"}:
        variants.update({name: {"ablation": changes} for name, changes in PALETTE_VARIANTS.items()})
    if suite in {"k", "all"}:
        sizes = config["palette"].get("sensitivity_k", [7, 9, 11])
        if not isinstance(sizes, list) or not sizes or any(type(k) is not int or k < 1 for k in sizes):
            raise ValueError("palette.sensitivity_k must be a non-empty list of positive integers.")
        if len(set(sizes)) != len(sizes):
            raise ValueError("palette.sensitivity_k contains duplicate sizes.")
        variants.update({f"K{k}": {"palette": {"k": k}} for k in sizes})

    experiments = []
    for name, changes in variants.items():
        for seed in seeds:
            run_config = copy.deepcopy(dict(config))
            run_config["training"]["seed"] = seed
            for section, fields in changes.items():
                run_config[section].update(fields)
            experiments.append({
                "run_id": f"{name}-seed-{seed}",
                "variant": name,
                "training_seed": seed,
                "palette_k": run_config["palette"]["k"],
                "config": run_config,
            })
    return experiments


def write_experiments(
    config: Mapping[str, Any], seeds: Sequence[int], suite: str, output_dir: Path
) -> dict[str, Any]:
    """Write a new directory containing YAML configurations and a run manifest."""

    experiments = build_experiments(config, seeds, suite)
    # Refuse an existing directory so that configurations from different runs cannot mix.
    output_dir.mkdir(parents=True, exist_ok=False)
    config_dir = output_dir / "configs"
    config_dir.mkdir()
    runs = []
    for experiment in experiments:
        relative_path = Path("configs") / f"{experiment['run_id']}.yaml"
        with (output_dir / relative_path).open("x", encoding="utf-8") as handle:
            yaml.safe_dump(experiment["config"], handle, sort_keys=False, allow_unicode=True)
        runs.append({
            key: value for key, value in experiment.items() if key != "config"
        } | {"config": relative_path.as_posix()})
    manifest = {
        "schema_version": 1,
        "suite": suite,
        "training_seeds": list(seeds),
        "palette_random_seed": config["palette"]["random_seed"],
        "runs": runs,
    }
    with (output_dir / "manifest.json").open("x", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, allow_nan=False)
        handle.write("\n")
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("config.yaml"))
    parser.add_argument("--output-dir", type=Path, required=True, help="New directory for the generated configurations.")
    parser.add_argument("--seeds", type=int, nargs="+", required=True, help="Distinct training seeds for the new runs.")
    parser.add_argument("--suite", choices=("components", "palette", "k", "all"), default="all")
    args = parser.parse_args()
    try:
        with args.config.open(encoding="utf-8") as handle:
            config = yaml.safe_load(handle)
        if not isinstance(config, Mapping):
            raise ValueError("Configuration root must be a mapping.")
        manifest = write_experiments(config, args.seeds, args.suite, args.output_dir)
    except (OSError, ValueError, yaml.YAMLError) as error:
        parser.error(str(error))
    print(json.dumps({"configurations": len(manifest["runs"]), "manifest": str(args.output_dir / "manifest.json")}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
