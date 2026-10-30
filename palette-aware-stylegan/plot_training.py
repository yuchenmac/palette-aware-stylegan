"""Plot recorded discriminator/generator total losses and their saved scalar EMAs."""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from typing import Dict, List, Sequence, Tuple


def read_loss_records(paths: Sequence[Path]) -> Tuple[List[Dict[str, float]], str]:
    """Read one selected training history; reject overlapping or mixed log types."""

    records: List[Dict[str, float]] = []
    kind = ""
    seen_steps = set()
    effective_batch = None
    for path in paths:
        with path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            fields = set(reader.fieldnames or [])
            current_kind = "summary" if "mean_d_total" in fields else "steps"
            prefix = "mean_" if current_kind == "summary" else ""
            required = {
                "step", "kimg", "loss_history_start_step", "loss_ema_half_life_steps",
                "loss_ema_d", "loss_ema_g", prefix + "d_total", prefix + "g_total",
            }
            if current_kind == "summary":
                required.add("interval_start_step")
            if not required.issubset(fields):
                raise ValueError(f"Missing loss-log columns in {path}.")
            if kind and kind != current_kind:
                raise ValueError("Select either per-step logs or interval summaries, not both.")
            kind = current_kind
            for row in reader:
                item = {name: float(row[name]) for name in required}
                step = item["step"]
                if not step.is_integer() or step <= 0 or item["kimg"] <= 0:
                    raise ValueError(f"Invalid step or kimg in {path}.")
                if step in seen_steps:
                    raise ValueError(
                        f"Overlapping step {int(step)}; select files from one checkpoint continuation."
                    )
                seen_steps.add(step)
                batch = item["kimg"] * 1000.0 / step
                if effective_batch is not None and not math.isclose(batch, effective_batch):
                    raise ValueError("The selected logs use different effective batch sizes.")
                effective_batch = batch
                if any(not math.isfinite(item[name]) for name in (
                    "step", "kimg", "loss_history_start_step", "loss_ema_half_life_steps"
                )) or item["loss_ema_half_life_steps"] <= 0:
                    raise ValueError(f"Invalid loss-log metadata in {path}.")
                # Leave gaps for non-finite loss values.
                for name in required - {
                    "step", "kimg", "loss_history_start_step", "loss_ema_half_life_steps"
                }:
                    if not math.isfinite(item[name]):
                        item[name] = math.nan
                if current_kind == "summary":
                    item["interval_start_step"] = float(row["interval_start_step"])
                records.append(item)
    if not records:
        raise ValueError("The selected loss logs contain no recorded steps.")
    return sorted(records, key=lambda item: item["step"]), kind


def plot_losses(paths: Sequence[Path], output: Path) -> None:
    """Draw saved observations without recomputing or extrapolating loss values."""

    records, kind = read_loss_records(paths)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    prefix = "mean_" if kind == "summary" else ""
    x, d, g, ema_d, ema_g = [], [], [], [], []
    previous = None
    for row in records:
        start = row["step"] if kind == "steps" else row["interval_start_step"]
        if previous is not None and (
            start > previous["step"] + 1
            or row["loss_history_start_step"] != previous["loss_history_start_step"]
        ):
            for values in (x, d, g, ema_d, ema_g):
                values.append(math.nan)
        x.append(row["kimg"])
        d.append(row[prefix + "d_total"])
        g.append(row[prefix + "g_total"])
        ema_d.append(row["loss_ema_d"])
        ema_g.append(row["loss_ema_g"])
        previous = row

    figure, axes = plt.subplots(1, 2, figsize=(10, 3.8), constrained_layout=True)
    observed_label = "Interval mean total loss" if kind == "summary" else "Per-step total loss"
    for axis, observed, smoothed, title in zip(
        axes, (d, g), (ema_d, ema_g), ("Discriminator", "Generator")
    ):
        axis.plot(x, observed, color="#537798", linewidth=0.8, alpha=0.65, label=observed_label)
        axis.plot(x, smoothed, color="#b4522c", linewidth=1.4, label="Saved loss EMA")
        axis.set(title=title, xlabel="Training progress (kimg)", ylabel="Total loss")
        axis.grid(alpha=0.2)
        axis.spines[["right", "top"]].set_visible(False)
        axis.legend(frameon=False, fontsize=8)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=300)
    plt.close(figure)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("logs", nargs="+", type=Path, help="Selected per-step CSVs or summary CSVs.")
    parser.add_argument("--output", type=Path, required=True, help="Output .pdf, .svg or .png file.")
    args = parser.parse_args()
    if args.output.suffix.lower() not in {".pdf", ".svg", ".png"}:
        parser.error("Use a .pdf, .svg or .png output filename.")
    try:
        plot_losses(args.logs, args.output)
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    print(f"Saved training curves to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
