"""Streaming loss records and scalar EMAs, independent of model-weight EMA."""

from __future__ import annotations

import csv
import math
import uuid
from pathlib import Path
from typing import Dict, Mapping, Optional


LOSS_FIELDS = (
    "d_total", "g_total", "d_adv", "g_adv", "r1", "palette_weighted", "path_weighted",
)


class LossLogger:
    """Record every completed main step and summarise all observed interval losses.

    Each process/resume segment uses separate CSV files. EMA and partial-interval
    accumulators are checkpointable; each segment records newly observed updates.
    """

    def __init__(
        self,
        directory: Path,
        *,
        log_interval: int,
        effective_batch_size: int,
        ema_half_life_steps: Optional[float] = None,
    ) -> None:
        self.directory = Path(directory)
        self.log_interval = int(log_interval)
        self.effective_batch_size = int(effective_batch_size)
        self.ema_half_life_steps = float(
            self.log_interval if ema_half_life_steps is None else ema_half_life_steps
        )
        if self.log_interval <= 0 or self.effective_batch_size <= 0:
            raise ValueError("Loss logging interval and effective batch size must be positive.")
        if not math.isfinite(self.ema_half_life_steps) or self.ema_half_life_steps <= 0.0:
            raise ValueError("Loss EMA half-life must be positive and finite.")
        self.decay = 2.0 ** (-1.0 / self.ema_half_life_steps)
        self.reset(completed_step=0)

    def reset(self, *, completed_step: int) -> None:
        """Start observation after a completed step, without historical loss values."""

        self.last_step = int(completed_step)
        self.history_start_step = self.last_step + 1
        self.interval_start_step = self.last_step + 1
        self.interval_steps = 0
        self.sums = {name: 0.0 for name in LOSS_FIELDS}
        self.loss_ema_d: Optional[float] = None
        self.loss_ema_g: Optional[float] = None
        self.step_path: Optional[Path] = None
        self.summary_path: Optional[Path] = None

    @property
    def step_fields(self) -> tuple[str, ...]:
        return (
            "step", "kimg", "loss_history_start_step", *LOSS_FIELDS,
            "loss_ema_d", "loss_ema_g", "loss_ema_half_life_steps",
        )

    @property
    def summary_fields(self) -> tuple[str, ...]:
        return (
            "step", "kimg", "loss_history_start_step", "interval_start_step",
            "interval_steps", "partial_interval",
            *(f"mean_{name}" for name in LOSS_FIELDS),
            "loss_ema_d", "loss_ema_g", "loss_ema_half_life_steps",
        )

    def _open_segment(self) -> None:
        if self.step_path is not None:
            return
        self.directory.mkdir(parents=True, exist_ok=True)
        segment = f"from-{self.last_step + 1:07d}-{uuid.uuid4().hex[:12]}"
        self.step_path = self.directory / f"loss-steps-{segment}.csv"
        self.summary_path = self.directory / f"loss-summary-{segment}.csv"
        for path, fields in (
            (self.step_path, self.step_fields), (self.summary_path, self.summary_fields)
        ):
            with path.open("x", newline="", encoding="utf-8") as handle:
                csv.DictWriter(handle, fieldnames=fields).writeheader()

    @staticmethod
    def _append(path: Path, fields: tuple[str, ...], row: Mapping[str, object]) -> None:
        with path.open("a", newline="", encoding="utf-8") as handle:
            csv.DictWriter(handle, fieldnames=fields).writerow(row)

    def record(self, step: int, losses: Mapping[str, float]) -> Optional[Dict[str, object]]:
        """Record one actual update; return a summary at the configured global cadence."""

        step = int(step)
        if step != self.last_step + 1:
            raise ValueError("Loss records must follow consecutive completed main steps.")
        values = {name: float(losses[name]) for name in LOSS_FIELDS}
        self._open_segment()
        self.loss_ema_d = (
            values["d_total"] if self.loss_ema_d is None
            else self.decay * self.loss_ema_d + (1.0 - self.decay) * values["d_total"]
        )
        self.loss_ema_g = (
            values["g_total"] if self.loss_ema_g is None
            else self.decay * self.loss_ema_g + (1.0 - self.decay) * values["g_total"]
        )
        for name, value in values.items():
            self.sums[name] += value
        self.interval_steps += 1
        self.last_step = step
        row = {
            "step": step,
            "kimg": step * self.effective_batch_size / 1000.0,
            "loss_history_start_step": self.history_start_step,
            **values,
            "loss_ema_d": self.loss_ema_d,
            "loss_ema_g": self.loss_ema_g,
            "loss_ema_half_life_steps": self.ema_half_life_steps,
        }
        assert self.step_path is not None
        self._append(self.step_path, self.step_fields, row)
        return self.flush() if step % self.log_interval == 0 else None

    def flush(self) -> Optional[Dict[str, object]]:
        """Write the current interval, including a labelled partial interval at run end."""

        if self.interval_steps == 0:
            return None
        # A restored partial interval is written after new observation begins.
        self._open_segment()
        row: Dict[str, object] = {
            "step": self.last_step,
            "kimg": self.last_step * self.effective_batch_size / 1000.0,
            "loss_history_start_step": self.history_start_step,
            "interval_start_step": self.interval_start_step,
            "interval_steps": self.interval_steps,
            "partial_interval": self.interval_steps != self.log_interval,
            **{f"mean_{name}": value / self.interval_steps for name, value in self.sums.items()},
            "loss_ema_d": self.loss_ema_d,
            "loss_ema_g": self.loss_ema_g,
            "loss_ema_half_life_steps": self.ema_half_life_steps,
        }
        assert self.summary_path is not None
        self._append(self.summary_path, self.summary_fields, row)
        self.sums = {name: 0.0 for name in LOSS_FIELDS}
        self.interval_steps = 0
        self.interval_start_step = self.last_step + 1
        return row

    def state_dict(self) -> Dict[str, object]:
        return {
            "version": 1,
            "log_interval": self.log_interval,
            "effective_batch_size": self.effective_batch_size,
            "ema_half_life_steps": self.ema_half_life_steps,
            "last_step": self.last_step,
            "history_start_step": self.history_start_step,
            "interval_start_step": self.interval_start_step,
            "interval_steps": self.interval_steps,
            "sums": dict(self.sums),
            "loss_ema_d": self.loss_ema_d,
            "loss_ema_g": self.loss_ema_g,
        }

    def load_state_dict(self, state: Mapping[str, object], *, completed_step: int) -> None:
        """Restore EMA and interval state; subsequent rows start a new file segment."""

        if int(state["version"]) != 1:
            raise ValueError("Unsupported loss-logging checkpoint version.")
        for name, expected in (
            ("log_interval", self.log_interval),
            ("effective_batch_size", self.effective_batch_size),
            ("ema_half_life_steps", self.ema_half_life_steps),
        ):
            if float(state[name]) != float(expected):
                raise ValueError(f"Loss-logging checkpoint setting differs: {name}.")
        if int(state["last_step"]) != int(completed_step):
            raise ValueError("Loss-logging state does not match the checkpoint training step.")
        self.reset(completed_step=completed_step)
        self.history_start_step = int(state["history_start_step"])
        self.interval_start_step = int(state["interval_start_step"])
        self.interval_steps = int(state["interval_steps"])
        if self.interval_steps < 0 or self.interval_start_step + self.interval_steps != self.last_step + 1:
            raise ValueError("Invalid partial-interval loss-logging state.")
        sums = state["sums"]
        if not isinstance(sums, Mapping):
            raise ValueError("Loss-logging sums must be a mapping.")
        self.sums = {name: float(sums[name]) for name in LOSS_FIELDS}
        self.loss_ema_d = None if state["loss_ema_d"] is None else float(state["loss_ema_d"])
        self.loss_ema_g = None if state["loss_ema_g"] is None else float(state["loss_ema_g"])
