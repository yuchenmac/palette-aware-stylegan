"""Craft metrics and dependency-light result aggregation.

Metric implementations are imported on demand so that JSON aggregation does not
require the training or image-processing dependencies.
"""

from importlib import import_module

__all__ = [
    "closure_rate",
    "flatness_violation",
    "line_continuity_index",
    "normalised_line_continuity",
    "palette_kl",
]


def __getattr__(name: str):
    if name not in __all__:
        raise AttributeError(name)
    module = "palette_metrics" if name in {"flatness_violation", "palette_kl"} else "structure_metrics"
    value = getattr(import_module(f"{__name__}.{module}"), name)
    globals()[name] = value
    return value
