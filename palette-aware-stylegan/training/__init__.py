"""Training loop for the Palette-aware StyleGAN method."""

__all__ = ["Trainer", "UnifiedAugmentation", "set_random_seed", "LossLogger"]


def __getattr__(name):
    if name == "LossLogger":
        from .loss_logging import LossLogger

        return LossLogger
    if name in {"Trainer", "UnifiedAugmentation", "set_random_seed"}:
        from . import trainer

        return getattr(trainer, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
