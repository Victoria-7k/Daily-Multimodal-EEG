"""Daily-affect EMA-bag classification route."""

from .ema_bags import BRANCHES, EXPERIMENT_BRANCHES, build_daily_affect_bags

__all__ = [
    "BRANCHES",
    "EXPERIMENT_BRANCHES",
    "DailyAffectOrdinalModel",
    "build_daily_affect_bags",
]


def __getattr__(name: str):
    if name == "DailyAffectOrdinalModel":
        from .model import DailyAffectOrdinalModel

        return DailyAffectOrdinalModel
    raise AttributeError(name)
