"""Basic reward matching RQL-Comparison highway_basic / HighwayEnvMEBasic training."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np

DEFAULT_BASIC_REWARD_CONFIG = Path(__file__).parent / "configs" / "HighwayEnv" / "basic_reward.json"


def _lmap(v: float, x: list[float] | tuple[float, float], y: list[float] | tuple[float, float]) -> float:
    """Linear map of value ``v`` from range ``x`` to range ``y``."""
    return y[0] + (v - x[0]) * (y[1] - y[0]) / (x[1] - x[0])


def load_basic_reward_config(path: str | Path | None = None) -> dict[str, Any]:
    """Load basic-reward coefficients used by the trained highway base model."""
    config_path = Path(path) if path is not None else DEFAULT_BASIC_REWARD_CONFIG
    with config_path.open("r") as handle:
        config = json.load(handle)
    return {key: value for key, value in config.items() if not key.startswith("_")}


def compute_basic_reward(
    *,
    forward_speed: float,
    crashed: bool,
    on_road: bool,
    config: Mapping[str, Any],
    lane_fraction: float = 0.0,
) -> float:
    """Compute the training-aligned reward for one timestep.

    Formula (HighwayEnvMEBasic / MergeEnvMEBasic):
        collision_reward * crashed
        + right_lane_reward * lane_fraction
        + high_speed_reward * clip(lmap(speed, reward_speed_range, [0, 1]), 0, 1)
        + base_reward
    Zero when off-road.
    """
    if not on_road:
        return 0.0

    speed_range = config.get("reward_speed_range", [20, 30])
    scaled_speed = _lmap(forward_speed, speed_range, [0, 1])
    return float(
        config.get("collision_reward", -0.5) * float(bool(crashed))
        + config.get("right_lane_reward", 0.0) * float(lane_fraction)
        + config.get("high_speed_reward", 0.4) * float(np.clip(scaled_speed, 0, 1))
        + config.get("base_reward", 1.0)
    )
