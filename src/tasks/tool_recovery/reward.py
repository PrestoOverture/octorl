"""Reward functions for tool-recovery final configurations."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any

from .verifier import verify

if TYPE_CHECKING:
    from .generator import ContentFamily


def continuous_reward(
    final_config: dict[str, Any],
    golden_config: dict[str, Any],
    family: "ContentFamily",
) -> float:
    """Score exact fields and give distance-based partial credit to integers."""
    del family  # Kept in the public interface so reward functions are interchangeable.
    if not golden_config:
        return 1.0

    field_scores: list[float] = []
    for name, expected in golden_config.items():
        if name not in final_config:
            field_scores.append(0.0)
            continue
        actual = final_config[name]
        if type(actual) is not type(expected):
            field_scores.append(0.0)
        elif actual == expected:
            field_scores.append(1.0)
        elif type(expected) is int:
            field_scores.append(1.0 / (1.0 + abs(actual - expected)))
        else:
            field_scores.append(0.0)

    reward = math.fsum(field_scores) / len(golden_config)
    return min(1.0, max(0.0, reward))


def binary_reward(
    final_config: dict[str, Any],
    golden_config: dict[str, Any],
    presented_config: dict[str, Any],
    fault_type: str,
    faulted_field: str | None,
    family: "ContentFamily",
    external_state: dict[str, Any],
) -> int:
    """Apply the preserved binary final-state verifier."""
    return verify(
        final_config,
        golden_config,
        presented_config,
        fault_type,
        faulted_field,
        family,
        external_state,
    )


__all__ = ["binary_reward", "continuous_reward"]
