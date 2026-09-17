#!/usr/bin/env python3
"""Compute the pre-registered R2 base-model pilot metrics."""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean
from typing import Any

from src.tasks.tool_recovery.generator import FAMILY_BY_NAME, is_modular


def _rate(values: list[int | float]) -> float | None:
    return sum(values) / len(values) if values else None


def _stats(values: list[int | float]) -> dict[str, int | float | None]:
    return {
        "mean_reward": _rate(values),
        "min_reward": min(values) if values else None,
        "max_reward": max(values) if values else None,
        "rollouts": len(values),
    }


def build_report(results: dict[str, Any]) -> dict[str, Any]:
    rollouts = results["rollouts"]
    reward_mode = results.get("reward_mode", "binary")
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rollouts:
        grouped[row["task_id"]].append(row)
    continuous_rewards = [row.get("continuous_reward", row["reward"]) for row in rollouts]
    fault_rewards = [row.get("binary_reward", row["reward"]) for row in rollouts if row["fault_type"] != "normal"]
    normal_rewards = [row.get("binary_reward", row["reward"]) for row in rollouts if row["fault_type"] == "normal"]
    per_fault = {}
    binary_per_fault = {}
    for fault_type in ("constraint_violation", "missing_dependency", "stale_version"):
        values = [row["reward"] for row in rollouts if row["fault_type"] == fault_type]
        binary_values = [
            row.get("binary_reward", row["reward"])
            for row in rollouts
            if row["fault_type"] == fault_type
        ]
        per_fault[fault_type] = _stats(values)
        binary_per_fault[fault_type] = {"reward_rate": _rate(binary_values), "rollouts": len(binary_values)}
    modular_rewards = [
        row.get("binary_reward", row["reward"])
        for row in rollouts
        if row.get("faulted_field")
        and is_modular(FAMILY_BY_NAME[row["family"]].get_field(row["faulted_field"]))
    ]
    fault_groups = [rows for rows in grouped.values() if rows[0]["fault_type"] != "normal"]
    mixed = sum(
        1
        for rows in fault_groups
        if len(rows) == 2
        and not math.isclose(rows[0]["reward"], rows[1]["reward"], rel_tol=0.0, abs_tol=1e-9)
    )
    terminations = Counter(row["termination_reason"] for row in rollouts)
    flags = Counter(flag for row in rollouts for flag in row.get("diagnostic_flags", []))
    fcr = _rate(fault_rewards)
    mixed_fraction = mixed / len(fault_groups) if fault_groups else None
    gate = bool(fcr is not None and 0.05 <= fcr <= 0.60 and mixed_fraction is not None and mixed_fraction >= 0.15)
    return {
        "model": results["model"],
        "reward_mode": reward_mode,
        "sampling": results.get("sampling"),
        "instance_count": len(grouped),
        "rollout_count": len(rollouts),
        "fcr": fcr,
        "npr": _rate(normal_rewards),
        "binary_fcr": fcr,
        "binary_npr": _rate(normal_rewards),
        "binary_per_fault_type": binary_per_fault,
        "mean_continuous_reward": _rate(continuous_rewards),
        "mean_continuous_fault_reward": _rate(
            [row.get("continuous_reward", row["reward"]) for row in rollouts if row["fault_type"] != "normal"]
        ),
        "mean_continuous_normal_reward": _rate(
            [row.get("continuous_reward", row["reward"]) for row in rollouts if row["fault_type"] == "normal"]
        ),
        "per_fault_type": per_fault,
        "modular_stratum": {"reward_rate": _rate(modular_rewards), "rollouts": len(modular_rewards)},
        "mixed_reward_group_fraction": mixed_fraction,
        "mixed_fault_groups": mixed,
        "fault_group_count": len(fault_groups),
        "tool_call_summary": {
            "mean_calls_per_episode": mean(row["tool_calls"] for row in rollouts) if rollouts else 0.0,
            "termination_reasons": dict(sorted(terminations.items())),
            "diagnostic_flags": dict(sorted(flags.items())),
        },
        "total_wall_time_seconds": results["total_wall_time_seconds"],
        "total_tokens_generated": sum(row["tokens_generated"] for row in rollouts),
        "gate": {
            "fcr_in_5_to_60_percent": bool(fcr is not None and 0.05 <= fcr <= 0.60),
            "mixed_group_at_least_15_percent": bool(mixed_fraction is not None and mixed_fraction >= 0.15),
            "r3_ready": gate,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, default=Path("artifacts/self_improve/pilot/results.json"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/self_improve/pilot/pilot_report.json"))
    args = parser.parse_args()
    report = build_report(json.loads(args.results.read_text(encoding="utf-8")))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
