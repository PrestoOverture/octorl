#!/usr/bin/env python3
"""Run the frozen R3 evaluator and add R3b binary/full-dev diagnostics."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean

def add_r3b_metrics(result: dict) -> dict:
    rollouts = result["rollouts"]
    groups: dict[str, list[dict]] = defaultdict(list)
    for row in rollouts:
        groups[row["task_id"]].append(row)
    if any(len(rows) != result["group_size"] for rows in groups.values()):
        raise AssertionError("dev evaluation does not contain complete G=4 groups")

    fault_rows = [row for row in rollouts if row["fault_type"] != "normal"]
    normal_rows = [row for row in rollouts if row["fault_type"] == "normal"]
    binary_mixed_groups = sum(
        min(row["binary_reward"] for row in rows)
        != max(row["binary_reward"] for row in rows)
        for rows in groups.values()
    )
    fault_groups = {
        task_id: rows
        for task_id, rows in groups.items()
        if rows[0]["fault_type"] != "normal"
    }
    binary_mixed_fault_groups = sum(
        min(row["binary_reward"] for row in rows)
        != max(row["binary_reward"] for row in rows)
        for rows in fault_groups.values()
    )
    result.update(
        {
            "full_dev_binary_pass_rate": mean(row["binary_reward"] for row in rollouts),
            "fault_only_binary_fcr": mean(row["binary_reward"] for row in fault_rows),
            "normal_only_binary_pass_rate": mean(row["binary_reward"] for row in normal_rows),
            "binary_mixed_group_fraction": binary_mixed_groups / len(groups),
            "binary_mixed_fault_group_fraction": (
                binary_mixed_fault_groups / len(fault_groups)
            ),
            "full_dev_instance_count": len(groups),
            "fault_instance_count": len(fault_groups),
            "normal_instance_count": len(groups) - len(fault_groups),
        }
    )
    return result


def main() -> None:
    from r3_eval import evaluate

    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--trajectories", type=Path, required=True)
    parser.add_argument("--model", default="/root/autodl-tmp/models/Qwen3-4B")
    parser.add_argument("--lora-path", type=Path)
    parser.add_argument("--eval-seed", type=int, default=310000)
    args = parser.parse_args()
    result = evaluate(
        manifest_path=args.manifest,
        output_path=args.output,
        trajectories_path=args.trajectories,
        model=args.model,
        lora_path=args.lora_path,
        eval_seed=args.eval_seed,
    )
    add_r3b_metrics(result)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    summary_keys = (
        "rollout_count",
        "continuous_reward_mean",
        "full_dev_binary_pass_rate",
        "fault_only_binary_fcr",
        "binary_mixed_group_fraction",
        "total_wall_time_seconds",
    )
    print(json.dumps({key: result[key] for key in summary_keys}, sort_keys=True))


if __name__ == "__main__":
    main()
