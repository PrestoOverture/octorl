#!/usr/bin/env python3
"""Offline binary-rescore and completion-length analysis for the R3 smoke run."""

from __future__ import annotations

import json
import math
import statistics
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


SOURCE_REL = Path("artifacts/self_improve/r3/raw/seed_42/trajectories.jsonl")
OUTPUT_REL = Path("artifacts/self_improve/r3/results/r3_rescore_report.json")
CANDIDATE_CAPS = (2000, 3000, 4000, 5000, 6000)
EXPECTED_TRAJECTORIES = 80
EXPECTED_GROUPS = 20
EXPECTED_GROUP_SIZE = 4
ZERO_STD_THRESHOLD = 1e-9


def load_trajectories(source_path: Path) -> list[dict[str, Any]]:
    trajectories: list[dict[str, Any]] = []
    with source_path.open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            reward = record.get("binary_reward")
            if reward not in (0, 1):
                raise ValueError(
                    f"line {line_number}: binary_reward must be 0 or 1, got {reward!r}"
                )
            continuous_reward = record.get("continuous_reward")
            if not isinstance(continuous_reward, (int, float)) or not math.isfinite(
                continuous_reward
            ):
                raise ValueError(
                    f"line {line_number}: continuous_reward must be finite, "
                    f"got {continuous_reward!r}"
                )
            if "task_id" not in record:
                raise ValueError(f"line {line_number}: missing task_id")
            if not isinstance(record.get("steps"), list):
                raise ValueError(f"line {line_number}: steps must be a list")
            trajectories.append(record)
    return trajectories


def completion_length(record: dict[str, Any], trajectory_index: int) -> int:
    total = 0
    for step_index, step in enumerate(record["steps"]):
        try:
            observation = step["observation"]
        except (KeyError, TypeError) as exc:
            raise ValueError(
                f"trajectory {trajectory_index}, step {step_index}: "
                "missing observation"
            ) from exc
        if not isinstance(observation, dict):
            raise ValueError(
                f"trajectory {trajectory_index}, step {step_index}: "
                "observation must be an object"
            )
        # Tool observations are structured objects without model-generated text,
        # so they contribute zero to this completion-text proxy.
        text = observation.get("text", "")
        if not isinstance(text, str):
            raise ValueError(
                f"trajectory {trajectory_index}, step {step_index}: "
                "observation.text must be a string"
            )
        total += len(text)
    return total


def nearest_rank(values: list[int], percentile: float) -> int:
    """Return the nearest-rank percentile (rank = ceil(p * N))."""
    if not values:
        raise ValueError("cannot compute a percentile of an empty sequence")
    ordered = sorted(values)
    rank = max(1, math.ceil(percentile * len(ordered)))
    return ordered[rank - 1]


def validate_shape(
    trajectories: list[dict[str, Any]], groups: dict[Any, list[tuple[int, dict[str, Any]]]]
) -> None:
    if len(trajectories) != EXPECTED_TRAJECTORIES:
        raise ValueError(
            f"expected {EXPECTED_TRAJECTORIES} trajectories, got {len(trajectories)}"
        )
    if len(groups) != EXPECTED_GROUPS:
        raise ValueError(f"expected {EXPECTED_GROUPS} groups, got {len(groups)}")
    wrong_sizes = {
        str(task_id): len(members)
        for task_id, members in groups.items()
        if len(members) != EXPECTED_GROUP_SIZE
    }
    if wrong_sizes:
        raise ValueError(
            f"expected every group to have size {EXPECTED_GROUP_SIZE}; got {wrong_sizes}"
        )


def analyze(trajectories: list[dict[str, Any]], source_file: str) -> dict[str, Any]:
    grouped: dict[Any, list[tuple[int, dict[str, Any]]]] = defaultdict(list)
    for trajectory_index, record in enumerate(trajectories):
        grouped[record["task_id"]].append((trajectory_index, record))
    validate_shape(trajectories, grouped)

    group_rows: list[dict[str, Any]] = []
    advantage_rows: list[dict[str, Any]] = []
    continuous_std_rows: list[dict[str, Any]] = []
    group_type_counts = {"all_pass": 0, "mixed": 0, "all_fail": 0}
    zero_group_advantages = 0
    zero_sign_advantages = 0
    zero_continuous_std_groups = 0

    for task_id, members in sorted(grouped.items(), key=lambda item: str(item[0])):
        binary_rewards = [int(record["binary_reward"]) for _, record in members]
        continuous_rewards = [float(record["continuous_reward"]) for _, record in members]
        n_pass = sum(binary_rewards)
        n_fail = len(binary_rewards) - n_pass
        if n_pass == len(binary_rewards):
            group_type = "all_pass"
        elif n_fail == len(binary_rewards):
            group_type = "all_fail"
        else:
            group_type = "mixed"
        group_type_counts[group_type] += 1
        group_rows.append(
            {
                "task_id": task_id,
                "n_pass": n_pass,
                "n_fail": n_fail,
                "group_type": group_type,
            }
        )

        group_mean = statistics.fmean(binary_rewards)
        reward_std = statistics.pstdev(continuous_rewards)
        continuous_std_rows.append({"task_id": task_id, "reward_std": reward_std})
        if reward_std < ZERO_STD_THRESHOLD:
            zero_continuous_std_groups += 1

        for rollout_index, (trajectory_index, record) in enumerate(members):
            binary_reward = int(record["binary_reward"])
            group_advantage = binary_reward - group_mean
            sign_advantage = 2 * binary_reward - 1
            if group_advantage == 0:
                zero_group_advantages += 1
            if sign_advantage == 0:
                zero_sign_advantages += 1
            advantage_rows.append(
                {
                    "trajectory_index": trajectory_index,
                    "task_id": task_id,
                    "rollout_index_within_group": rollout_index,
                    "binary_reward": binary_reward,
                    "A_group": group_advantage,
                    "A_sign": sign_advantage,
                }
            )

    trajectory_count = len(trajectories)
    group_count = len(grouped)
    pass_count = sum(int(record["binary_reward"]) for record in trajectories)
    fractions = {
        f"{group_type}_frac": count / group_count
        for group_type, count in group_type_counts.items()
    }
    if not math.isclose(sum(fractions.values()), 1.0, rel_tol=0.0, abs_tol=1e-12):
        raise AssertionError("group-type fractions do not sum to 1.0")

    length_rows: list[dict[str, Any]] = []
    lengths: list[int] = []
    for trajectory_index, record in enumerate(trajectories):
        length = completion_length(record, trajectory_index)
        lengths.append(length)
        length_rows.append(
            {
                "trajectory_index": trajectory_index,
                "task_id": record["task_id"],
                "binary_reward": int(record["binary_reward"]),
                "completion_length_characters": length,
            }
        )

    total_character_mass = sum(lengths)
    cap_rows: list[dict[str, Any]] = []
    for cap in CANDIDATE_CAPS:
        truncated = [
            (record, length)
            for record, length in zip(trajectories, lengths)
            if length > cap
        ]
        truncated_count = len(truncated)
        truncated_pass_count = sum(
            int(record["binary_reward"]) for record, _ in truncated
        )
        truncated_fail_count = truncated_count - truncated_pass_count
        truncated_tail_characters = sum(length - cap for _, length in truncated)
        cap_rows.append(
            {
                "cap_characters": cap,
                "truncated_count": truncated_count,
                "fraction_trajectories_truncated": truncated_count / trajectory_count,
                "truncated_tail_characters": truncated_tail_characters,
                "fraction_total_character_mass_in_truncated_tails": (
                    truncated_tail_characters / total_character_mass
                ),
                "among_truncated": {
                    "pass_count": truncated_pass_count,
                    "fail_count": truncated_fail_count,
                    "pass_frac": (
                        truncated_pass_count / truncated_count
                        if truncated_count
                        else 0.0
                    ),
                    "fail_frac": (
                        truncated_fail_count / truncated_count
                        if truncated_count
                        else 0.0
                    ),
                },
                "truncation_rate_by_binary_reward": {
                    "pass": truncated_pass_count / pass_count if pass_count else 0.0,
                    "fail": (
                        truncated_fail_count / (trajectory_count - pass_count)
                        if trajectory_count != pass_count
                        else 0.0
                    ),
                },
            }
        )

    return {
        "group_composition": {
            "groups": group_rows,
            "aggregate": {
                **{f"{key}_count": value for key, value in group_type_counts.items()},
                **fractions,
            },
            "overall_binary_pass_count": pass_count,
            "overall_binary_fail_count": trajectory_count - pass_count,
            "overall_binary_pass_rate": pass_count / trajectory_count,
        },
        "advantage_comparison": {
            "definitions": {
                "A_group": "binary_reward - mean(group_binary_rewards)",
                "A_sign": "2 * binary_reward - 1",
                "continuous_reward_std": "population standard deviation (pstdev)",
                "zero_std_threshold": ZERO_STD_THRESHOLD,
            },
            "trajectories": sorted(
                advantage_rows, key=lambda row: row["trajectory_index"]
            ),
            "frac_zero_advantage_group": zero_group_advantages / trajectory_count,
            "frac_zero_advantage_sign": zero_sign_advantages / trajectory_count,
            "continuous_reward_std_by_group": continuous_std_rows,
            "frac_reward_zero_std": zero_continuous_std_groups / group_count,
        },
        "completion_lengths": {
            "unit": "characters",
            "proxy_definition": (
                "sum(len(step['observation'].get('text', '')) for step in steps); "
                "character count is used as a completion-token proxy and "
                "structured tool observations without text contribute zero"
            ),
            "percentile_method": "nearest-rank (rank = ceil(p * N))",
            "percentiles": {
                "p50": nearest_rank(lengths, 0.50),
                "p90": nearest_rank(lengths, 0.90),
                "p95": nearest_rank(lengths, 0.95),
                "p99": nearest_rank(lengths, 0.99),
                "max": max(lengths),
            },
            "total_character_mass": total_character_mass,
            "trajectories": length_rows,
            "candidate_caps": cap_rows,
        },
        "meta": {
            "trajectory_count": trajectory_count,
            "group_count": group_count,
            "group_size": EXPECTED_GROUP_SIZE,
            "source_file": source_file,
            "analysis_timestamp": datetime.now(timezone.utc)
            .isoformat(timespec="seconds")
            .replace("+00:00", "Z"),
        },
    }


def print_summary(report: dict[str, Any], output_path: Path) -> None:
    composition = report["group_composition"]
    aggregate = composition["aggregate"]
    advantages = report["advantage_comparison"]
    completions = report["completion_lengths"]

    print("R3 offline rescore diagnostic")
    print(
        f"Input: {report['meta']['trajectory_count']} trajectories, "
        f"{report['meta']['group_count']} task_id groups, "
        f"G={report['meta']['group_size']}"
    )
    print("\nBinary group composition")
    for group in composition["groups"]:
        print(
            f"  {group['task_id']}: pass={group['n_pass']} "
            f"fail={group['n_fail']} type={group['group_type']}"
        )
    print(
        "  Aggregate: "
        f"all_pass={aggregate['all_pass_count']} "
        f"({aggregate['all_pass_frac']:.1%}), "
        f"mixed={aggregate['mixed_count']} ({aggregate['mixed_frac']:.1%}), "
        f"all_fail={aggregate['all_fail_count']} "
        f"({aggregate['all_fail_frac']:.1%})"
    )
    print(
        f"  Overall pass rate: {composition['overall_binary_pass_count']}/"
        f"{report['meta']['trajectory_count']} "
        f"({composition['overall_binary_pass_rate']:.1%})"
    )

    print("\nAdvantage comparison")
    print(
        "  Zero group-centered binary advantages: "
        f"{advantages['frac_zero_advantage_group']:.1%}"
    )
    print(
        "  Zero fixed-sign binary advantages: "
        f"{advantages['frac_zero_advantage_sign']:.1%}"
    )
    print(
        "  Continuous-reward zero-std groups: "
        f"{advantages['frac_reward_zero_std']:.1%}"
    )

    percentiles = completions["percentiles"]
    print("\nCompletion lengths (character-count proxy)")
    print(
        f"  p50={percentiles['p50']}, p90={percentiles['p90']}, "
        f"p95={percentiles['p95']}, p99={percentiles['p99']}, "
        f"max={percentiles['max']}"
    )
    for cap in completions["candidate_caps"]:
        cross_tab = cap["among_truncated"]
        rates = cap["truncation_rate_by_binary_reward"]
        print(
            f"  cap={cap['cap_characters']}: "
            f"truncated={cap['fraction_trajectories_truncated']:.1%}, "
            f"tail_mass={cap['fraction_total_character_mass_in_truncated_tails']:.1%}, "
            f"truncated_pass/fail={cross_tab['pass_frac']:.1%}/"
            f"{cross_tab['fail_frac']:.1%}, "
            f"pass/fail_truncation_rate={rates['pass']:.1%}/{rates['fail']:.1%}"
        )
    print(f"\nReport written to {output_path}")


def main() -> None:
    repo_root = Path(__file__).resolve().parents[4]
    source_path = repo_root / SOURCE_REL
    output_path = repo_root / OUTPUT_REL
    trajectories = load_trajectories(source_path)
    report = analyze(trajectories, SOURCE_REL.as_posix())
    with output_path.open("w", encoding="utf-8") as output:
        json.dump(report, output, indent=2, ensure_ascii=False)
        output.write("\n")
    print_summary(report, output_path)


if __name__ == "__main__":
    main()
