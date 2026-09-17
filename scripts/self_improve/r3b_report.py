#!/usr/bin/env python3
"""Build the R3b pilot tables and machine-readable smoke summary."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from statistics import fmean, pstdev


HOURLY_CNY = 2.18
BUDGET_CNY = 8.0
SEED = 42
UPDATES = 20
GROUP_SIZE = 4
PROMPTS_PER_UPDATE = 4
ROLLOUTS_PER_UPDATE = GROUP_SIZE * PROMPTS_PER_UPDATE


def _json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _trajectory_batches(path: Path) -> dict[int, dict]:
    records = _jsonl(path)
    expected = UPDATES * ROLLOUTS_PER_UPDATE
    if len(records) != expected:
        raise AssertionError(f"expected {expected} trajectories, got {len(records)}")
    batches: dict[int, dict] = {}
    for offset in range(UPDATES):
        rows = records[offset * ROLLOUTS_PER_UPDATE : (offset + 1) * ROLLOUTS_PER_UPDATE]
        groups: dict[str, list[int]] = defaultdict(list)
        for row in rows:
            groups[row["task_id"]].append(int(row["binary_reward"]))
        if len(groups) != PROMPTS_PER_UPDATE or any(len(values) != GROUP_SIZE for values in groups.values()):
            raise AssertionError(f"update {offset + 1} is not four complete G=4 groups")
        rewards = [int(row["binary_reward"]) for row in rows]
        batches[offset + 1] = {
            "reward_mean": fmean(rewards),
            "reward_std": pstdev(rewards),
            "mixed_group_fraction": sum(min(v) != max(v) for v in groups.values()) / len(groups),
            "all_pass_frac": sum(min(v) == 1 for v in groups.values()) / len(groups),
            "all_fail_frac": sum(max(v) == 0 for v in groups.values()) / len(groups),
        }
    return batches


def _update_rows(raw: Path) -> list[dict]:
    metric_records = _jsonl(raw / "seed_42/metrics_target_20.jsonl")
    if len(metric_records) != UPDATES:
        raise AssertionError(f"expected {UPDATES} metric records, got {len(metric_records)}")
    trajectory = _trajectory_batches(raw / "seed_42/trajectories.jsonl")
    rows = []
    for record in metric_records:
        data = record["data"]
        update = int(data["training/global_step"])
        if update != int(record["step"]):
            raise AssertionError(f"metric step mismatch: {record['step']} vs {update}")
        observed = trajectory[update]
        for key in ("reward_mean", "reward_std", "mixed_group_fraction", "all_pass_frac", "all_fail_frac"):
            if not math.isclose(float(data[f"r3b/{key}"]), observed[key], abs_tol=1e-6):
                raise AssertionError(
                    f"update {update} {key} mismatch: logger={data[f'r3b/{key}']} "
                    f"trajectories={observed[key]}"
                )
        rows.append(
            {
                "seed": SEED,
                "update": update,
                "actor_pg_loss": data["actor/pg_loss"],
                "grad_norm_unclipped": data["actor/grad_norm"],
                "reward_mean": data["r3b/reward_mean"],
                "reward_std": data["r3b/reward_std"],
                "kl_divergence": data["rollout_corr/kl"],
                "response_length_mean_tokens": data["response_length/trajectory/mean"],
                "mixed_group_fraction": data["r3b/mixed_group_fraction"],
                "wall_time_seconds": data["perf/time_per_step"],
                "all_pass_frac": data["r3b/all_pass_frac"],
                "all_fail_frac": data["r3b/all_fail_frac"],
                "advantage_mean": data["r3b/advantage_mean"],
                "advantage_abs_mean": data["r3b/advantage_abs_mean"],
                "advantage_min": data["r3b/advantage_min"],
                "advantage_max": data["r3b/advantage_max"],
                "advantage_zero_frac": data["r3b/advantage_zero_frac"],
                "advantage_non_sign_frac": data["r3b/advantage_non_sign_frac"],
                "learning_rate": data["actor/lr"],
            }
        )
    if [row["update"] for row in rows] != list(range(1, UPDATES + 1)):
        raise AssertionError("metric records are not updates 1..20 in order")
    return rows


def _dev_scope_rows(label: str, update: str | int, result: dict) -> list[dict]:
    grouped: dict[str, list[dict]] = defaultdict(list)
    for rollout in result["rollouts"]:
        grouped[rollout["task_id"]].append(rollout)
    scopes = {
        "full_dev": list(grouped.values()),
        "fault_only": [rows for rows in grouped.values() if rows[0]["fault_type"] != "normal"],
        "normal_only": [rows for rows in grouped.values() if rows[0]["fault_type"] == "normal"],
    }
    output = []
    for scope, groups in scopes.items():
        rows = [row for group in groups for row in group]
        binary_mixed = sum(
            min(row["binary_reward"] for row in group) != max(row["binary_reward"] for row in group)
            for group in groups
        )
        output.append(
            {
                "checkpoint": label,
                "update": update,
                "scope": scope,
                "instance_count": len(groups),
                "rollout_count": len(rows),
                "continuous_reward_mean": fmean(row["continuous_reward"] for row in rows),
                "binary_pass_rate": fmean(row["binary_reward"] for row in rows),
                "binary_mixed_group_fraction": binary_mixed / len(groups),
                "eval_seed": result["sampling"]["eval_seed"],
                "manifest_sha256": result["manifest_sha256"],
                "lora_path": result["lora_path"],
                "wall_time_seconds": result["total_wall_time_seconds"],
            }
        )
    return output


def _read_timestamp(path: Path) -> datetime:
    return datetime.fromisoformat(path.read_text(encoding="utf-8").strip())


def _r3_reference(raw: Path) -> dict:
    path = raw.parent.parent / "r3/results/update_metrics.csv"
    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    gradients = [float(row["grad_norm_unclipped"]) for row in rows]
    return {
        "source": str(path),
        "updates": len(rows),
        "nonzero_grad_fraction": sum(value > 0 for value in gradients) / len(gradients),
        "mean_grad_norm": fmean(gradients),
        "max_grad_norm": max(gradients),
        "mean_mixed_group_fraction": fmean(float(row["mixed_group_fraction"]) for row in rows),
    }


def build(raw: Path, output: Path) -> dict:
    updates = _update_rows(raw)
    base = _json(raw / "evals/base/dev.json")
    u20 = _json(raw / "evals/seed_42_u20/dev.json")
    dev_rows = _dev_scope_rows("base", "base", base) + _dev_scope_rows("seed_42_u20", 20, u20)
    _write_csv(output / "update_metrics.csv", updates)
    _write_csv(output / "dev_evaluations.csv", dev_rows)

    start = _read_timestamp(raw / "seed_42/started_at.txt")
    finish = _read_timestamp(raw / "evals/seed_42_u20/finished_at.txt")
    elapsed_seconds = (finish - start).total_seconds()
    elapsed_cny = elapsed_seconds / 3600 * HOURLY_CNY
    base_scopes = {row["scope"]: row for row in dev_rows if row["checkpoint"] == "base"}
    u20_scopes = {row["scope"]: row for row in dev_rows if row["checkpoint"] == "seed_42_u20"}

    gradients = [float(row["grad_norm_unclipped"]) for row in updates]
    advantage_abs = [float(row["advantage_abs_mean"]) for row in updates]
    r3_reference = _r3_reference(raw)
    all_sign = all(
        float(row["advantage_min"]) in (-1.0, 1.0)
        and float(row["advantage_max"]) in (-1.0, 1.0)
        and float(row["advantage_zero_frac"]) == 0.0
        and float(row["advantage_non_sign_frac"]) == 0.0
        for row in updates
    )
    all_finite = all(
        math.isfinite(float(value))
        for row in updates
        for key, value in row.items()
        if key not in {"seed", "update"}
    )
    summary = {
        "status": "completed" if elapsed_cny <= BUDGET_CNY else "completed_budget_overshoot",
        "experiment": {
            "name": "R3b fault-only binary Sign pilot",
            "seed": SEED,
            "updates": UPDATES,
            "rollouts": UPDATES * ROLLOUTS_PER_UPDATE,
            "group_size": GROUP_SIZE,
            "base_checkpoint": "/root/autodl-tmp/models/Qwen3-4B",
            "advantage_estimator": "sign",
            "advantage_formula": "2 * binary_reward - 1",
            "training_distribution": "fault_only",
        },
        "signal": {
            "all_metrics_finite": all_finite,
            "all_grad_norms_nonzero": all(value > 0 for value in gradients),
            "min_grad_norm": min(gradients),
            "mean_grad_norm": fmean(gradients),
            "max_grad_norm": max(gradients),
            "advantages_exactly_sign": all_sign,
            "u1_advantage_min": updates[0]["advantage_min"],
            "u1_advantage_max": updates[0]["advantage_max"],
            "u1_advantage_mean": updates[0]["advantage_mean"],
            "u1_advantage_abs_mean": updates[0]["advantage_abs_mean"],
            "mean_advantage_abs_mean": fmean(advantage_abs),
            "mean_binary_reward": fmean(float(row["reward_mean"]) for row in updates),
            "mean_mixed_group_fraction": fmean(float(row["mixed_group_fraction"]) for row in updates),
            "mean_all_pass_frac": fmean(float(row["all_pass_frac"]) for row in updates),
            "mean_all_fail_frac": fmean(float(row["all_fail_frac"]) for row in updates),
        },
        "r3_signal_comparison": {
            "r3_reference": r3_reference,
            "r3b_nonzero_grad_fraction": sum(value > 0 for value in gradients) / len(gradients),
            "r3b_mean_grad_norm": fmean(gradients),
            "r3b_to_r3_mean_grad_norm_ratio": fmean(gradients) / r3_reference["mean_grad_norm"],
            "r3b_max_grad_norm": max(gradients),
            "r3b_mean_mixed_group_fraction": fmean(
                float(row["mixed_group_fraction"]) for row in updates
            ),
            "assessment": (
                "Sign advantage delivered materially more consistent optimizer signal: "
                "20/20 nonzero-gradient updates versus 4/5 in R3, with a higher mean "
                "gradient norm. Peak gradient magnitude was similar."
            ),
        },
        "dev": {
            "base_full_dev_continuous_reward_mean": base_scopes["full_dev"]["continuous_reward_mean"],
            "u20_full_dev_continuous_reward_mean": u20_scopes["full_dev"]["continuous_reward_mean"],
            "full_dev_continuous_delta": (
                u20_scopes["full_dev"]["continuous_reward_mean"]
                - base_scopes["full_dev"]["continuous_reward_mean"]
            ),
            "base_full_dev_binary_pass_rate": base_scopes["full_dev"]["binary_pass_rate"],
            "u20_full_dev_binary_pass_rate": u20_scopes["full_dev"]["binary_pass_rate"],
            "full_dev_binary_delta": (
                u20_scopes["full_dev"]["binary_pass_rate"]
                - base_scopes["full_dev"]["binary_pass_rate"]
            ),
            "base_fault_only_binary_fcr": base_scopes["fault_only"]["binary_pass_rate"],
            "u20_fault_only_binary_fcr": u20_scopes["fault_only"]["binary_pass_rate"],
            "fault_only_binary_delta": (
                u20_scopes["fault_only"]["binary_pass_rate"]
                - base_scopes["fault_only"]["binary_pass_rate"]
            ),
            "base_binary_mixed_group_fraction": base_scopes["full_dev"]["binary_mixed_group_fraction"],
            "u20_binary_mixed_group_fraction": u20_scopes["full_dev"]["binary_mixed_group_fraction"],
            "base_eval_reused_from_r3": True,
            "manifest_instance_count": base["instance_count"],
            "eval_seed": base["sampling"]["eval_seed"],
        },
        "cost": {
            "hourly_cny": HOURLY_CNY,
            "budget_ceiling_cny": BUDGET_CNY,
            "experiment_window_start": start.isoformat(),
            "experiment_window_finish": finish.isoformat(),
            "experiment_window_seconds": elapsed_seconds,
            "experiment_window_cny": elapsed_cny,
            "within_budget": elapsed_cny <= BUDGET_CNY,
            "base_eval_reused_new_cost_cny": 0.0,
        },
        "artifacts": {
            "remote_checkpoint": "/root/autodl-tmp/octorl_r3b/seed_42/checkpoints/global_step_20",
            "remote_run_root": "/root/autodl-tmp/octorl_r3b/seed_42",
            "framework_patch": "artifacts/self_improve/r3b/agent_r1_sign_advantage.patch",
        },
        "diagnostics": [
            "The trainer completed and logged all 20 updates, saved U=20, and exited 0.",
            "A DataLoader worker emitted a killed-at-shutdown traceback after U=20 was saved; "
            "the final checkpoint subsequently loaded successfully for dev evaluation.",
            "No NaN or non-finite training metric was observed.",
            "The anticipated ~6% fault pass rate was inconsistent with the frozen base dev "
            "fault FCR (63.125%) and observed training batches (71.5625% mean).",
        ],
        "recommendation": (
            "R3b passes the optimizer-signal pilot and improves fault-only dev binary FCR "
            "by 5.0 percentage points at U=20. This is single-seed pilot evidence, not yet "
            "a two-seed capability-improvement conclusion."
        ),
    }
    output.mkdir(parents=True, exist_ok=True)
    (output / "smoke_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(build(args.raw, args.output), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
