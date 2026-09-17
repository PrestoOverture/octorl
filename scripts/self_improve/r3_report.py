#!/usr/bin/env python3
"""Build the R3 smoke-gate metrics, dev table, curves, and summary."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from statistics import mean, pstdev

import matplotlib.pyplot as plt


HOURLY_CNY = 2.18
FINAL_TEST_RESERVE_CNY = 3.0
GROUP_SIZE = 4
BATCH_SIZE = 4
ROLLOUTS_PER_UPDATE = GROUP_SIZE * BATCH_SIZE
FULL_UPDATE_COUNT = 200
FULL_DEV_EVAL_COUNT = 25  # base + (U1, U5, U10..U100) for two seeds
CLIP_GRAD = 1.0
R3_REMOTE_STARTED_AT = "2026-09-16T17:57:48+08:00"
R3_REMOTE_FINISHED_AT = "2026-09-16T19:18:51+08:00"


def _json_lines(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _metrics(raw: Path) -> list[dict]:
    sources = [
        raw / "seed_42/metrics_target_1.jsonl",
        raw / "seed_42/metrics_target_5_attempt1_cpu_gradient.jsonl",
        raw / "seed_42/metrics_target_5.jsonl",
    ]
    rows: dict[int, dict] = {}
    for path in sources:
        for record in _json_lines(path):
            data = record["data"]
            rows[int(data["training/global_step"])] = data
    if sorted(rows) != [1, 2, 3, 4, 5]:
        raise AssertionError(f"expected updates 1..5, found {sorted(rows)}")
    return [rows[step] for step in sorted(rows)]


def _trajectory_stats(raw: Path) -> dict[int, dict]:
    rows = _json_lines(raw / "seed_42/trajectories.jsonl")
    if len(rows) != 5 * ROLLOUTS_PER_UPDATE:
        raise AssertionError(f"expected 80 successful smoke trajectories, found {len(rows)}")
    result = {}
    for index in range(5):
        batch = rows[index * ROLLOUTS_PER_UPDATE : (index + 1) * ROLLOUTS_PER_UPDATE]
        rewards = [float(row["continuous_reward"]) for row in batch]
        groups: dict[str, list[float]] = defaultdict(list)
        for row in batch:
            groups[row["task_id"]].append(float(row["continuous_reward"]))
        if len(groups) != BATCH_SIZE or any(len(values) != GROUP_SIZE for values in groups.values()):
            raise AssertionError(f"update {index + 1} does not contain four G=4 groups")
        mixed = sum(max(values) - min(values) > 1e-12 for values in groups.values())
        result[index + 1] = {
            "reward_mean": mean(rewards),
            "reward_std": pstdev(rewards),
            "mixed_group_fraction": mixed / len(groups),
        }
    return result


def _peak_vram_mib(raw: Path) -> float:
    # Only successful smoke processes: U1 and the successful U3-U5 retry.
    values = []
    for path in [raw / "seed_42/u1_vram.csv", raw / "seed_42/u5_retry3_vram.csv"]:
        with path.open(newline="", encoding="utf-8") as stream:
            for row in csv.reader(stream):
                try:
                    values.append(float(row[1]))
                except (IndexError, ValueError):
                    continue
    return max(values)


def _dev_rows(raw: Path) -> list[dict]:
    specs = [
        ("base", None, raw / "evals/base/dev.json"),
        ("seed_42", 1, raw / "evals/seed_42_u1/dev.json"),
        ("seed_42", 5, raw / "evals/seed_42_u5/dev.json"),
    ]
    rows = []
    for seed, update, path in specs:
        data = json.loads(path.read_text(encoding="utf-8"))
        rows.append(
            {
                "seed": seed,
                "update": "base" if update is None else update,
                "continuous_reward_mean": data["continuous_reward_mean"],
                "binary_fcr": data["binary_fcr"],
                "binary_npr": data["binary_npr"],
                "mixed_group_fraction": data["mixed_group_fraction"],
                "rollout_count": data["rollout_count"],
                "wall_time_seconds": data["total_wall_time_seconds"],
                "manifest_sha256": data["manifest_sha256"],
                "lora_path": data["lora_path"],
            }
        )
    return rows


def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _plot(rows: list[dict], output: Path) -> None:
    updates = [row["update"] for row in rows]
    fig, axes = plt.subplots(2, 2, figsize=(10, 7), constrained_layout=True)
    series = [
        ("actor_pg_loss", "Actor policy loss"),
        ("kl_divergence", "KL divergence from base"),
        ("reward_mean", "Continuous reward (train)"),
        ("mixed_group_fraction", "Mixed-group fraction"),
    ]
    for axis, (key, title) in zip(axes.flat, series, strict=True):
        values = [row[key] for row in rows]
        axis.plot(updates, values, marker="o", color="#2457C5", linewidth=2)
        if key == "reward_mean":
            std = [row["reward_std"] for row in rows]
            axis.fill_between(
                updates,
                [value - spread for value, spread in zip(values, std, strict=True)],
                [value + spread for value, spread in zip(values, std, strict=True)],
                color="#2457C5",
                alpha=0.15,
                label="±1 population SD",
            )
            axis.legend(frameon=False, fontsize=8)
        axis.set_title(title)
        axis.set_xlabel("Update")
        axis.grid(alpha=0.25)
    fig.suptitle("R3 seed 42 — five-update smoke gate (budget stop)", fontsize=13)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=160)
    plt.close(fig)


def build(raw: Path, output: Path) -> dict:
    metric_records = _metrics(raw)
    trajectory = _trajectory_stats(raw)
    metric_rows = []
    for data in metric_records:
        update = int(data["training/global_step"])
        grad = float(data["actor/grad_norm"])
        metric_rows.append(
            {
                "seed": 42,
                "update": update,
                "actor_pg_loss": data["actor/pg_loss"],
                "actor_kl_loss": data["actor/kl_loss"],
                "grad_norm_unclipped": grad,
                "grad_norm_clipped": min(grad, CLIP_GRAD),
                "reward_mean": trajectory[update]["reward_mean"],
                "reward_std": trajectory[update]["reward_std"],
                "kl_divergence": data["rollout_corr/kl"],
                "response_length_mean_tokens": data["response_length/trajectory/mean"],
                "mixed_group_fraction": trajectory[update]["mixed_group_fraction"],
                "wall_time_seconds": data["perf/time_per_step"],
                "tokens_per_second": data["perf/throughput"],
                "learning_rate": data["actor/lr"],
            }
        )
    dev_rows = _dev_rows(raw)
    _write_csv(output / "update_metrics.csv", metric_rows)
    _write_csv(output / "dev_evaluations.csv", dev_rows)
    _plot(metric_rows, output / "seed_42_smoke_curves.png")

    step_seconds = sum(float(row["wall_time_seconds"]) for row in metric_rows)
    measured_hours = step_seconds / 3600
    projected_training_hours = measured_hours / 5 * FULL_UPDATE_COUNT
    dev_seconds = [float(row["wall_time_seconds"]) for row in dev_rows]
    projected_dev_hours = mean(dev_seconds) * FULL_DEV_EVAL_COUNT / 3600
    projected_training_cny = projected_training_hours * HOURLY_CNY
    projected_dev_cny = projected_dev_hours * HOURLY_CNY
    projected_total_cny = projected_training_cny + projected_dev_cny + FINAL_TEST_RESERVE_CNY
    actual_window_hours = (
        datetime.fromisoformat(R3_REMOTE_FINISHED_AT) - datetime.fromisoformat(R3_REMOTE_STARTED_AT)
    ).total_seconds() / 3600
    base = dev_rows[0]
    u5 = dev_rows[-1]
    summary = {
        "status": "stopped_at_smoke_gate",
        "stop_reason": "projected_cost_exceeds_50_cny",
        "completed_seeds": {"42": 5, "137": 0},
        "smoke": {
            "updates": 5,
            "rollouts": 80,
            "step_wall_time_seconds": step_seconds,
            "updates_per_hour": 5 / measured_hours,
            "rollouts_per_hour": 80 / measured_hours,
            "peak_vram_mib": _peak_vram_mib(raw),
            "all_metrics_finite": all(
                math.isfinite(float(value))
                for row in metric_rows
                for key, value in row.items()
                if key not in {"seed", "update"}
            ),
            "max_grad_norm": max(float(row["grad_norm_unclipped"]) for row in metric_rows),
            "nan_observed": False,
        },
        "cost": {
            "hourly_cny": HOURLY_CNY,
            "observed_remote_window_start": R3_REMOTE_STARTED_AT,
            "observed_remote_window_finish": R3_REMOTE_FINISHED_AT,
            "observed_remote_window_hours": actual_window_hours,
            "observed_remote_window_cny": actual_window_hours * HOURLY_CNY,
            "projected_training_hours": projected_training_hours,
            "projected_training_cny": projected_training_cny,
            "projected_dev_hours_from_observed_mean": projected_dev_hours,
            "projected_dev_cny_from_observed_mean": projected_dev_cny,
            "reserved_final_test_cny": FINAL_TEST_RESERVE_CNY,
            "projected_total_cny": projected_total_cny,
            "budget_ceiling_cny": 50.0,
        },
        "dev": {
            "base_continuous_reward_mean": base["continuous_reward_mean"],
            "u1_continuous_reward_mean": dev_rows[1]["continuous_reward_mean"],
            "u5_continuous_reward_mean": u5["continuous_reward_mean"],
            "u5_delta_vs_base": u5["continuous_reward_mean"] - base["continuous_reward_mean"],
            "base_binary_fcr": base["binary_fcr"],
            "u5_binary_fcr": u5["binary_fcr"],
            "u5_binary_fcr_delta_vs_base": u5["binary_fcr"] - base["binary_fcr"],
        },
        "recommendation": "no improvement (provisional; smoke-only, seed 137 not run due budget stop)",
        "remote_checkpoint_root": "/root/autodl-tmp/octorl_r3/seed_42/checkpoints",
        "remote_checkpoints": [
            f"/root/autodl-tmp/octorl_r3/seed_42/checkpoints/global_step_{step}" for step in range(1, 6)
        ],
        "diagnostics": [
            "Initial U3 attempt hit the FSDP2 CPU-gradient/CUDA-parameter mismatch with policy offload.",
            "A policy-offload-disabled retry could not start vLLM because the resident actor reduced free VRAM.",
            "The successful U3-U5 retry used manual parameter and optimizer offload with policy offload disabled.",
            "A dataloader worker was killed during shutdown after U5 was saved; the trainer exited 0.",
            "The framework logs one pre-clip global grad norm; clipped norm is derived as min(norm, 1.0).",
        ],
    }
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
