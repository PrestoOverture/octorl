#!/usr/bin/env python3
"""R6 stage selection with an injectable chooser distribution q."""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import pandas as pd

from src.curriculum.failure_driven import (
    CELLS, PI0, build_stage_rows, cell_stats, failure_rates, selector_seed, within_cell_order,
)
from src.curriculum.model_chooser import distribution_from_q


SEEDS = (42, 137, 2718)
ARMS = ("fixed", "failure_driven")


def rule_q(stats: Mapping[str, Mapping[str, int]]) -> dict[str, float]:
    rates = failure_rates(stats)
    total = sum(PI0[cell] * rates[cell] for cell in CELLS)
    return {cell: PI0[cell] * rates[cell] / total for cell in CELLS}


def selection_for_q(
    *, q: Mapping[str, float], seed: int, stage: int, cursor: Mapping[str, int],
    seen_in_warmup: set[str], pool: pd.DataFrame,
) -> tuple[list[str], dict[str, int], dict[str, float], dict[str, int]]:
    p, counts = distribution_from_q(q)
    by_id = {row["extra_info"]["task_id"]: row.to_dict() for _, row in pool.iterrows()}
    if len(by_id) != 320:
        raise ValueError("fault-only pool must contain 320 unique tasks")
    orders = {}
    for cell in CELLS:
        instances = [task_id for task_id, row in by_id.items()
                     if row["extra_info"]["fault_type"] == cell]
        orders[cell] = within_cell_order(instances, seen_in_warmup, selector_seed(seed, cell))
    selected, next_cursor = build_stage_rows(
        counts, orders, cursor, train_seed=seed, stage_index=stage,
    )
    return selected, next_cursor, p, counts


def write_stage(
    output_dir: Path, *, q: Mapping[str, float], stats: Mapping[str, Mapping[str, int]],
    seed: int, stage: int, cursor: Mapping[str, int], seen_in_warmup: set[str],
    pool: pd.DataFrame, chooser: Mapping[str, Any] | None = None,
) -> tuple[dict[str, int], dict[str, Any]]:
    selected, next_cursor, p, counts = selection_for_q(
        q=q, seed=seed, stage=stage, cursor=cursor, seen_in_warmup=seen_in_warmup, pool=pool,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    by_id = {row["extra_info"]["task_id"]: row.to_dict() for _, row in pool.iterrows()}
    record = {
        "schema": "tool_recovery_v1", "record_type": "stage_distribution",
        "version": "r6-ch-v1", "selector_version": "ch-v1", "train_seed": int(seed),
        "window": {"start_update": 1 + 10 * (stage - 1), "end_update": 20 + 10 * (stage - 1)},
        "rho": 0.4,
        "cells": {cell: {**stats[cell], "f_c": failure_rates(stats)[cell], "q_c": float(q[cell]),
                         "p_c": p[cell], "count": counts[cell],
                         "selector_seed": selector_seed(seed, cell)} for cell in CELLS},
    }
    if chooser is not None:
        record["chooser"] = dict(chooser)
    (output_dir / "stage_distribution.json").write_text(
        json.dumps(record, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    selection = {"tasks": selected, "cursor_before": dict(cursor), "cursor_after": next_cursor}
    (output_dir / "selection.json").write_text(
        json.dumps(selection, indent=2, sort_keys=True) + "\n", encoding="utf-8",
    )
    pd.DataFrame([by_id[task] for task in selected]).to_parquet(output_dir / "train.parquet", index=False)
    return next_cursor, record


def _jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def _warmup_path(workspace: Path, seed: int) -> Path:
    if seed in (42, 137):
        return workspace / f"artifacts/self_improve/r4/warmup_records/seed_{seed}.jsonl"
    return workspace / "artifacts/self_improve/r4r/remote_records/seed_2718/warmup/attributed.jsonl"


def g0_regression(workspace: Path) -> dict[str, Any]:
    """Rebuild every R4r selection from records and compare the frozen outputs."""
    remote = workspace / "artifacts/self_improve/r4r/remote_records"
    pool = pd.read_parquet(workspace / "artifacts/self_improve/r3/data/train_fault_only.parquet")
    checks = []
    for seed in SEEDS:
        warmup_path = _warmup_path(workspace, seed)
        warmup = _jsonl(warmup_path)
        seen = {row["task_id"] for row in warmup}
        for arm in ARMS:
            sources: list[dict[str, Any]] = list(warmup)
            cursor = dict.fromkeys(CELLS, 0)
            for stage in range(1, 7):
                start = 20 + 10 * (stage - 1)
                window = (start - 19, start)
                stats = cell_stats(sources, window)
                frozen_dir = remote / f"seed_{seed}/{arm}/stage_{stage}"
                frozen_distribution = json.loads((frozen_dir / "stage_distribution.json").read_text())
                frozen_stats = {cell: {key: frozen_distribution["cells"][cell][key]
                                       for key in ("n", "fail", "infra")} for cell in CELLS}
                if stats != frozen_stats:
                    raise AssertionError(f"stats mismatch: seed={seed} arm={arm} stage={stage}")
                q = dict(PI0) if arm == "fixed" else rule_q(stats)
                selected, next_cursor, p, counts = selection_for_q(
                    q=q, seed=seed, stage=stage, cursor=cursor,
                    seen_in_warmup=seen, pool=pool,
                )
                frozen_counts = {cell: frozen_distribution["cells"][cell]["count"] for cell in CELLS}
                if counts != frozen_counts:
                    raise AssertionError(f"count mismatch: seed={seed} arm={arm} stage={stage}")
                if any(abs(p[cell] - frozen_distribution["cells"][cell]["p_c"]) > 1e-12 for cell in CELLS):
                    raise AssertionError(f"p mismatch: seed={seed} arm={arm} stage={stage}")
                selection = {"tasks": selected, "cursor_before": cursor, "cursor_after": next_cursor}
                selection_bytes = (json.dumps(selection, indent=2, sort_keys=True) + "\n").encode()
                if selection_bytes != (frozen_dir / "selection.json").read_bytes():
                    raise AssertionError(f"selection mismatch: seed={seed} arm={arm} stage={stage}")
                frozen_train = pd.read_parquet(frozen_dir / "train.parquet")
                frozen_ids = [row["task_id"] for row in frozen_train["extra_info"]]
                if selected != frozen_ids:
                    raise AssertionError(f"parquet task order mismatch: seed={seed} arm={arm} stage={stage}")
                checks.append({"seed": seed, "arm": arm, "stage": stage, "stats": True,
                               "counts": True, "p_tolerance": True, "selection_bytes": True,
                               "train_task_ids": True})
                cursor = next_cursor
                sources.extend(_jsonl(frozen_dir / "attributed.jsonl"))
    return {"windows": len(checks), "failure_driven_stages": 18, "fixed_stages": 18,
            "all_pass": True, "checks": checks}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--g0-regression", action="store_true")
    parser.add_argument("--output", type=Path,
                        default=Path("artifacts/self_improve/r6/g0_regression.json"))
    args = parser.parse_args()
    if not args.g0_regression:
        parser.error("this CPU contract exposes --g0-regression; runtime selection imports write_stage")
    workspace = Path(__file__).resolve().parents[2]
    result = g0_regression(workspace)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({key: result[key] for key in ("windows", "failure_driven_stages", "fixed_stages", "all_pass")}, sort_keys=True))


if __name__ == "__main__":
    main()
