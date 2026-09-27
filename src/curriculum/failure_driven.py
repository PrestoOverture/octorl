"""Pure R4 failure-driven selector operations (fd-v1).

Mappings retain the frozen cell order.  Random permutations use an explicit
PCG64 seed; JSON serialization is canonical for reproducible stage records.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np


CELLS = ("constraint_violation", "missing_dependency", "stale_version")
PI0 = dict(zip(CELLS, (0.375, 0.3125, 0.3125)))


def selector_seed(train_seed: int, cell: str) -> int:
    if cell not in CELLS:
        raise ValueError(f"unknown cell: {cell}")
    return int(hashlib.sha256(f"r4|{int(train_seed)}|{cell}".encode()).hexdigest()[:8], 16)


def cell_stats(records: Iterable[Mapping[str, Any]], window: tuple[int, int]) -> dict[str, dict[str, int]]:
    """Count rollout exposure/failure over inclusive global steps [start, end]."""
    start, end = window
    if start > end:
        raise ValueError("window start exceeds end")
    stats = {c: {"n": 0, "fail": 0, "infra": 0} for c in CELLS}
    for row in records:
        step = int(row["global_step"])
        if not start <= step <= end:
            continue
        cell = row["fault_type"]
        if cell not in stats:
            raise ValueError(f"unknown fault_type: {cell}")
        flags = row.get("diagnostic_flags") or []
        if "infra_error" in flags or row.get("termination_reason") == "infra_error":
            stats[cell]["infra"] += 1
            continue
        reward = row["binary_reward"]
        if reward not in (0, 1):
            raise ValueError(f"binary_reward must be 0 or 1: {reward}")
        stats[cell]["n"] += 1
        stats[cell]["fail"] += int(reward == 0)
    return stats


def failure_rates(stats: Mapping[str, Mapping[str, int]]) -> dict[str, float]:
    return {c: (stats[c]["fail"] + 1) / (stats[c]["n"] + 2) for c in CELLS}


def mix(f: Mapping[str, float], pi0: Mapping[str, float] = PI0, rho: float = 0.4) -> dict[str, float]:
    if not 0 <= rho <= 1:
        raise ValueError("rho must be in [0, 1]")
    if not math.isclose(sum(pi0[c] for c in CELLS), 1.0, abs_tol=1e-12):
        raise ValueError("pi0 must sum to one")
    values = [float(f[c]) for c in CELLS]
    if any(not math.isfinite(v) or v < 0 for v in values):
        raise ValueError("failure rates must be finite and nonnegative")
    total = sum(pi0[c] * f[c] for c in CELLS)
    if total <= 0:
        raise ValueError("weighted failure rates sum to zero")
    return {c: rho * pi0[c] + (1 - rho) * pi0[c] * f[c] / total for c in CELLS}


def apply_cap(p: Mapping[str, float], pi0: Mapping[str, float] = PI0, max_p: float = 0.6) -> dict[str, float]:
    if max_p * len(CELLS) < 1:
        raise ValueError("cap cannot accommodate total probability")
    result = {c: float(p[c]) for c in CELLS}
    if not math.isclose(sum(result.values()), 1.0, abs_tol=1e-12):
        raise ValueError("p must sum to one")
    uncapped = set(CELLS)
    while True:
        high = [c for c in CELLS if c in uncapped and result[c] > max_p + 1e-12]
        if not high:
            break
        excess = sum(result[c] - max_p for c in high)
        for c in high:
            result[c] = max_p
            uncapped.remove(c)
        denominator = sum(pi0[c] for c in uncapped)
        if not uncapped or denominator <= 0:
            raise ValueError("no cells available for redistribution")
        for c in uncapped:
            result[c] += excess * pi0[c] / denominator
    return result


def allocate(p: Mapping[str, float], total: int = 40) -> dict[str, int]:
    if total < 0 or not math.isclose(sum(p[c] for c in CELLS), 1.0, abs_tol=1e-12):
        raise ValueError("invalid total or probabilities")
    raw = {c: total * p[c] for c in CELLS}
    counts = {c: math.floor(raw[c]) for c in CELLS}
    remainder = total - sum(counts.values())
    for c in sorted(CELLS, key=lambda c: -(raw[c] - counts[c]))[:remainder]:
        counts[c] += 1
    return counts


def within_cell_order(instances: Sequence[Any], seen_in_warmup: Iterable[Any], seed: int) -> list[Any]:
    """Permute unseen and seen references separately; seen references follow unseen."""
    if not isinstance(seed, int):
        raise TypeError("seed must be an integer")
    seen = set(seen_in_warmup)
    unique = list(dict.fromkeys(instances))
    if len(unique) != len(instances):
        raise ValueError("instance references must be unique")
    rng = np.random.Generator(np.random.PCG64(seed))
    unseen = [x for x in unique if x not in seen]
    used = [x for x in unique if x in seen]
    return [unseen[i] for i in rng.permutation(len(unseen))] + [used[i] for i in rng.permutation(len(used))]


def stage_permutation_seed(train_seed: int, stage_index: int) -> int:
    """A2 stage-order seed, shared by fixed and failure-driven arms."""
    if not isinstance(train_seed, int) or not isinstance(stage_index, int) or not 1 <= stage_index <= 6:
        raise ValueError("train_seed must be an integer and stage_index must be 1..6")
    payload = f"r4|{train_seed}|stage|{stage_index}".encode()
    return int(hashlib.sha256(payload).hexdigest()[:8], 16)


def build_stage_rows(
    counts: Mapping[str, int], orders: Mapping[str, Sequence[Any]], cursor: Mapping[str, int],
    *, train_seed: int, stage_index: int,
) -> tuple[list[Any], dict[str, int]]:
    """Select 40 refs, cycle exhausted pools, then permute the stage deterministically."""
    if sum(counts[c] for c in CELLS) != 40:
        raise ValueError("a stage must contain exactly 40 groups")
    rows: list[Any] = []
    next_cursor = dict(cursor)
    for c in CELLS:
        order = orders[c]
        n = counts[c]
        if n < 0 or (n and not order):
            raise ValueError(f"cell {c} has no selectable instances")
        start = cursor.get(c, 0)
        rows.extend(order[(start + i) % len(order)] for i in range(n))
        next_cursor[c] = start + n
    rng = np.random.Generator(np.random.PCG64(stage_permutation_seed(train_seed, stage_index)))
    return [rows[i] for i in rng.permutation(len(rows))], next_cursor


def stage_distribution_record(
    *, train_seed: int, window: tuple[int, int], input_record_refs: Sequence[str],
    stats: Mapping[str, Mapping[str, int]], rho: float, pi0: Mapping[str, float] = PI0,
    max_p: float = 0.6, groups: int = 40,
) -> dict[str, Any]:
    f = failure_rates(stats)
    before_cap = mix(f, pi0, rho)
    p = apply_cap(before_cap, pi0, max_p)
    counts = allocate(p, groups)
    return {
        "schema": "tool_recovery_v1", "record_type": "stage_distribution", "version": "fd-v1",
        "selector_version": "fd-v1",
        "window": {"start_update": window[0], "end_update": window[1]},
        "input_record_refs": list(input_record_refs), "train_seed": int(train_seed), "rho": rho,
        "cells": {c: {**stats[c], "f_c": f[c], "p_c": p[c], "count": counts[c],
                       "selector_seed": selector_seed(train_seed, c)} for c in CELLS},
    }


def canonical_json(record: Mapping[str, Any]) -> str:
    return json.dumps(record, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False) + "\n"


def write_stage_distribution(path: str | Path, record: Mapping[str, Any]) -> None:
    Path(path).write_text(canonical_json(record), encoding="utf-8")
