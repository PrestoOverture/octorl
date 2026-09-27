"""Paired R4 Q1/Q2 analysis. Example: --model base=base.json --model fixed_42=fixed.json.

The CLI accepts arbitrary model-to-file mappings and writes JSON plus Markdown.
No test data is accessed unless its path is explicitly supplied by the caller.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping

import numpy as np


BOOTSTRAP_SEED = 20260927
CAVEAT = "CI covers test-instance and evaluation-sampling uncertainty only; training-seed variance is not estimated (n=3)"


def read_eval(path: str | Path) -> dict[tuple[str, int], dict[str, Any]]:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw.get("rollouts"), list):
        raise ValueError(f"{path}: expected top-level rollouts list")
    records = {}
    for row in raw["rollouts"]:
        key = (row["task_id"], int(row["rollout_index"]))
        if key in records:
            raise ValueError(f"{path}: duplicate pairing key {key}")
        if row["binary_reward"] not in (0, 1):
            raise ValueError(f"{path}: nonbinary reward at {key}")
        records[key] = row
    return records


def pair_models(a: Mapping, b: Mapping) -> list[tuple[dict, dict]]:
    if set(a) != set(b):
        missing_a = sorted(set(b) - set(a))[:3]
        missing_b = sorted(set(a) - set(b))[:3]
        raise ValueError(f"pairing keys differ: absent from A={missing_a}; absent from B={missing_b}")
    pairs = []
    for key in sorted(a):
        if a[key]["fault_type"] != b[key]["fault_type"]:
            raise ValueError(f"fault_type differs at {key}")
        pairs.append((a[key], b[key]))
    return pairs


def instance_differences(a: Mapping, b: Mapping) -> dict[str, float]:
    grouped: dict[str, list[float]] = defaultdict(list)
    for left, right in pair_models(a, b):
        if left["fault_type"] != "normal":
            grouped[left["task_id"]].append(right["binary_reward"] - left["binary_reward"])
    if not grouped:
        raise ValueError("no fault instances")
    sizes = {len(v) for v in grouped.values()}
    if len(sizes) != 1:
        raise ValueError("unequal rollout count per fault instance")
    return {k: float(np.mean(grouped[k])) for k in sorted(grouped)}


def clustered_ci(d: np.ndarray, *, resamples: int = 10000, seed: int = BOOTSTRAP_SEED) -> tuple[float, float]:
    """Percentile CI, resampling complete instances with replacement."""
    values = np.asarray(d, dtype=float)
    if values.ndim != 1 or len(values) == 0:
        raise ValueError("d must be a nonempty instance vector")
    rng = np.random.Generator(np.random.PCG64(seed))
    idx = rng.integers(0, len(values), size=(resamples, len(values)))
    draws = values[idx].mean(axis=1)
    return tuple(float(x) for x in np.quantile(draws, (0.025, 0.975)))


def sign_test(a: Mapping, b: Mapping) -> dict[str, Any]:
    up = down = 0
    for left, right in pair_models(a, b):
        if left["fault_type"] == "normal":
            continue
        delta = right["binary_reward"] - left["binary_reward"]
        up += delta > 0
        down += delta < 0
    n = up + down
    p_greater = sum(math.comb(n, k) for k in range(up, n + 1)) / (2**n) if n else 1.0
    p_less = sum(math.comb(n, k) for k in range(0, up + 1)) / (2**n) if n else 1.0
    return {"up": up, "down": down, "one_sided_p_greater": p_greater,
            "one_sided_p_less": p_less, "role": "descriptive_only_not_clustered"}


def fault_rate(records: Mapping) -> float:
    rows = [r["binary_reward"] for r in records.values() if r["fault_type"] != "normal"]
    if not rows:
        raise ValueError("no fault rollouts")
    return float(np.mean(rows))


def normal_rate(records: Mapping) -> float:
    rows = [r["binary_reward"] for r in records.values() if r["fault_type"] == "normal"]
    if not rows:
        raise ValueError("no normal rollouts")
    return float(np.mean(rows))


def _base_fault_rates(base_dev: Mapping) -> np.ndarray:
    by_task: dict[str, list[int]] = defaultdict(list)
    for row in base_dev.values():
        if row["fault_type"] != "normal":
            by_task[row["task_id"]].append(row["binary_reward"])
    if len(by_task) != 40 or {len(v) for v in by_task.values()} != {4}:
        raise ValueError("power source must contain 40 fault instances with four rollouts each")
    return np.array([np.mean(by_task[k]) for k in sorted(by_task)], dtype=float)


def calibrate_coupling(base_dev: Mapping, a: Mapping, b: Mapping) -> dict[str, float]:
    """Match null discordance to an observed paired checkpoint comparison."""
    rates = _base_fault_rates(base_dev)
    expected = float(np.mean(2 * rates * (1 - rates)))
    fault_pairs = [(left, right) for left, right in pair_models(a, b) if left["fault_type"] != "normal"]
    observed = sum(left["binary_reward"] != right["binary_reward"] for left, right in fault_pairs) / len(fault_pairs)
    if expected <= 0 or observed > expected:
        raise ValueError("observed discordance cannot be calibrated to q in [0, 1]")
    return {"observed_discordance_rate": observed, "independent_null_discordance_rate": expected,
            "q": observed / expected, "fault_rollout_pairs": len(fault_pairs)}


def synthetic_power(base_dev: Mapping, *, couplings: Mapping[str, float], reps: int = 1000,
                    resamples: int = 10000, seed: int = BOOTSTRAP_SEED) -> dict[str, Any]:
    """Estimate coverage and power with independent and partially coupled draws."""
    rates = _base_fault_rates(base_dev)
    result: dict[str, Any] = {"reps": reps, "resamples": resamples, "seed": seed, "conditions": {}}
    conditions = {"independent": 1.0, **couplings}
    for condition_index, (condition, q) in enumerate(conditions.items()):
        if not 0 <= q <= 1:
            raise ValueError(f"coupling probability out of range for {condition}")
        table = {}
        for shift in (0.0, 0.05, 0.10):
            rng = np.random.Generator(np.random.PCG64(seed + 100 * condition_index + int(shift * 100)))
            covered = detected = 0
            widths = []
            for _ in range(reps):
                shared = rng.random((40, 4))
                independent = rng.random((40, 4)) < q
                u_a = np.where(independent, rng.random((40, 4)), shared)
                u_b = np.where(independent, rng.random((40, 4)), shared)
                a = u_a < rates[:, None]
                b = u_b < np.clip(rates + shift, 0, 1)[:, None]
                d = (b.astype(int) - a.astype(int)).mean(axis=1)
                low, high = clustered_ci(d, resamples=resamples, seed=seed)
                covered += low <= 0 <= high
                detected += low > 0
                widths.append(high - low)
            table[f"{shift:.2f}"] = {"coverage_of_zero": covered / reps,
                                      "detection_rate": detected / reps,
                                      "mean_ci_width": float(np.mean(widths))}
        result["conditions"][condition] = {"q": q, "shifts": table}
        result.setdefault("mde_at_80pct_power", {})[condition] = next(
            (float(shift) for shift in ("0.05", "0.10") if table[shift]["detection_rate"] >= 0.8),
            ">0.10 within tested shifts",
        )
    return result


def compare(a: Mapping, b: Mapping, *, seed: int = BOOTSTRAP_SEED, resamples: int = 10000) -> dict[str, Any]:
    d = instance_differences(a, b)
    values = np.array(list(d.values()), dtype=float)
    return {
        "difference": float(values.mean()), "ci95": list(clustered_ci(values, seed=seed, resamples=resamples)),
        "per_instance": d, "sign_test": sign_test(a, b),
        "a_fault_rate": fault_rate(a), "b_fault_rate": fault_rate(b),
    }


def pooled_ci(comparisons: Mapping[str, Mapping], *, seed: int = BOOTSTRAP_SEED, resamples: int = 10000) -> list[float]:
    ids = [set(c["per_instance"]) for c in comparisons.values()]
    if any(item != ids[0] for item in ids):
        raise ValueError("fault instance keys differ across seeds")
    ordered = sorted(ids[0])
    pooled_d = np.mean([[c["per_instance"][task] for task in ordered] for c in comparisons.values()], axis=0)
    return list(clustered_ci(pooled_d, seed=seed, resamples=resamples))


def analyze(models: Mapping[str, Mapping], *, seed: int = BOOTSTRAP_SEED, resamples: int = 10000) -> dict[str, Any]:
    base = models.get("base")
    fixed = {name.removeprefix("fixed_") if name.startswith("fixed_") else "single": data
             for name, data in models.items() if name == "fixed" or name.startswith("fixed_")}
    fd = {name.removeprefix("failure_driven_") if name.startswith("failure_driven_") else "single": data
          for name, data in models.items() if name == "failure_driven" or name.startswith("failure_driven_")}
    if not fixed and not fd:
        raise ValueError("provide fixed and/or failure_driven models")
    if fd and set(fd) != set(fixed):
        raise ValueError("fixed and failure_driven seed mappings differ")
    if base is None and not fd:
        raise ValueError("base is required for Q1")
    result: dict[str, Any] = {"caveat": CAVEAT, "bootstrap_seed": seed, "resamples": resamples}
    q1 = {s: compare(base, data, seed=seed, resamples=resamples) for s, data in fixed.items()} if base else {}
    q2 = {s: compare(fixed[s], data, seed=seed, resamples=resamples) for s, data in fd.items()}
    result["Q1_by_seed"] = q1
    result["Q2_by_seed"] = q2
    for label, comparisons in (("Q1", q1), ("Q2", q2)):
        if len(comparisons) == 3:
            result[label + "_pooled"] = {
                "difference": float(np.mean([c["difference"] for c in comparisons.values()])),
                "ci95": pooled_ci(comparisons, seed=seed, resamples=resamples),
            }
    base_normal = normal_rate(base) if base else None
    result["base_normal_pass_rate"] = base_normal
    result["normal_guardrail"] = {
        name: {"rate": normal_rate(data), "passes": base_normal is not None and normal_rate(data) >= base_normal - 0.05}
        for name, data in models.items() if name != "base"
    }
    result["Q1_decision"] = "not_assessed"
    result["Q2_decision"] = "not_assessed"
    # Preregistered all-three-seed claims are unavailable in a single-seed dry run.
    if len(q1) == 3:
        passes = all(result["normal_guardrail"]["fixed_" + s]["passes"] for s in q1)
        if all(c["difference"] > 0 for c in q1.values()) and result["Q1_pooled"]["ci95"][0] > 0 and passes:
            result["Q1_decision"] = "Q1_fixed_improves"
        else:
            result["Q1_decision"] = "no_evidence_of_improvement"
    if len(q2) == 3:
        guardrail = all(result["normal_guardrail"]["failure_driven_" + s]["passes"] for s in q2)
        bounds = result["Q2_pooled"]["ci95"]
        if all(c["difference"] > 0 for c in q2.values()) and bounds[0] > 0 and guardrail:
            result["Q2_decision"] = "Q2_failure_driven_better"
        elif all(c["difference"] < 0 for c in q2.values()) and bounds[1] < 0:
            result["Q2_decision"] = "Q2_failure_driven_worse"
        else:
            result["Q2_decision"] = "no_evidence_of_difference"
    return result


def markdown_table(result: Mapping[str, Any]) -> str:
    lines = ["| Estimand | Seed | Difference | 95% instance CI | Sign up/down (descriptive) |",
             "|---|---|---:|---|---:|"]
    for label in ("Q1", "Q2"):
        for seed, item in result[label + "_by_seed"].items():
            sign = item["sign_test"]
            lines.append(f"| {label} | {seed} | {item['difference']:.6f} | [{item['ci95'][0]:.6f}, {item['ci95'][1]:.6f}] | {sign['up']}/{sign['down']} |")
        if label + "_pooled" in result:
            item = result[label + "_pooled"]
            lines.append(f"| {label} pooled | all | {item['difference']:.6f} | [{item['ci95'][0]:.6f}, {item['ci95'][1]:.6f}] | — |")
    if "synthetic_power" in result:
        lines.append("\nSynthetic MDE at 80% detection: " + json.dumps(
            result["synthetic_power"]["mde_at_80pct_power"], sort_keys=True))
    return "\n".join(lines) + f"\n\n{result['caveat']}\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", action="append", required=True, metavar="NAME=FILE")
    parser.add_argument("--out-json", type=Path, required=True)
    parser.add_argument("--out-md", type=Path, required=True)
    parser.add_argument("--power-base", type=Path, default=Path(__file__).resolve().parents[2] /
                        "artifacts/self_improve/r3c/results/base_dev_eval.json")
    r3c = Path(__file__).resolve().parents[2] / "artifacts/self_improve/r3c/results"
    parser.add_argument("--q-hi-a", type=Path, default=r3c / "seed_42/u20_dev.json")
    parser.add_argument("--q-hi-b", type=Path, default=r3c / "seed_137/u20_dev.json")
    parser.add_argument("--q-lo-a", type=Path, default=r3c / "seed_42/u50_dev.json")
    parser.add_argument("--q-lo-b", type=Path, default=r3c / "seed_42/u80_dev.json")
    args = parser.parse_args()
    paths = {}
    for item in args.model:
        if "=" not in item:
            parser.error("--model requires NAME=FILE")
        name, path = item.split("=", 1)
        if name in paths:
            parser.error(f"duplicate model name {name}")
        paths[name] = path
    result = analyze({name: read_eval(path) for name, path in paths.items()})
    if result["Q2_decision"] == "no_evidence_of_difference":
        power_base = read_eval(args.power_base)
        calibrations = {
            "coupled_q_hi": calibrate_coupling(power_base, read_eval(args.q_hi_a), read_eval(args.q_hi_b)),
            "coupled_q_lo": calibrate_coupling(power_base, read_eval(args.q_lo_a), read_eval(args.q_lo_b)),
        }
        result["synthetic_power"] = synthetic_power(power_base,
            couplings={name: item["q"] for name, item in calibrations.items()})
        result["synthetic_power"]["calibrations"] = calibrations
    args.out_json.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    args.out_md.write_text(markdown_table(result), encoding="utf-8")
    print(json.dumps({"Q1_decision": result["Q1_decision"], "Q2_decision": result["Q2_decision"],
                      "Q2_pooled": result.get("Q2_pooled")}, sort_keys=True))


if __name__ == "__main__":
    main()
