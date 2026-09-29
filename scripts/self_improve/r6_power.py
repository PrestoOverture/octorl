#!/usr/bin/env python3
"""R6 pre-launch power: unchanged R4r control plus pooled three-seed design."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from scripts.self_improve import r4_analysis as r4
from scripts.self_improve.r4r_analysis import calibrations, synthetic_power


SEED = 20260930
REPS = 1000
RESAMPLES = 10000
N = 160
SHIFTS = (0.0, 0.03, 0.05, 0.10)
POOLED_SHIFTS = (0.0, 0.03, 0.05)


def r4r_regression(workspace: Path) -> dict[str, Any]:
    base, calibrated = calibrations(workspace)
    result = synthetic_power(base, couplings={key: value["q"] for key, value in calibrated.items()})
    result.update(n=160, calibrations=calibrated, source="base dev per-instance fault rates",
                  instance_rate_sampling="replacement, PCG64(20260928)",
                  aa_ci=list(r4.clustered_ci(np.zeros(160), seed=20260928)))
    result["null_coverage_pass"] = {
        key: (.93 <= value["shifts"]["0.00"]["coverage_of_zero"] <= .97
              if key == "independent" else value["shifts"]["0.00"]["detection_rate"] <= .05)
        for key, value in result["conditions"].items()
    }
    result["null_acceptance_rule"] = (
        "independent: coverage of 0 in [0.93, 0.97]; "
        "coupled: coverage reported, Type I (shift 0 detection) <= 0.05"
    )
    expected = json.loads((workspace / "artifacts/self_improve/r4r/prelaunch_power.json").read_text())
    if result != expected:
        raise RuntimeError("R4r power regression differs field-for-field")
    return {"deep_equal_all_fields": True, "sha_control": "artifacts/self_improve/r4r/prelaunch_power.json"}


def pooled_three_seed_power(
    source_rates: np.ndarray, *, couplings: dict[str, float], n: int = N,
    shifts: tuple[float, ...] = POOLED_SHIFTS, reps: int = REPS,
    resamples: int = RESAMPLES, seed: int = SEED,
) -> dict[str, Any]:
    if n <= 0 or len(source_rates) == 0:
        raise ValueError("source rates and n must be non-empty")
    indices = np.random.Generator(np.random.PCG64(seed)).integers(0, n, size=(resamples, n))
    result: dict[str, Any] = {"reps": reps, "resamples": resamples, "seed": seed,
                              "seeds_per_rep": 3, "conditions": {}}
    for condition_index, (name, q) in enumerate({"independent": 1.0, **couplings}.items()):
        if not 0 <= q <= 1:
            raise ValueError("coupling q must be in [0,1]")
        table = {}
        for shift in shifts:
            rng = np.random.Generator(np.random.PCG64(seed + 100 * condition_index + int(shift * 100)))
            detected = covered = 0
            all_positive_count = 0
            for _ in range(reps):
                rates = rng.choice(source_rates, size=n, replace=True)
                per_seed_d = []
                per_seed_q = []
                for _seed_index in range(3):
                    shared = rng.random((n, 4))
                    independent = rng.random((n, 4)) < q
                    arm_a = np.where(independent, rng.random((n, 4)), shared) < rates[:, None]
                    arm_b = np.where(independent, rng.random((n, 4)), shared) < np.clip(rates + shift, 0, 1)[:, None]
                    differences = (arm_b.astype(int) - arm_a.astype(int)).mean(axis=1)
                    per_seed_d.append(differences)
                    per_seed_q.append(float(differences.mean()))
                all_positive = all(value > 0 for value in per_seed_q)
                all_positive_count += all_positive
                pooled_d = np.mean(np.stack(per_seed_d), axis=0)
                low, high = np.quantile(pooled_d[indices].mean(axis=1), (0.025, 0.975))
                detected += all_positive and low > 0
                covered += low <= 0 <= high
            table[f"{shift:.2f}"] = {
                "coverage_of_zero": covered / reps,
                "all_three_Q_positive_rate": all_positive_count / reps,
                "detection_rate": detected / reps,
            }
        result["conditions"][name] = {"q": q, "shifts": table}
    return result


def build_report(workspace: Path, *, reps: int = REPS, resamples: int = RESAMPLES) -> dict[str, Any]:
    regression = r4r_regression(workspace)
    test3_seal = json.loads((workspace / "artifacts/self_improve/r6/data/test3_seal.json").read_text())
    fault_n = sum(test3_seal["counts"][cell]
                  for cell in ("constraint_violation", "missing_dependency", "stale_version"))
    if fault_n != N or test3_seal["model_evaluated"] is not False:
        raise RuntimeError("power requires the sealed, unevaluated test3 sample with 160 fault instances")
    base, calibrated = calibrations(workspace)
    couplings = {key: value["q"] for key, value in calibrated.items()}
    inherited = synthetic_power(base, couplings=couplings, n=N, shifts=SHIFTS,
                                reps=reps, resamples=resamples, seed=SEED)
    inherited["calibrations"] = calibrated
    inherited["n"] = N
    inherited["source"] = "base dev per-instance fault rates; test3 fault-instance count n=160"
    inherited["instance_rate_sampling"] = f"replacement, PCG64({SEED})"
    inherited["aa_ci"] = list(r4.clustered_ci(np.zeros(N), seed=SEED))
    inherited["null_coverage_pass"] = .93 <= inherited["conditions"]["independent"]["shifts"]["0.00"]["coverage_of_zero"] <= .97
    source_rates = r4._base_fault_rates(base)
    pooled = pooled_three_seed_power(source_rates, couplings=couplings, reps=reps,
                                     resamples=resamples, seed=SEED)
    return {"schema": "r6-prelaunch-power-v1", "r4r_regression": regression,
            "test3_manifest_sha256": test3_seal["manifest_sha256"],
            "test3_single_seed": inherited, "pooled_three_seed": pooled,
            "aa_ci": [0.0, 0.0]}


def markdown(report: dict[str, Any]) -> str:
    inherited = report["test3_single_seed"]
    lines = [
        "# R6 pre-launch power", "",
        f"{N} fault instances; 4 rollouts; {inherited['reps']} replications; "
        f"{inherited['resamples']} instance bootstrap resamples; PCG64({SEED}).", "",
        "The single-seed simulation is the unchanged R4r procedure with the test3 n=160 design. "
        "The pooled simulation requires all three seed-level effects to be positive and the pooled CI lower bound to exceed zero.", "",
        "| Condition | q | Independent-procedure null coverage | +0 detection | +.03 | +.05 | +.10 |", "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for name, condition in inherited["conditions"].items():
        shifts = condition["shifts"]
        lines.append(f"| {name} | {condition['q']:.9f} | {shifts['0.00']['coverage_of_zero']:.3f} | "
                     + " | ".join(f"{shifts[key]['detection_rate']:.3f}" for key in ("0.00", "0.03", "0.05", "0.10")) + " |")
    lines += ["", "## Pooled over three seeds", "",
              "| Condition | q | +0 detection | +.03 | +.05 |",
              "|---|---:|---:|---:|---:|"]
    for name, condition in report["pooled_three_seed"]["conditions"].items():
        shifts = condition["shifts"]
        lines.append(f"| {name} | {condition['q']:.9f} | "
                     + " | ".join(f"{shifts[key]['detection_rate']:.3f}" for key in ("0.00", "0.03", "0.05")) + " |")
    lines += ["", f"Independent null coverage pass: `{inherited['null_coverage_pass']}`.",
              "A/A CI: `[0, 0]`.", "R4r regression: all fields deep-equal.", "",
              r4.CAVEAT]
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reps", type=int, default=REPS)
    parser.add_argument("--resamples", type=int, default=RESAMPLES)
    parser.add_argument("--output-root", type=Path, default=Path("artifacts/self_improve/r6"))
    args = parser.parse_args()
    workspace = Path(__file__).resolve().parents[2]
    report = build_report(workspace, reps=args.reps, resamples=args.resamples)
    args.output_root.mkdir(parents=True, exist_ok=True)
    (args.output_root / "prelaunch_power.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (args.output_root / "prelaunch_power.md").write_text(markdown(report), encoding="utf-8")
    print(markdown(report), end="")
    if not report["test3_single_seed"]["null_coverage_pass"]:
        raise RuntimeError("frozen independent null coverage is outside [0.93, 0.97]")


if __name__ == "__main__":
    main()
