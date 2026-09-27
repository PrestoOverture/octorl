"""Frozen Q1/Q2 controls, paired dry run, and synthetic uncertainty checks."""

from pathlib import Path

import numpy as np
import pytest

from scripts.self_improve.r4_analysis import (
    analyze, calibrate_coupling, clustered_ci, compare, instance_differences, pair_models,
    read_eval, synthetic_power,
)


BASE = Path("artifacts/self_improve/r3c/results/base_dev_eval.json")
U80 = Path("artifacts/self_improve/r3c/results/seed_42/u80_dev.json")


def test_a_a_is_zero_with_degenerate_ci():
    base = read_eval(BASE)
    result = compare(base, base)
    assert set(result["per_instance"].values()) == {0.0}
    assert result["difference"] == 0.0
    assert result["ci95"] == [0.0, 0.0]


def test_pairing_keys_must_match():
    base = read_eval(BASE)
    other = dict(base)
    other.pop(next(iter(other)))
    with pytest.raises(ValueError, match="pairing keys differ"):
        pair_models(base, other)


def test_dev_dry_run_has_eight_up_zero_down():
    result = analyze({"base": read_eval(BASE), "fixed": read_eval(BASE), "failure_driven": read_eval(U80)})
    q2 = result["Q2_by_seed"]["single"]
    assert q2["sign_test"]["up"] == 8
    assert q2["sign_test"]["down"] == 0
    assert q2["difference"] == pytest.approx(0.05)
    assert result["Q2_decision"] == "not_assessed"


def test_cluster_bootstrap_retains_instance_dependence():
    # All four rollouts of an instance have the same outcome.  A rollout-level
    # bootstrap would divide the interval width by roughly two.
    d = np.array([1.0] * 20 + [-1.0] * 20)
    low, high = clustered_ci(d, seed=20260927)
    assert high - low > 0.48


def _synthetic_eval(fault_reward: int, normal_reward: int):
    return {(f"fault_{i}", j): {"task_id": f"fault_{i}", "rollout_index": j,
                                  "fault_type": "stale_version", "binary_reward": fault_reward}
            for i in range(40) for j in range(4)} | {
                (f"normal_{i}", j): {"task_id": f"normal_{i}", "rollout_index": j,
                                     "fault_type": "normal", "binary_reward": normal_reward}
                for i in range(10) for j in range(4)}


def test_three_seed_q1_and_q2_decision_rules():
    base = _synthetic_eval(0, 1)
    fixed = _synthetic_eval(1, 1)
    fd = _synthetic_eval(1, 1)
    models = {"base": base}
    for seed in (42, 137, 2718):
        models[f"fixed_{seed}"] = fixed
        models[f"failure_driven_{seed}"] = fd
    result = analyze(models, resamples=100)
    assert result["Q1_decision"] == "Q1_fixed_improves"
    assert result["Q2_decision"] == "no_evidence_of_difference"
    for seed in (42, 137, 2718):
        models[f"failure_driven_{seed}"] = base
    result = analyze(models, resamples=100)
    assert result["Q2_decision"] == "Q2_failure_driven_worse"
    for seed in (42, 137, 2718):
        models[f"fixed_{seed}"] = base
        models[f"failure_driven_{seed}"] = fd
    result = analyze(models, resamples=100)
    assert result["Q2_decision"] == "Q2_failure_driven_better"


def _base_instance_rates():
    base = read_eval(BASE)
    by_task = {}
    for row in base.values():
        if row["fault_type"] != "normal":
            by_task.setdefault(row["task_id"], []).append(row["binary_reward"])
    assert len(by_task) == 40 and {len(x) for x in by_task.values()} == {4}
    return np.array([np.mean(by_task[k]) for k in sorted(by_task)])


def test_null_coverage_and_power_controls():
    base = read_eval(BASE)
    root = Path("artifacts/self_improve/r3c/results")
    hi = calibrate_coupling(base, read_eval(root / "seed_42/u20_dev.json"),
                            read_eval(root / "seed_137/u20_dev.json"))
    lo = calibrate_coupling(base, read_eval(root / "seed_42/u50_dev.json"),
                            read_eval(root / "seed_42/u80_dev.json"))
    assert hi["observed_discordance_rate"] == 0.0375
    assert lo["observed_discordance_rate"] == 0.00625
    assert hi["q"] == pytest.approx(0.2608695652173913)
    assert lo["q"] == pytest.approx(0.04347826086956522)
    report = synthetic_power(base, couplings={"coupled_q_hi": hi["q"], "coupled_q_lo": lo["q"]})
    assert 0.93 <= report["conditions"]["independent"]["shifts"]["0.00"]["coverage_of_zero"] <= 0.97
    for name, condition in report["conditions"].items():
        for shift in ("0.00", "0.05", "0.10"):
            row = condition["shifts"][shift]
            assert 0 <= row["detection_rate"] <= 1
            print(name, shift, row)


def test_power_report_has_both_random_number_designs():
    report = synthetic_power(read_eval(BASE), couplings={"coupled_q_hi": 0.25, "coupled_q_lo": 0.05},
                             reps=20, resamples=100)
    assert set(report["conditions"]) == {"independent", "coupled_q_hi", "coupled_q_lo"}
    assert set(report["conditions"]["independent"]["shifts"]) == {"0.00", "0.05", "0.10"}
    assert report["conditions"]["coupled_q_lo"]["shifts"]["0.00"]["mean_ci_width"] > 0
    assert set(report["mde_at_80pct_power"]) == {"independent", "coupled_q_hi", "coupled_q_lo"}
