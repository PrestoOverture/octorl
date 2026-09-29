import json
from pathlib import Path

import numpy as np

from scripts.self_improve.r6_power import pooled_three_seed_power, r4r_regression
from scripts.self_improve.r6_prepare_test3 import COUNTS, START_SEED, generate


def test_test3_exact_counts_dedup_and_second_generation_identical(tmp_path):
    first = generate(Path.cwd(), tmp_path)
    before = {path.name: path.read_bytes() for path in tmp_path.iterdir()}
    second = generate(Path.cwd(), tmp_path)
    after = {path.name: path.read_bytes() for path in tmp_path.iterdir()}
    assert first == second
    assert before == after
    assert first["counts"] == COUNTS
    assert all(item["overlap_count"] == 0 for item in first["dedup_evidence"].values())
    assert first["model_evaluated"] is False
    assert first["seed_range"][0] == START_SEED
    assert first["seed_range"][0] > 300281


def test_r4r_power_regression_is_field_for_field():
    assert r4r_regression(Path.cwd())["deep_equal_all_fields"]


def test_pooled_power_aa_and_structure_small_control():
    report = pooled_three_seed_power(np.array([0.5]), couplings={"q_hi": 0.25, "q_lo": 0.05},
                                     n=8, shifts=(0.0,), reps=4, resamples=20, seed=20260930)
    assert set(report["conditions"]) == {"independent", "q_hi", "q_lo"}
    assert all("detection_rate" in condition["shifts"]["0.00"]
               for condition in report["conditions"].values())


def test_written_power_aa_is_exact_if_present():
    path = Path("artifacts/self_improve/r6/prelaunch_power.json")
    if path.exists():
        assert json.loads(path.read_text())["aa_ci"] == [0.0, 0.0]
