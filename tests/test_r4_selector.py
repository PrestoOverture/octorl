"""The nine frozen selector synthetic controls and the A1 infra amendment."""

import hashlib
import json
import subprocess
import sys

import pytest

from src.curriculum.failure_driven import (
    CELLS, PI0, allocate, apply_cap, build_stage_rows, canonical_json,
    cell_stats, failure_rates, mix, selector_seed, stage_distribution_record, stage_permutation_seed,
    within_cell_order,
)


def test_equal_failure_rates_return_pi0():
    assert all(mix(dict.fromkeys(CELLS, 0.5), PI0, 0.4)[c] == pytest.approx(PI0[c]) for c in CELLS)


def test_nearly_equal_failure_rates_are_continuous():
    f = dict(zip(CELLS, (0.5, 0.5, 0.5 + 1e-9)))
    p = mix(f, PI0, 0.4)
    assert max(abs(p[c] - PI0[c]) for c in CELLS) < 1e-8


def test_fixed_rho_returns_pi0_for_any_failure_rates():
    assert mix(dict(zip(CELLS, (0.1, 0.8, 0.3))), PI0, 1.0) == PI0


def test_zero_exposure_has_half_smoothed_failure_rate():
    stats = {c: {"n": 0, "fail": 0, "infra": 0} for c in CELLS}
    assert failure_rates(stats) == dict.fromkeys(CELLS, 0.5)


def test_one_failure_cell_cap_coverage_and_normalization():
    p = apply_cap(mix(dict(zip(CELLS, (1.0, 0.0, 0.0))), PI0, 0.4), PI0)
    assert max(p.values()) <= 0.6
    assert all(p[c] >= 0.4 * PI0[c] for c in CELLS)
    assert abs(sum(p.values()) - 1) <= 1e-12


def test_allocation_uses_largest_remainder_and_yaml_ties():
    p = {CELLS[0]: 0.375, CELLS[1]: 0.3125, CELLS[2]: 0.3125}
    counts = allocate(p, 40)
    assert counts == dict(zip(CELLS, (15, 13, 12)))
    assert sum(counts.values()) == 40
    assert all(abs(counts[c] - 40 * p[c]) <= 1 for c in CELLS)


def test_stage_distribution_byte_identical_across_processes():
    code = """from src.curriculum.failure_driven import *
s={c:{'n':10,'fail':i+1,'infra':0} for i,c in enumerate(CELLS)}
print(canonical_json(stage_distribution_record(train_seed=42,window=(1,20),input_record_refs=['warmup.jsonl'],stats=s,rho=0.4)),end='')"""
    a = subprocess.check_output([sys.executable, "-c", code])
    b = subprocess.check_output([sys.executable, "-c", code])
    assert a == b
    assert hashlib.sha256(a).hexdigest() == hashlib.sha256(b).hexdigest()


def test_infra_error_only_is_excluded_not_policy_diagnostics():
    rows = [
        {"global_step": 3, "fault_type": CELLS[0], "binary_reward": 0, "diagnostic_flags": ["no_read_config"]},
        {"global_step": 3, "fault_type": CELLS[0], "binary_reward": 0, "diagnostic_flags": ["infra_error"]},
    ]
    assert cell_stats(rows, (1, 20))[CELLS[0]] == {"n": 1, "fail": 1, "infra": 1}


def test_failure_rate_uses_rate_not_failure_count():
    stats = {CELLS[0]: {"n": 100, "fail": 50}, CELLS[1]: {"n": 10, "fail": 5},
             CELLS[2]: {"n": 20, "fail": 10}}
    rates = failure_rates(stats)
    assert rates[CELLS[0]] == pytest.approx(51 / 102)
    assert rates[CELLS[1]] == pytest.approx(6 / 12)
    assert rates[CELLS[2]] == pytest.approx(11 / 22)


def test_stage_record_applies_cap():
    stats = {CELLS[0]: {"n": 100, "fail": 100, "infra": 0},
             CELLS[1]: {"n": 100, "fail": 0, "infra": 0},
             CELLS[2]: {"n": 100, "fail": 0, "infra": 0}}
    record = stage_distribution_record(train_seed=42, window=(1, 20), input_record_refs=["x"], stats=stats, rho=0.4)
    assert record["cells"][CELLS[0]]["p_c"] == pytest.approx(0.6)


def test_seen_instances_go_last_and_orders_cycle():
    order = within_cell_order([1, 2, 3], {2}, selector_seed(42, CELLS[0]))
    assert set(order[:2]) == {1, 3} and order[-1] == 2
    orders = {CELLS[0]: order, CELLS[1]: [4], CELLS[2]: [5]}
    rows, cursor = build_stage_rows(dict(zip(CELLS, (38, 1, 1))), orders, dict.fromkeys(CELLS, 0),
                                    train_seed=42, stage_index=1)
    assert rows.count(4) == rows.count(5) == 1
    assert len(rows) == 40 and cursor[CELLS[0]] == 38


def test_a2_worked_example_and_real_warmup_counts():
    stats = {CELLS[0]: {"n": 120, "fail": 15, "infra": 0},
             CELLS[1]: {"n": 104, "fail": 53, "infra": 0},
             CELLS[2]: {"n": 96, "fail": 23, "infra": 0}}
    p = apply_cap(mix(failure_rates(stats), PI0, 0.4), PI0)
    assert tuple(round(p[c], 6) for c in CELLS) == (0.253571, 0.460261, 0.286169)
    assert tuple(allocate(p)[c] for c in CELLS) == (10, 18, 12)
    for seed, expected in ((42, (9, 19, 12)), (137, (12, 19, 9))):
        path = f"artifacts/self_improve/r4/warmup_records/seed_{seed}.jsonl"
        records = [json.loads(line) for line in open(path, encoding="utf-8")]
        assert len(records) == 320
        warmup_stats = cell_stats(records, (1, 20))
        distribution = stage_distribution_record(train_seed=seed, window=(1, 20),
            input_record_refs=[path], stats=warmup_stats, rho=0.4)
        assert tuple(distribution["cells"][c]["count"] for c in CELLS) == expected
        fixed = stage_distribution_record(train_seed=seed, window=(1, 20),
            input_record_refs=[path], stats=warmup_stats, rho=1.0)
        assert tuple(fixed["cells"][c]["count"] for c in CELLS) == (15, 13, 12)


def test_a2_stage_rows_are_interleaved_and_cross_process_identical():
    code = """import json
from src.curriculum.failure_driven import CELLS, build_stage_rows
counts=dict(zip(CELLS,(9,19,12)))
orders={c:[(c,i) for i in range(counts[c])] for c in CELLS}
rows,_=build_stage_rows(counts,orders,dict.fromkeys(CELLS,0),train_seed=SEED,stage_index=1)
print(json.dumps(rows,separators=(',',':')))"""
    for seed in (42, 137):
        script = code.replace("SEED", str(seed))
        a = subprocess.check_output([sys.executable, "-c", script])
        b = subprocess.check_output([sys.executable, "-c", script])
        assert hashlib.sha256(a).hexdigest() == hashlib.sha256(b).hexdigest()
        cells = [row[0] for row in json.loads(a)]
        runs = [1]
        for previous, current in zip(cells, cells[1:]):
            runs.append(runs[-1] + 1 if previous == current else 1)
        assert max(runs) <= 8
        assert stage_permutation_seed(seed, 1) == int(hashlib.sha256(f"r4|{seed}|stage|1".encode()).hexdigest()[:8], 16)
