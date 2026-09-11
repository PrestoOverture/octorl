import os
from math import comb
from pathlib import Path

import pytest

from scripts.baseline_eval import satisfiable
from scripts.p2_d1_gap import (BUCKET_PLAN, SOURCES, annotate_rollout,
                               choose_task, cluster_bootstrap, gap, predictions,
                               prepare_catalog_history)
from src.injector.buckets import BUCKETS


def test_predictions_are_strictly_prequential():
    assert predictions([]) == {"B1_pred": None, "B_emp_pred": None, "B2_pred": None}
    before = predictions([0, 8])
    after = predictions([0, 8, 4])
    assert before["B_emp_pred"] == 0.0
    assert after["B_emp_pred"] == pytest.approx(1 / 3)


def test_rollout_failures_are_explicit_not_silent():
    good = annotate_rollout(2, {
        "termination": "stop", "turn_count": 2, "sampling_seeds": [11, 12],
        "reward": 0.0,
    })
    assert good["completed"] is True
    assert good["failure_kind"] is None
    assert good["rollout_index"] == 2

    failed = annotate_rollout(3, {
        "termination": "api_error", "turn_count": 0, "sampling_seeds": [],
        "reward": 0.0,
    })
    assert failed["completed"] is False
    assert failed["failure_kind"] == "api_sandbox_or_grader_exception"


def test_exact_homogeneous_binomial_control_has_zero_gap():
    # The complete Binomial(8, 0.5) population contains C(8, k) copies of k.
    # Its mean and mixed fraction equal their analytic values exactly.
    rows = []
    for k in range(9):
        rows.extend({"K": k, "is_mixed": 0 < k < 8} for _ in range(comb(8, k)))
    assert len(rows) == 256
    assert gap(rows) == pytest.approx(0.0, abs=1e-15)


def test_repo_cluster_bootstrap_preserves_exact_zero_gap_control():
    rows = []
    for repo in ("r0", "r1", "r2"):
        for k in range(9):
            rows.extend({"K": k, "is_mixed": 0 < k < 8,
                         "task": {"repo": repo}}
                        for _ in range(comb(8, k)))
    lo, hi = cluster_bootstrap(rows, "synthetic-null", B=200)
    assert lo == pytest.approx(0.0, abs=1e-15)
    assert hi == pytest.approx(0.0, abs=1e-15)


def test_task_sampling_is_stable_and_with_replacement():
    tasks = [
        {"bucket": "b", "repo": f"r{i}", "source": "mutation", "seed": i,
         "difficulty_type": "condition-inversion"}
        for i in range(3)
    ]
    first = [choose_task(tasks, "b", group) for group in range(1, 20)]
    second = [choose_task(tasks, "b", group) for group in range(1, 20)]
    assert first == second
    assert len({row["repo"] for row in first}) > 1
    assert len(first) > len({(row["repo"], row["seed"]) for row in first})


def test_catalog_history_replacements_validate_every_frozen_transition(tmp_path):
    old_git_dir = os.environ.get("GIT_DIR")
    try:
        mappings = prepare_catalog_history(tmp_path)
        assert len(mappings) >= 8
        assert all(len(original) == len(replacement) == 40
                   for original, replacement in mappings.items())
    finally:
        if old_git_dir is None:
            os.environ.pop("GIT_DIR", None)
        else:
            os.environ["GIT_DIR"] = old_git_dir


def test_structural_feasibility_is_seed_invariant(tmp_path):
    old_git_dir = os.environ.get("GIT_DIR")
    try:
        prepare_catalog_history(tmp_path)
        repo = Path(__file__).resolve().parents[1] / "repos/train/config_parser"
        wanted = {bucket.id: bucket for bucket in BUCKETS}
        for bucket_id in BUCKET_PLAN:
            for source in SOURCES:
                outcomes = [bool(satisfiable(repo, wanted[bucket_id], seed, source))
                            for seed in (0, 1, 49)]
                assert outcomes == [outcomes[0]] * 3
    finally:
        if old_git_dir is None:
            os.environ.pop("GIT_DIR", None)
        else:
            os.environ["GIT_DIR"] = old_git_dir
