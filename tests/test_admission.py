"""F1 admission checks and F6 provenance pinning.

Admission judges a fixed instance against the real grader: the clean tree must
pass, the buggy tree must produce a genuine test *failure*, the known repair must
pass, and for multi-edit tasks each advertised defect must independently matter.
"""
from __future__ import annotations

import dataclasses
import json
import shutil
from pathlib import Path

import pytest

from src.environment.harness import (
    ADMISSION_RULE_VERSION, admission_report, admit, freeze_admitted_manifest, package,
    repo_ref,
)
from src.environment.sandbox import create_pool, docker_available
from src.injector.core import Difficulty, Edit
from tests.reward_fixtures import BUGGY, CLEAN, SUITE, instance as make_instance

ROOT = Path(__file__).resolve().parents[1]
IMAGE = (ROOT / "artifacts/p1/docker-image.id").read_text().strip()
REPO = ROOT / "repos/train/record_index"
needs_docker = pytest.mark.skipif(not docker_available(), reason="docker daemon unavailable")


@pytest.fixture(scope="module")
def pool():
    with create_pool(IMAGE, size=1) as value:
        yield value


def retarget(instance, **changes):
    return dataclasses.replace(instance, **changes)


# ── the four admission conditions ──

@needs_docker
def test_a_well_formed_instance_is_admitted(tmp_path, pool):
    row = admit(make_instance(tmp_path), pool)
    assert row["admitted"] is True, row["reasons"]
    assert row["reasons"] == []
    assert set(row["checks"]) == {"clean", "buggy", "repair"}
    assert row["checks"]["clean"]["reward"] == 1
    assert row["checks"]["buggy"]["reward"] == 0
    assert row["admission_rule_version"] == ADMISSION_RULE_VERSION


@needs_docker
def test_condition_2_rejects_an_undetected_defect(tmp_path, pool):
    """The F1 degenerate class: a 'bug' no test exercises."""
    instance = make_instance(tmp_path)
    untested = CLEAN + "\ndef never_tested(z):\n    return z\n"
    (instance.root / "subject.py").write_text(untested.replace("return z", "return z + 1"))
    row = admit(retarget(instance,
                         originals={"subject.py": untested},
                         modified={"subject.py": untested.replace("return z", "return z + 1")}), pool)
    assert row["admitted"] is False
    assert "buggy:no_genuine_test_failure" in row["reasons"]


@needs_docker
def test_condition_2_rejects_a_collection_error_as_evidence(tmp_path, pool):
    """A tree that fails to import is not a 'detected defect'."""
    instance = make_instance(tmp_path)
    broken = CLEAN + "\nthis is not python\n"
    row = admit(retarget(instance, modified={"subject.py": broken}), pool)
    assert row["admitted"] is False
    assert any("buggy" in r for r in row["reasons"]), row["reasons"]


@needs_docker
def test_condition_1_rejects_a_clean_tree_that_already_fails(tmp_path, pool):
    instance = make_instance(tmp_path)
    already_failing = CLEAN.replace("return x * 2", "return x * 3")
    row = admit(retarget(instance,
                         originals={"subject.py": already_failing},
                         modified={"subject.py": already_failing.replace("x + y", "x - y")}), pool)
    assert row["admitted"] is False
    assert any("clean:" in r for r in row["reasons"]), row["reasons"]


# ── condition 4: each advertised defect must independently matter ──

def two_edit_instance(tmp_path, second_edit_matters: bool):
    """count=2 instance whose second edit does or does not affect any test."""
    clean = CLEAN + "\ndef third(v):\n    return v + 1\n"
    tested = "def test_e_third():\n    assert third(1) == 2\n"
    suite = SUITE.replace("from subject import target, other_func",
                          "from subject import target, other_func, third")
    suite = suite + (tested if second_edit_matters else "")
    root = tmp_path / "tree"
    (root / "tests").mkdir(parents=True)
    both_broken = clean.replace("x + y", "x - y").replace("return v + 1", "return v + 2")
    (root / "subject.py").write_text(both_broken)
    (root / "tests/test_visible.py").write_text(suite)
    hidden = tmp_path / "hidden"
    hidden.mkdir()
    (hidden / "test_hidden.py").write_text(suite)
    from src.environment.harness import TaskInstance
    return TaskInstance(
        "toy2:11", "toy", "toy@sha256:" + "2" * 64,
        Difficulty("mutation", "wrong-return-value", 2, "L0", "single-file"), 11,
        root, hidden, "Repair two defects", {"subject.py": clean}, {"subject.py": both_broken},
        [Edit("subject.py", "target", 2, "x + y", "x - y", "fixture"),
         Edit("subject.py", "third", 8, "v + 1", "v + 2", "fixture")],
        {"subject.py": ["tests/test_visible.py"]})


@needs_docker
def test_condition_4_admits_when_both_defects_matter(tmp_path, pool):
    row = admit(two_edit_instance(tmp_path, second_edit_matters=True), pool)
    assert row["admitted"] is True, row["reasons"]
    assert {"edit_0", "edit_1"} <= set(row["checks"])


@needs_docker
def test_condition_4_rejects_when_one_defect_does_not_matter(tmp_path, pool):
    """Without this check, a 'two-defect' task is silently a one-defect task —
    exactly the parcel_ledger cross-file shape from F3."""
    row = admit(two_edit_instance(tmp_path, second_edit_matters=False), pool)
    assert row["admitted"] is False
    assert "edit_1:no_genuine_test_failure" in row["reasons"], row["reasons"]


@needs_docker
def test_advertised_edit_count_must_match(tmp_path, pool):
    instance = two_edit_instance(tmp_path, second_edit_matters=True)
    row = admit(retarget(instance, edits=instance.edits[:1]), pool)
    assert row["admitted"] is False
    assert "advertised_edit_count_mismatch" in row["reasons"]


# ── the admitted population is recorded, never silently resampled ──

@needs_docker
def test_manifest_freezing_is_admission_gated_and_reports_rejections(tmp_path, pool):
    good = make_instance(tmp_path / "good")
    bad = two_edit_instance(tmp_path / "bad", second_edit_matters=False)
    report_path = tmp_path / "admission_report.json"
    manifest = freeze_admitted_manifest(
        tmp_path / "m.json", [good, bad], pool,
        manifest_id="m1", manifest_role="dev", report_path=report_path)
    assert [i["task_id"] for i in manifest["instances"]] == [good.task_id]
    report = json.loads(report_path.read_text())
    assert (report["requested"], report["admitted"], report["rejected"]) == (2, 1, 1)
    assert report["admission_rule_version"] == ADMISSION_RULE_VERSION
    assert report["admitted_proportions"]["source"]["mutation"]["proportion"] == 1.0


@needs_docker
def test_freezing_refuses_when_nothing_is_admissible(tmp_path, pool):
    bad = two_edit_instance(tmp_path / "bad", second_edit_matters=False)
    with pytest.raises(ValueError, match="no_admitted_instances"):
        freeze_admitted_manifest(tmp_path / "m.json", [bad], pool, manifest_id="m2",
                                 manifest_role="dev", report_path=tmp_path / "r.json")


def test_admission_report_counts_reasons_without_a_sandbox():
    rows = [{"admitted": True, "repo": "a", "reasons": [],
             "difficulty": {"source": "mutation", "type": "off-by-one", "count": 1,
                            "hint": "L0", "span": "single-function"}},
            {"admitted": False, "repo": "a", "reasons": ["buggy:no_genuine_test_failure"],
             "difficulty": {"source": "mutation", "type": "off-by-one", "count": 1,
                            "hint": "L0", "span": "single-function"}}]
    report = admission_report(rows)
    assert report["rejected"] == 1
    cell = report["by_cell"]["a|off-by-one|mutation|c1-L0-single-function"]
    assert cell == {"admitted": 1, "rejected": 1, "reasons": {"buggy:no_genuine_test_failure": 1}}


# ── F6: provenance actually covers what regeneration depends on ──

@pytest.mark.parametrize("mutate", ["catalog", "hidden_generator", "admission_rule"])
def test_repo_ref_covers_regeneration_inputs(tmp_path, monkeypatch, mutate):
    """Repairing catalogs or hidden tests must not silently change regenerated
    instances while leaving the fingerprint identical."""
    repo = tmp_path / "record_index"
    shutil.copytree(REPO, repo)
    before = repo_ref(repo)
    if mutate == "catalog":
        catalog = json.loads((repo / "injector_sources.json").read_text())
        catalog["history"] = catalog["history"][:-1]
        (repo / "injector_sources.json").write_text(json.dumps(catalog))
    elif mutate == "hidden_generator":
        import src.environment.harness as harness_module
        patched = dict(harness_module.REPO_GENERATORS)
        patched["record_index"] = patched["record_index"] + "\n# extra property\n"
        monkeypatch.setattr(harness_module, "REPO_GENERATORS", patched)
    else:
        import src.environment.harness as harness_module
        monkeypatch.setattr(harness_module, "ADMISSION_RULE_VERSION", "changed-rule-v2")
    assert repo_ref(repo) != before


def test_verification_seed_is_instance_scoped_not_rollout_scoped(tmp_path):
    """All G=8 rollouts of one group must grade against identical examples."""
    instance = package(REPO, Difficulty("mutation", "condition-inversion", 1, "L0",
                                        "single-function"), 11, tmp_path / "a")
    assert len({instance.verification_seed for _ in range(8)}) == 1
    same = package(REPO, Difficulty("mutation", "condition-inversion", 1, "L0",
                                    "single-function"), 11, tmp_path / "b")
    assert same.verification_seed == instance.verification_seed
    other = package(REPO, Difficulty("mutation", "condition-inversion", 1, "L0",
                                     "single-function"), 12, tmp_path / "c")
    assert other.verification_seed != instance.verification_seed


@needs_docker
def test_rollout_seed_still_varies_the_trajectory(tmp_path, pool):
    """The instance-scoped verification seed must not freeze the rollout itself."""
    from src.environment.harness import ScriptedPolicy, run_trajectory
    instance = package(REPO, Difficulty("mutation", "condition-inversion", 1, "L0",
                                        "single-function"), 11, tmp_path)
    seeds = []
    for sampling_seed in (11, 12):
        with pool.lease(instance.root) as box:
            record = run_trajectory(instance, box, ScriptedPolicy([
                {"name": "list_files", "arguments": {"path": "."}}]),
                sampling_seed=sampling_seed, trajectory_id=f"t{sampling_seed}",
                policy_version="p", base_model_revision="none", manifest_ref="m")
        seeds.append(record["sampling_seed"])
    assert seeds == [11, 12]
    assert instance.verification_seed not in seeds
