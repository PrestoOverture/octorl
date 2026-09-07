"""P1.21: end-to-end integration over the real sandbox, plus the P1.31
regeneration round-trip. Docker-backed tests skip when no daemon is reachable."""
from __future__ import annotations

import ast
import copy
import difflib
import json
import shutil
import subprocess
from pathlib import Path

import pytest

from harness.context_builder import build_prompt
from src.environment.harness import (
    ScriptedPolicy, affected_tests_for, package, regenerate, repo_ref, run_trajectory,
)
from src.environment.record import TRAJECTORY_SCHEMA, harness_version, validate
from src.environment.sandbox import create_pool, docker_available
from src.injector.core import Difficulty

ROOT = Path(__file__).resolve().parents[1]
IMAGE = (ROOT / "artifacts/p1/docker-image.id").read_text().strip()
REPO = ROOT / "repos/train/record_index"
# mutation is the only source that works on this repo; see the P1.20 handoff.
EASY = Difficulty("mutation", "condition-inversion", 1, "L0", "single-function")
HARD = Difficulty("mutation", "condition-inversion", 2, "L0", "single-file")


needs_docker = pytest.mark.skipif(not docker_available(), reason="docker daemon unavailable")


@pytest.fixture(scope="module")
def pool():
    with create_pool(IMAGE, size=1) as p:
        yield p


@pytest.fixture
def instance(tmp_path):
    return package(REPO, EASY, 11, tmp_path)


def unified(path: str, before: str, after: str) -> str:
    return "".join(difflib.unified_diff(
        before.splitlines(keepends=True), after.splitlines(keepends=True),
        fromfile=f"a/{path}", tofile=f"b/{path}"))


def constant_backfill(source: str) -> str:
    """Rewrite the first module-level function's body to a bare constant."""
    tree = ast.parse(source)
    lines = source.splitlines(keepends=True)
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.end_lineno > node.body[0].lineno:
            indent = " " * (len(lines[node.body[0].lineno - 1]) - len(lines[node.body[0].lineno - 1].lstrip()))
            return "".join(lines[:node.body[0].lineno - 1] + [f"{indent}return 42\n"] + lines[node.end_lineno:])
    raise AssertionError("no suitable function to backfill")


def drive(instance, pool, calls, *, seed=11, trajectory_id="t-1"):
    with pool.lease(instance.root) as box:
        return run_trajectory(
            instance, box, ScriptedPolicy(calls), sampling_seed=seed, trajectory_id=trajectory_id,
            policy_version="scripted-1", base_model_revision="none", manifest_ref="test-manifest")


def repair_calls(instance, source_for):
    path = instance.edits[0].path
    return [
        {"name": "list_files", "arguments": {"path": "."}},
        {"name": "read_file", "arguments": {"path": path}},
        {"name": "run_tests", "arguments": {}},
        {"name": "apply_patch", "arguments": {"diff": unified(path, instance.modified[path], source_for(path))}},
        {"name": "run_tests", "arguments": {}},
    ]


# ── end-to-end reward path ──

@needs_docker
def test_restoring_the_original_earns_reward(instance, pool):
    record = drive(instance, pool, repair_calls(instance, lambda p: instance.originals[p]))
    assert record["reward"] == 1, record["turns"][-1]["tool_calls"][0]["stdout"][-600:]
    assert record["cheat_flags"] == []


@needs_docker
def test_cosmetic_edit_earns_nothing(instance, pool):
    path = instance.edits[0].path
    record = drive(instance, pool, repair_calls(instance, lambda p: "# review pending\n" + instance.modified[p]),
                   trajectory_id="t-2")
    assert record["reward"] == 0
    assert record["cheat_flags"] == []
    assert any(t["tool_calls"][0]["name"] == "apply_patch" and t["tool_calls"][0]["exit_code"] == 0
               for t in record["turns"]), "the patch must have applied; otherwise this proves nothing"


@needs_docker
def test_constant_backfill_is_flagged_end_to_end(instance, pool):
    record = drive(instance, pool, repair_calls(instance, lambda p: constant_backfill(instance.originals[p])),
                   trajectory_id="t-3")
    assert record["reward"] == 0
    assert any("constant_backfill" in f or "literal_replacement" in f for f in record["cheat_flags"]), record["cheat_flags"]


# ── record integrity ──

@needs_docker
def test_every_record_validates_and_carries_live_provenance(instance, pool):
    record = drive(instance, pool, repair_calls(instance, lambda p: instance.originals[p]), trajectory_id="t-4")
    validate(record, TRAJECTORY_SCHEMA)
    assert record["harness_version"] == harness_version(ROOT / "harness")
    assert record["context_augmentations"] == []
    assert record["external_memory_active"] is False
    assert record["turns"] and all(t["end"] >= t["start"] for t in record["turns"])
    assert record["token_counts"]["input"] > 0 and record["token_counts"]["tool"] > 0


@needs_docker
def test_presubmit_check_runs_before_submission(instance, pool):
    """Policy stops right after patching; the harness pre-submit choice must
    still put a run_tests turn on the record."""
    path = instance.edits[0].path
    calls = [{"name": "apply_patch",
              "arguments": {"diff": unified(path, instance.modified[path], instance.originals[path])}}]
    record = drive(instance, pool, calls, trajectory_id="t-5")
    names = [t["tool_calls"][0]["name"] for t in record["turns"]]
    assert names == ["apply_patch", "run_tests"], names
    assert record["turns"][-1]["run_tests"] is not None


# ── the harness/ surface is load-bearing ──

def test_prompt_construction_flows_through_harness_files(tmp_path):
    shutil.copytree(ROOT / "harness", tmp_path / "harness")
    before_version = harness_version(tmp_path / "harness")
    before_prompt = build_prompt({"task_id": "a"}, [], 11, prompt_path=tmp_path / "harness/prompt.md")
    (tmp_path / "harness/prompt.md").write_text("Totally different instructions.\n")
    assert harness_version(tmp_path / "harness") != before_version
    after_prompt = build_prompt({"task_id": "a"}, [], 11, prompt_path=tmp_path / "harness/prompt.md")
    assert after_prompt[0]["content"] != before_prompt[0]["content"]


# ── task_metadata.json drives test selection ──

@needs_docker
def test_affected_tests_drive_run_tests_selection(instance, pool):
    path = instance.edits[0].path
    expected = instance.affected_tests[path]
    assert (ROOT / "repos/train/record_index/tests/test_record_index.py").exists()
    assert expected and all("::" in t for t in expected), expected

    with pool.lease(instance.root) as box:
        assert json.loads((box.root / "task_metadata.json").read_text())["affected_tests"] == instance.affected_tests
        seen = []
        original_run = box.run

        def spy(argv, **kwargs):
            seen.append(argv)
            return original_run(argv, **kwargs)

        box.run = spy
        calls = [{"name": "apply_patch",
                  "arguments": {"diff": unified(path, instance.modified[path], instance.originals[path])}},
                 {"name": "run_tests", "arguments": {}}]
        run_trajectory(instance, box, ScriptedPolicy(calls), sampling_seed=11, trajectory_id="t-6",
                       policy_version="scripted-1", base_model_revision="none", manifest_ref="test-manifest")
        selected = [a for a in seen if "pytest" in a and not any(x.startswith("hidden_tests") or x == "tests/" for x in a)]
        assert selected, seen
        assert [a for a in selected[0] if a.endswith(".py") or "::" in a] == expected


def test_affected_tests_are_narrower_than_the_whole_module(instance):
    path = instance.edits[0].path
    every = sorted(
        f"tests/{p.name}::{n.name}"
        for p in (instance.root / "tests").glob("test_*.py")
        for n in ast.walk(ast.parse(p.read_text()))
        if isinstance(n, ast.FunctionDef) and n.name.startswith("test"))
    assert 0 < len(instance.affected_tests[path]) < len(every)


def test_missing_metadata_falls_back_to_whole_modules(tmp_path):
    """Guard the fallback path used when no test references the edited symbol."""
    instance = package(REPO, EASY, 11, tmp_path)
    assert affected_tests_for(instance.root, instance.edits) == instance.affected_tests
    class Anon:
        path, function = instance.edits[0].path, "no_such_symbol_anywhere"
    assert affected_tests_for(instance.root, [Anon()])[Anon.path] == ["tests/test_record_index.py"]


# ── packaging hygiene ──

def test_instance_never_ships_the_injector_catalog(instance):
    names = {p.relative_to(instance.root).as_posix() for p in instance.root.rglob("*") if p.is_file()}
    assert "injector_sources.json" not in names, "the catalog records the fix verbatim"
    assert not any(n.startswith(".git") for n in names)
    assert "task_metadata.json" in names
    assert json.loads((instance.root / "task_metadata.json").read_text()).keys() == {
        "task_id", "issue", "affected_tests"}


def test_packaging_does_not_mutate_the_source_repo(tmp_path):
    before = repo_ref(REPO)
    package(REPO, EASY, 11, tmp_path)
    assert repo_ref(REPO) == before


# ── P1.31 regeneration round-trip ──

@pytest.mark.parametrize("repo_name", ["record_index", "slot_planner", "config_parser"])
@pytest.mark.parametrize("difficulty", [EASY, HARD], ids=["c1-single-function", "c2-single-file"])
def test_frozen_triple_regenerates_byte_identical_sources(tmp_path, repo_name, difficulty):
    repo = ROOT / "repos/train" / repo_name
    first = package(repo, difficulty, 20260906, tmp_path / "a")
    spec = {"task_id": first.task_id, "repo_ref": first.repo_ref,
            "injector_seed": first.injector_seed, "difficulty": {
                "source": difficulty.source, "type": difficulty.type, "count": difficulty.count,
                "hint": difficulty.hint, "span": difficulty.span}}
    second = regenerate(spec, ROOT / "repos", tmp_path / "b")
    assert second.modified == first.modified
    assert second.originals == first.originals
    assert second.task_id == first.task_id
    files = lambda i: {p.relative_to(i.root).as_posix(): p.read_bytes()
                       for p in sorted(i.root.rglob("*")) if p.is_file()}
    assert files(second) == files(first)


def test_regeneration_refuses_a_drifted_repository(tmp_path):
    instance = package(REPO, EASY, 11, tmp_path / "a")
    spec = {"task_id": instance.task_id, "repo_ref": f"record_index@sha256:{'0' * 64}",
            "injector_seed": 11, "difficulty": {"source": "mutation", "type": "condition-inversion",
                                                "count": 1, "hint": "L0", "span": "single-function"}}
    with pytest.raises(ValueError, match="drifted"):
        regenerate(spec, ROOT / "repos", tmp_path / "b")
