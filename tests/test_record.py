import copy
import json
from pathlib import Path

import pytest
from jsonschema import ValidationError

from src.environment.record import (
    ACCEPTOR_SCHEMA, MANIFEST_SCHEMA, REGISTRY_SCHEMA, TRAJECTORY_SCHEMA,
    append_record, content_hash, export_schemas, freeze_manifest, harness_version,
    load_manifest, new_record, validate,
)
from harness.context_builder import build_prompt

ROOT = Path(__file__).resolve().parents[1]


def example():
    return new_record(trajectory_id="test-1", task_id="toy-1", difficulty={"source": "mutation", "type": "off-by-one", "count": 1, "hint": "L0", "span": "single-function"}, injector_seed=20260906, sampling_seed=20260906, repo_ref="sha256:fixture", manifest_ref="test-manifest", container_id="mock-test", policy_version="mock-1", base_model_revision="mock-fixture", harness_root=ROOT / "harness")


def test_required_fields_and_finite_numbers(tmp_path):
    record = example()
    validate(record)
    for name in TRAJECTORY_SCHEMA["required"]:
        broken = copy.deepcopy(record)
        del broken[name]
        with pytest.raises(ValidationError):
            validate(broken)
    record["behavior_logprobs"] = [float("nan")]
    with pytest.raises(ValueError):
        validate(record)


def test_raw_outputs_roundtrip(tmp_path):
    record = example()
    raw = "hello\x00\n\r\t中文\n"
    record["turns"] = [{"index": 0, "start": 1.0, "end": 2.0, "assistant": "{}", "tool_calls": [{"name": "read_file", "arguments": {"path": "a.py"}, "stdout": raw, "stderr": "error\n", "exit_code": 0}], "run_tests": None, "token_counts": {"input": 3, "output": 2, "tool": 8}}]
    append_record(tmp_path / "records.jsonl", record)
    assert json.loads((tmp_path / "records.jsonl").read_text())["turns"][0]["tool_calls"][0]["stdout"] == raw


def test_manifest_immutable_role_and_registry(tmp_path):
    record = example()
    manifest = {"schema_version": 1, "manifest_id": "m1", "manifest_role": "heldout", "instances": [{"task_id": record["task_id"], "repo_ref": record["repo_ref"], "injector_seed": record["injector_seed"], "difficulty": record["difficulty"]}]}
    path = tmp_path / "manifest.json"
    freeze_manifest(path, manifest)
    assert load_manifest(path) == manifest
    freeze_manifest(path, manifest)
    manifest["manifest_role"] = "train"
    with pytest.raises(ValueError):
        freeze_manifest(path, manifest)
    registry = {"config_hash": content_hash({"seed": 20260906}), "seeds": [20260906], "pinned_commits": {"test": "fixture"}, "manifest_ids": ["m1"], "artifact_paths": [str(path)], "wandb_run": None, "harness_version": record["harness_version"], "harness_diff_ref": None}
    append_record(tmp_path / "registry.jsonl", registry, REGISTRY_SCHEMA)


def test_prompt_replay_and_hash_changes(tmp_path):
    instance = {"task_id": "a", "issue": "fix boundary"}
    assert build_prompt(instance, [], 20260906) == build_prompt(instance, [], 20260906)
    import shutil
    shutil.copytree(ROOT / "harness", tmp_path / "harness")
    before = harness_version(tmp_path / "harness")
    with (tmp_path / "harness/prompt.md").open("a") as out:
        out.write("\nAdditional instruction\n")
    assert before != harness_version(tmp_path / "harness")
    with pytest.raises(TypeError):
        build_prompt(instance, [], None)


def test_exported_definitions_validate(tmp_path):
    export_schemas(tmp_path)
    assert len(list(tmp_path.glob("*.json"))) == 4
    assert json.loads((tmp_path / "trajectory-v1.json").read_text()) == TRAJECTORY_SCHEMA
