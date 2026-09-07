"""Validated, append-only experiment provenance; no learning framework types."""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any

from jsonschema import Draft202012Validator


def obj(properties: dict[str, Any]) -> dict[str, Any]:
    return {"type": "object", "properties": properties, "required": list(properties), "additionalProperties": False}


TEXT = {"type": "string", "minLength": 1}
RAW = {"type": "string"}
INT = {"type": "integer"}
NAT = {"type": "integer", "minimum": 0}
NUM = {"type": "number"}
NULL_TEXT = {"type": ["string", "null"]}
HASH = {"type": "string", "pattern": "^[0-9a-f]{64}$"}
DIFFICULTY = obj({"source": {"enum": ["mutation", "commit-rollback", "feat-add"]}, "type": TEXT, "count": {"enum": [1, 2, 3]}, "hint": {"enum": ["L0", "L1", "L2"]}, "span": {"enum": ["single-function", "single-file", "cross-file"]}})
TOKENS = obj({"input": NAT, "output": NAT, "tool": NAT})
SEGMENT = obj({"name": TEXT, "start": NUM, "end": NUM})
TEST_RESULT = {"anyOf": [{"type": "null"}, obj({"exit_code": INT, "passed": NAT, "failed": NAT, "stdout": RAW, "stderr": RAW})]}
TOOL = obj({"name": TEXT, "arguments": {"type": "object"}, "stdout": RAW, "stderr": RAW, "exit_code": INT})
TURN = obj({"index": NAT, "start": NUM, "end": NUM, "assistant": RAW, "tool_calls": {"type": "array", "items": TOOL}, "run_tests": TEST_RESULT, "token_counts": TOKENS})
TRAJECTORY_SCHEMA = {"$schema": "https://json-schema.org/draft/2020-12/schema", "$id": "https://octorl.local/schema/trajectory-v1.json", **obj({
    "schema_version": {"const": 1}, "trajectory_id": TEXT, "task_id": TEXT, "difficulty": DIFFICULTY,
    "injector_seed": INT, "sampling_seed": INT, "repo_ref": TEXT, "manifest_ref": TEXT, "container_id": TEXT,
    "policy_version_at_generation": TEXT, "policy_version_at_update": TEXT,
    "behavior_logprobs": {"type": "array", "items": NUM},
    "policy_lineage": obj({"base_model_revision": TEXT, "adapter_id": NULL_TEXT, "consolidation_generation": NAT}),
    "turns": {"type": "array", "items": TURN}, "token_counts": TOKENS,
    "wall_clock_segments": {"type": "array", "items": SEGMENT},
    "context_augmentations": {"type": "array", "maxItems": 0}, "external_memory_active": {"const": False},
    "harness_version": HASH,
    "provenance": obj({"episode_id": NULL_TEXT, "rule_id": NULL_TEXT, "alpha": {"type": ["number", "null"]}, "beta": {"type": ["number", "null"]}, "evidence_ids": {"type": ["array", "null"], "items": TEXT}, "model_id": NULL_TEXT}),
    "reward": {"enum": [0, 1]}, "cheat_flags": {"type": "array", "items": TEXT},
})}
MANIFEST_SCHEMA = obj({"schema_version": {"const": 1}, "manifest_id": TEXT, "manifest_role": {"enum": ["train", "dev", "audit", "heldout"]}, "instances": {"type": "array", "minItems": 1, "items": obj({"task_id": TEXT, "repo_ref": TEXT, "injector_seed": INT, "difficulty": DIFFICULTY})}})
REGISTRY_SCHEMA = obj({"config_hash": HASH, "seeds": {"type": "array", "minItems": 1, "items": INT}, "pinned_commits": {"type": "object", "minProperties": 1, "additionalProperties": TEXT}, "manifest_ids": {"type": "array", "items": TEXT}, "artifact_paths": {"type": "array", "items": TEXT}, "wandb_run": NULL_TEXT, "harness_version": HASH, "harness_diff_ref": {"type": "null"}})
ACCEPTOR_SCHEMA = obj({"candidate_version": HASH, "incumbent_version": HASH, "acceptor": TEXT, "dev_scores": {"type": "array", "items": NUM}, "decision": {"enum": ["accept", "reject", "indeterminate"]}, "eprocess_wealth_trace": {"type": "array", "items": NUM}, "audit_result": {"type": ["object", "null"]}})


def canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)


def content_hash(value: Any) -> str:
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def validate(value: dict[str, Any], schema: dict[str, Any] = TRAJECTORY_SCHEMA) -> None:
    canonical(value)  # Reject NaN/Infinity, which JSON Schema considers numbers.
    Draft202012Validator(schema).validate(value)
    for segment in value.get("wall_clock_segments", []) + value.get("turns", []):
        if segment["end"] < segment["start"]:
            raise ValueError("wall-clock interval ends before it starts")


def append_record(path: Path | str, record: dict[str, Any], schema: dict[str, Any] = TRAJECTORY_SCHEMA) -> None:
    validate(record, schema)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # One append write keeps each record contiguous for concurrent local writers.
    import os
    fd = os.open(path, os.O_APPEND | os.O_CREAT | os.O_WRONLY, 0o644)
    try:
        payload = (canonical(record) + "\n").encode()
        if os.write(fd, payload) != len(payload):
            raise OSError("short JSONL write")
        os.fsync(fd)
    finally:
        os.close(fd)


def freeze_manifest(path: Path | str, manifest: dict[str, Any]) -> None:
    validate(manifest, MANIFEST_SCHEMA)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = canonical(manifest) + "\n"
    try:
        with path.open("x") as stream:
            stream.write(payload)
    except FileExistsError:
        if path.read_text() != payload:
            raise ValueError("frozen manifest cannot be replaced or reassigned") from None


def load_manifest(path: Path | str) -> dict[str, Any]:
    manifest = json.loads(Path(path).read_text())
    validate(manifest, MANIFEST_SCHEMA)
    return manifest


def harness_version(root: Path | str) -> str:
    root = Path(root)
    return content_hash({name: (root / name).read_text() for name in ("agent_loop.py", "context_builder.py", "presubmit_check.py", "prompt.md")})


def new_record(*, trajectory_id: str, task_id: str, difficulty: dict[str, Any], injector_seed: int, sampling_seed: int, repo_ref: str, manifest_ref: str, container_id: str, policy_version: str, base_model_revision: str, harness_root: Path | str, adapter_id: str | None = None) -> dict[str, Any]:
    return {"schema_version": 1, "trajectory_id": trajectory_id, "task_id": task_id, "difficulty": difficulty, "injector_seed": injector_seed, "sampling_seed": sampling_seed, "repo_ref": repo_ref, "manifest_ref": manifest_ref, "container_id": container_id, "policy_version_at_generation": policy_version, "policy_version_at_update": policy_version, "behavior_logprobs": [], "policy_lineage": {"base_model_revision": base_model_revision, "adapter_id": adapter_id, "consolidation_generation": 0}, "turns": [], "token_counts": {"input": 0, "output": 0, "tool": 0}, "wall_clock_segments": [], "context_augmentations": [], "external_memory_active": False, "harness_version": harness_version(harness_root), "provenance": {"episode_id": None, "rule_id": None, "alpha": None, "beta": None, "evidence_ids": None, "model_id": None}, "reward": 0, "cheat_flags": []}


def export_schemas(directory: Path | str) -> None:
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    for name, schema in [("trajectory-v1", TRAJECTORY_SCHEMA), ("manifest-v1", MANIFEST_SCHEMA), ("registry-v1", REGISTRY_SCHEMA), ("acceptor-v1", ACCEPTOR_SCHEMA)]:
        Draft202012Validator.check_schema(schema)
        (directory / f"{name}.json").write_text(json.dumps(schema, indent=2) + "\n")
