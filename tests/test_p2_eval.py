from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest


ROOT = Path(__file__).resolve().parents[1]


def _load_script(name: str):
    path = ROOT / "scripts" / name
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_compile_bucket_table_rates_and_splits():
    module = _load_script("p2_bucket_table.py")
    data = {
        "model": {"rollouts_per_instance": 8},
        "specs": [
            {"repo_name": "train_repo", "bucket": module.EASY_BUCKET,
             "repo": "/tmp/repos/train/train_repo"},
            {"repo_name": "held_repo", "bucket": module.EASY_BUCKET,
             "repo": "/tmp/repos/held_out/held_repo"},
        ],
        "pass_at_1": {
            f"train_repo|{module.EASY_BUCKET}": {
                "rollouts": 8, "mean_reward": 0.25, "mixed_group": True},
            f"held_repo|{module.EASY_BUCKET}": {
                "rollouts": 8, "mean_reward": 0.0, "mixed_group": False},
        },
    }
    result = module.compile_bucket_table(data)
    easy = result["easy_bucket"]
    assert easy["pass_at_1"] == 0.125
    assert easy["pass_at_8"] == 0.5
    assert easy["mixed_group_ratio"] == 0.5
    assert (easy["n_instances"], easy["n_train"], easy["n_held_out"]) == (2, 1, 1)


def test_assistant_turn_format_requires_tools_and_valid_json():
    module = _load_script("baseline_eval.py")

    no_tools = SimpleNamespace(tool_calls=None)
    valid_tools = SimpleNamespace(tool_calls=[
        SimpleNamespace(function=SimpleNamespace(arguments='{"path": "."}')),
        SimpleNamespace(function=SimpleNamespace(arguments='{}')),
    ])
    one_invalid = SimpleNamespace(tool_calls=[
        SimpleNamespace(function=SimpleNamespace(arguments='{"path": "."}')),
        SimpleNamespace(function=SimpleNamespace(arguments='{broken')),
    ])

    assert module.assistant_turn_format(no_tools) == {
        "total_turns": 1, "tool_call_turns": 0, "valid_json_turns": 0}
    assert module.assistant_turn_format(valid_tools) == {
        "total_turns": 1, "tool_call_turns": 1, "valid_json_turns": 1}
    assert module.assistant_turn_format(one_invalid) == {
        "total_turns": 1, "tool_call_turns": 1, "valid_json_turns": 0}


def test_derive_seed_is_stable_and_turn_specific():
    module = _load_script("baseline_eval.py")
    first = module.derive_seed(20260910, "repo|bucket|11", 2, 3)
    assert first == module.derive_seed(20260910, "repo|bucket|11", 2, 3)
    assert 0 <= first < 2**31
    assert first != module.derive_seed(20260910, "repo|bucket|11", 2, 4)
    assert first != module.derive_seed(20260910, "repo|bucket|11", 3, 3)


def test_request_args_are_explicit_and_seed_is_optional():
    module = _load_script("baseline_eval.py")
    common = dict(top_k=20, top_p=0.95, task_key="repo|bucket|11",
                  rollout_index=2, turn_index=3)
    unseeded, no_seed = module.build_request_args(
        "model", [], sampling_master_seed=None, **common)
    assert no_seed is None and "seed" not in unseeded
    assert unseeded["temperature"] == 1.0
    assert unseeded["top_p"] == 0.95
    assert unseeded["extra_body"] == {"top_k": 20}

    seeded, seed = module.build_request_args(
        "model", [], sampling_master_seed=20260910, **common)
    assert seeded["seed"] == seed == module.derive_seed(
        20260910, "repo|bucket|11", 2, 3)


def test_checkpoint_metadata_and_compatibility(tmp_path, capsys):
    module = _load_script("baseline_eval.py")
    path = tmp_path / "eval.ckpt"
    metadata = {
        "model_path": "model", "request_params": module.request_params(20, 0.95),
        "enforce_eager": True,
    }
    assert module.load_checkpoint(path, metadata) == {}
    assert json.loads(path.read_text().splitlines()[0]) == {"run_metadata": metadata}

    with path.open("a") as stream:
        stream.write(json.dumps({"key": "repo|bucket|11", "rollout_details": []}) + "\n")
    assert "repo|bucket|11" in module.load_checkpoint(path, metadata)

    mismatched = {**metadata, "enforce_eager": False}
    with pytest.raises(ValueError, match="enforce_eager"):
        module.load_checkpoint(path, mismatched)

    path.write_text(json.dumps({"key": "old", "rewards": [0.0]}) + "\n")
    assert module.load_checkpoint(path, metadata) == {}
    assert "deprecated checkpoint without run_metadata" in capsys.readouterr().err
