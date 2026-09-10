from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace


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
