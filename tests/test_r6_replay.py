import json
import subprocess
import sys
from pathlib import Path

import pytest

from scripts.self_improve.r6_offline_replay import (
    MockBackend, canonical_metric, default_choosers, load_adapter_map, lora_request_id, run_g1,
    run_g2, run_smoke,
)
from src.curriculum.failure_driven import PI0


def _contains_tv_key(value):
    if isinstance(value, dict):
        return any("TV" in key or _contains_tv_key(item) for key, item in value.items())
    if isinstance(value, list):
        return any(_contains_tv_key(item) for item in value)
    return False


def _g1(tmp_path, mode):
    return run_g1(Path.cwd(), MockBackend(mode), tmp_path,
                  choosers=default_choosers(Path.cwd()), deterministic_wall_time=True)


def test_mock_constant_has_zero_input_sensitivity_and_no_tv_fields(tmp_path):
    metrics = _g1(tmp_path, "constant")
    assert metrics["input_sensitivity"] == 0
    assert not _contains_tv_key(metrics)
    assert not _contains_tv_key(json.loads((tmp_path / "g1_metrics.json").read_text()))
    assert not (tmp_path / "g2_metrics.json").exists()


def test_mock_proportional_real_windows_detect_input_sensitivity_and_is_deterministic(tmp_path):
    first_dir, second_dir = tmp_path / "first", tmp_path / "second"
    first = _g1(first_dir, "proportional")
    _g1(second_dir, "proportional")
    assert first["input_sensitivity"] > 0.10
    assert first["G1"]["pass"]
    assert (first_dir / "g1_metrics.json").read_bytes() == (second_dir / "g1_metrics.json").read_bytes()
    assert first["query_count"]["total"] == 2304
    assert set(first["sample_level_validity"]) == {
        "base", "u20_42", "u20_137", "u20_2718", "u80_42", "u80_137", "u80_2718",
    }


def test_mock_garbage_every_aggregate_falls_back_to_pi0(tmp_path):
    metrics = _g1(tmp_path, "garbage")
    assert sum(metrics["fallback_stage_counts"].values()) == metrics["sample_stage_count"] == 36 * 3
    records = [json.loads(line) for line in (tmp_path / "completions.jsonl").read_text().splitlines()]
    assert not any(record["valid"] for record in records)
    assert PI0 == {"constraint_violation": .375, "missing_dependency": .3125, "stale_version": .3125}


def test_g2_refuses_missing_or_failed_g1_and_cli_exits_nonzero(tmp_path):
    completions = tmp_path / "completions.jsonl"
    completions.write_text("")
    with pytest.raises(RuntimeError, match="missing sibling"):
        run_g2(completions, tmp_path / "g2")
    (tmp_path / "g1_metrics.json").write_text(json.dumps({"G1": {"pass": False}}))
    with pytest.raises(RuntimeError, match="not true"):
        run_g2(completions, tmp_path / "g2")
    process = subprocess.run([
        sys.executable, "scripts/self_improve/r6_offline_replay.py", "--phase", "g2",
        "--completions", str(completions), "--output-dir", str(tmp_path / "cli-g2"),
    ], capture_output=True, text=True)
    assert process.returncode != 0
    assert "G2 refused" in process.stderr


def test_g2_uses_saved_passing_g1_completions_only(tmp_path):
    g1 = tmp_path / "g1"
    _g1(g1, "proportional")
    metrics = run_g2(g1 / "completions.jsonl", tmp_path / "g2")
    assert set(metrics) == {"schema", "source_completions_sha256", "TV_identity", "TV_noise", "G2"}
    assert (tmp_path / "g2/g2_metrics.json").is_file()


def test_adapter_map_resolves_without_default_checkpoint_tree(tmp_path):
    mapping = {}
    for seed in (42, 137, 2718):
        mapping[str(seed)] = {}
        for kind in ("u20", "u80"):
            path = tmp_path / f"{kind}_{seed}"
            path.mkdir()
            (path / "adapter_model.safetensors").write_bytes(f"{kind}-{seed}".encode())
            mapping[str(seed)][kind] = str(path)
    path = tmp_path / "adapter-map.json"
    path.write_text(json.dumps(mapping))
    choosers = load_adapter_map(path)
    assert set(choosers) == {42, 137, 2718}
    assert all(len(value) == 2 for value in choosers.values())


def test_smoke_has_exactly_one_window_base_and_u20_adapter(tmp_path):
    metrics = run_smoke(Path.cwd(), MockBackend("constant"), tmp_path,
                        choosers=default_choosers(Path.cwd()), deterministic_wall_time=True)
    records = [json.loads(line) for line in (tmp_path / "completions.jsonl").read_text().splitlines()]
    assert metrics["query_count"] == len(records) == 16
    assert {(row["seed"], row["arm"], row["stage"]) for row in records} == {(137, "failure_driven", 1)}
    assert {row["chooser_id"] for row in records} == {"base", "u20_137"}
    assert {row["query_kind"] for row in records} == {"sample"}


def test_lora_request_id_is_deterministic_positive_int32():
    path = "/root/autodl-tmp/octorl_r3c/seed_137/checkpoints/global_step_20/actor/lora_adapter"
    assert lora_request_id(path) == lora_request_id(path)
    assert 0 < lora_request_id(path) <= 2**31 - 1


def test_canonical_metric_removes_platform_specific_tail_bits():
    assert canonical_metric(0.008333333333333333) == 0.0083333333333333
    assert canonical_metric(0.008333333333333331) == 0.0083333333333333
    assert canonical_metric(0.005208333333333333) == 0.0052083333333333
    assert canonical_metric(0.005208333333333329) == 0.0052083333333333
    assert canonical_metric(0.009895833333333335) == 0.0098958333333333
    assert canonical_metric(0.009895833333333336) == 0.0098958333333333
