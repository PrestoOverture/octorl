import json
from pathlib import Path

from scripts.self_improve.r6_offline_replay import MockBackend, run_replay
from src.curriculum.failure_driven import PI0


def test_mock_constant_has_zero_input_sensitivity(tmp_path):
    metrics = run_replay(Path.cwd(), MockBackend("constant"), tmp_path, deterministic_wall_time=True)
    assert metrics["input_sensitivity"] == 0


def test_mock_proportional_real_windows_detect_input_sensitivity_and_is_deterministic(tmp_path):
    first_dir, second_dir = tmp_path / "first", tmp_path / "second"
    first = run_replay(Path.cwd(), MockBackend("proportional"), first_dir, deterministic_wall_time=True)
    second = run_replay(Path.cwd(), MockBackend("proportional"), second_dir, deterministic_wall_time=True)
    assert first["input_sensitivity"] > 0.10
    assert first["G1"]["pass"]
    assert (first_dir / "metrics.json").read_bytes() == (second_dir / "metrics.json").read_bytes()
    assert first["query_count"]["total"] == 2304


def test_mock_garbage_every_aggregate_falls_back_to_pi0(tmp_path):
    metrics = run_replay(Path.cwd(), MockBackend("garbage"), tmp_path, deterministic_wall_time=True)
    assert metrics["fallback_stages"] == metrics["sample_stage_count"] == 36 * 3
    assert metrics["fallback_q_all_pi0"]
    records = [json.loads(line) for line in (tmp_path / "completions.jsonl").read_text().splitlines()]
    assert not any(record["valid"] for record in records)
    # The aggregate fallback itself is tested directly in test_r6_chooser; this
    # confirms all replay groups exercised that path.
    assert PI0 == {"constraint_violation": .375, "missing_dependency": .3125, "stale_version": .3125}
