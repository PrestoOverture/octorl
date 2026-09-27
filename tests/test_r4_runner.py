"""CPU gates for the R4 stage runner and attribution join."""

import hashlib
import json
import subprocess
import sys

import pandas as pd
import pytest

from scripts.self_improve import r4_run_branch as runner
from scripts.self_improve.r4_attribute_steps import attribute, write_records
from scripts.self_improve.r4_run_branch import (
    dry_run, prepare_stage, stage_total_epochs, verify_stage_metrics,
)
from src.curriculum.failure_driven import CELLS, canonical_json, cell_stats, stage_distribution_record


def test_epoch_trap_fixed_for_all_boundaries():
    # Same loop-range calculation as Agent-R1 ray_trainer.py:907-910,942-943.
    for start in (20, 30, 70):
        dataloader_length = 40 // 4
        current_epoch = start // dataloader_length
        original_batches = sum(dataloader_length for _ in range(current_epoch, 1))
        patched_batches = sum(dataloader_length for _ in range(current_epoch, stage_total_epochs(start)))
        assert original_batches == 0
        assert patched_batches == 10
        assert start + patched_batches == start + 10


def test_six_stage_dry_run_uses_one_command_shape():
    fixed = dry_run(42, "fixed").splitlines()
    fd = dry_run(42, "failure_driven").splitlines()
    assert len(fixed) == len(fd) == 6
    for left, right in zip(fixed, fd):
        normalized = right.replace("R4_RHO=0.4", "R4_RHO=1.0").replace(
            "R4_ARM=failure_driven", "R4_ARM=fixed").replace("/failure_driven/", "/fixed/").replace(
            " 42 failure_driven ", " 42 fixed ")
        assert left == normalized
    assert "global_step_20" in fixed[0]
    assert "global_step_70" in fixed[-1]


@pytest.mark.parametrize("seed,arm,expected", [
    (42, "fixed", (15, 13, 12)), (42, "failure_driven", (9, 19, 12)),
    (137, "fixed", (15, 13, 12)), (137, "failure_driven", (12, 19, 9)),
])
def test_stage1_record_uses_selector_exactly(tmp_path, seed, arm, expected):
    workspace = __import__("pathlib").Path.cwd()
    warmup = workspace / f"artifacts/self_improve/r4/warmup_records/seed_{seed}.jsonl"
    records = [json.loads(line) for line in warmup.open()]
    stage_dir, cursor, actual = prepare_stage(seed=seed, arm=arm, stage=1,
        workspace=workspace, root=tmp_path, record_sources=[(warmup, records)],
        cursor=dict.fromkeys(CELLS, 0), seen_in_warmup={r["task_id"] for r in records})
    direct = stage_distribution_record(train_seed=seed, window=(1, 20),
        input_record_refs=[str(warmup)], stats=cell_stats(records, (1, 20)),
        rho=1.0 if arm == "fixed" else 0.4)
    data = (stage_dir / "stage_distribution.json").read_bytes()
    assert data == canonical_json(direct).encode()
    assert tuple(actual["cells"][c]["count"] for c in CELLS) == expected
    assert sum(cursor.values()) == 40
    parquet = pd.read_parquet(stage_dir / "train.parquet")
    selected = json.loads((stage_dir / "selection.json").read_text())["tasks"]
    assert [row["task_id"] for row in parquet["extra_info"]] == selected


def test_attribute_corrupt_step_aborts_before_output(tmp_path):
    run = tmp_path / "run"
    (run / "rollouts").mkdir(parents=True)
    task = "service_1_stale_version"
    trajectory = {"task_id": task, "instance_seed": 1, "binary_reward": 0,
                  "diagnostic_flags": ["no_read_config"]}
    (run / "trajectories.jsonl").write_text(json.dumps(trajectory) + "\n")
    dump = {"gts": task, "score": 0}
    (run / "rollouts/21.jsonl").write_text(json.dumps(dump) + "\n")
    output = tmp_path / "out.jsonl"
    rows = attribute(run, start_step=21, end_step=21)
    write_records(output, rows)
    assert rows[0]["global_step"] == 21
    output.unlink()
    dump["score"] = 1
    (run / "rollouts/21.jsonl").write_text(json.dumps(dump) + "\n")
    with pytest.raises(ValueError, match="multiset mismatch"):
        write_records(output, attribute(run, start_step=21, end_step=21))
    assert not output.exists()


def test_stage_metrics_enforce_numerical_stop(tmp_path):
    path = tmp_path / "metrics_target_30.jsonl"
    rows = [{"step": step, "data": {"actor/pg_loss": 0.1,
             "actor/grad_norm": 0.2}} for step in range(21, 31)]
    def write():
        path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    write()
    verify_stage_metrics(tmp_path, start_update=20)
    rows[4]["data"]["actor/pg_loss"] = float("nan")
    write()
    with pytest.raises(RuntimeError, match="numerical stop at U25"):
        verify_stage_metrics(tmp_path, start_update=20)
    rows[4]["data"]["actor/pg_loss"] = 0.1
    rows[4]["data"]["actor/grad_norm"] = 101.0
    write()
    with pytest.raises(RuntimeError, match="numerical stop at U25"):
        verify_stage_metrics(tmp_path, start_update=20)


@pytest.mark.parametrize("arm", ["fixed", "failure_driven"])
def test_execute_carries_cursor_and_attribution_across_six_stages(tmp_path, monkeypatch, arm):
    workspace = __import__("pathlib").Path.cwd()
    monkeypatch.setattr(runner, "run_dev_health", lambda **_kwargs: None)

    def simulate_stage(command, *, log_path, max_seconds, env=None):
        stage = int(command[4])
        start = int(command[5])
        attempt_dir = log_path.parent
        checkpoint = attempt_dir / "checkpoints" / f"global_step_{start+10}"
        (checkpoint / "actor" / "lora_adapter").mkdir(parents=True)
        (checkpoint / "actor" / "lora_adapter" / "adapter_model.safetensors").write_bytes(b"lora")
        (checkpoint / "actor" / "optim_world_size_1_rank_0.pt").write_bytes(b"optim")
        (checkpoint / "actor" / "extra_state_world_size_1_rank_0.pt").write_bytes(b"extra")
        selected = json.loads((attempt_dir.parent / "selection.json").read_text())["tasks"]
        trajectories = []
        rollout_dir = attempt_dir / "rollouts"
        rollout_dir.mkdir()
        for index, step in enumerate(range(start + 1, start + 11)):
            dumped = []
            for task_id in selected[index * 4:(index + 1) * 4]:
                for _ in range(4):
                    trajectories.append({"task_id": task_id, "instance_seed": 0,
                        "binary_reward": 0, "diagnostic_flags": [],
                        "run_id": f"r4_seed_42_{arm}", "arm": arm,
                        "seed": 42, "stage": stage})
                    dumped.append({"gts": task_id, "score": 0})
            (rollout_dir / f"{step}.jsonl").write_text(
                "".join(json.dumps(row) + "\n" for row in dumped))
        (attempt_dir / "trajectories.jsonl").write_text(
            "".join(json.dumps(row) + "\n" for row in trajectories))
        (attempt_dir / f"metrics_target_{start+10}.jsonl").write_text(
            "".join(json.dumps({"step": step, "data": {"actor/pg_loss": 0.1,
                "actor/grad_norm": 0.2}}) + "\n" for step in range(start + 1, start + 11)))
        return 0, 0.1

    monkeypatch.setattr(runner, "_run_bounded", simulate_stage)
    runner.execute(42, arm, workspace=workspace, root=tmp_path)
    branch = tmp_path / "seed_42" / arm
    cursor = dict.fromkeys(CELLS, 0)
    for stage in range(1, 7):
        stage_dir = branch / f"stage_{stage}"
        selection = json.loads((stage_dir / "selection.json").read_text())
        assert selection["cursor_before"] == cursor
        cursor = selection["cursor_after"]
        assert len(selection["tasks"]) == 40
        attributed = [json.loads(line) for line in (stage_dir / "attributed.jsonl").read_text().splitlines()]
        assert len(attributed) == 160
        assert {row["stage"] for row in attributed} == {stage}
        assert {row["global_step"] for row in attributed} == set(range(11 + 10 * stage, 21 + 10 * stage))
    assert sum(cursor.values()) == 240
    for stage in range(1, 7):
        actor = branch / f"stage_{stage}" / "attempt_1" / "checkpoints" / f"global_step_{20 + 10 * stage}" / "actor"
        assert (actor / "lora_adapter" / "adapter_model.safetensors").is_file()
        assert (actor / "optim_world_size_1_rank_0.pt").exists() == (stage == 6)


def test_prune_only_touches_superseded_r4_stage_state(tmp_path):
    root = tmp_path / "octorl_r4"
    stage = root / "seed_42" / "fixed" / "stage_1" / "attempt_1" / "checkpoints" / "global_step_30" / "actor"
    source = tmp_path / "octorl_r3c" / "seed_42" / "checkpoints" / "global_step_20" / "actor"
    for actor in (stage, source):
        (actor / "lora_adapter").mkdir(parents=True)
        (actor / "lora_adapter" / "adapter_model.safetensors").write_bytes(b"lora")
        (actor / "optim_world_size_1_rank_0.pt").write_bytes(b"optim")
        (actor / "extra_state_world_size_1_rank_0.pt").write_bytes(b"extra")
    assert runner._prune_optimizer_state(source.parent, root) == []
    assert (source / "optim_world_size_1_rank_0.pt").is_file()
    removed = runner._prune_optimizer_state(stage.parent, root)
    assert len(removed) == 2
    assert not (stage / "optim_world_size_1_rank_0.pt").exists()
    assert (stage / "lora_adapter" / "adapter_model.safetensors").is_file()


def test_stage_timeout_caps_each_attempt(tmp_path, monkeypatch):
    seen = []
    monkeypatch.setattr(runner, "_run_bounded",
                        lambda command, *, log_path, max_seconds, env=None: seen.append(max_seconds) or (1, 0.1))
    with pytest.raises(RuntimeError, match="stage 1 failed twice"):
        runner.execute(42, "fixed", workspace=__import__("pathlib").Path.cwd(), root=tmp_path)
    assert seen == [runner.STAGE_TIMEOUT_SECONDS] * 2


def test_failed_dev_health_is_recorded_not_fatal(tmp_path, monkeypatch):
    monkeypatch.setattr(runner, "_run_bounded", lambda command, *, log_path, max_seconds, env=None: (1, 0.1))
    stage_dir = tmp_path / "seed_42" / "fixed" / "stage_3"
    stage_dir.mkdir(parents=True)
    runner.run_dev_health(seed=42, arm="fixed", stage=3, checkpoint=tmp_path / "ckpt",
                          stage_dir=stage_dir, workspace=tmp_path, root=tmp_path)
    assert (stage_dir / "dev_health" / "FAILED").read_text() == "exit_code=1\n"
    assert json.loads((stage_dir / "dev_health" / "timing.json").read_text())["exit_code"] == 1
