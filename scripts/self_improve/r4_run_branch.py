#!/usr/bin/env python3
"""Prepare and run one R4 arm through six matched 10-update stages.

Use --dry-run to inspect all commands. --execute trains and must be run inside
the dedicated r4a3 tmux session on the GPU host.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import shlex
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import pandas as pd

from src.curriculum.failure_driven import (
    CELLS, build_stage_rows, cell_stats, selector_seed, stage_distribution_record,
    within_cell_order, write_stage_distribution,
)
try:
    from .r4_attribute_steps import attribute, write_records
except ImportError:
    from r4_attribute_steps import attribute, write_records


RATE_CNY_PER_HOUR = 2.18
BRANCH_LIMIT_CNY = 15.0
GLOBAL_LIMIT_CNY = 80.0
STAGE_TIMEOUT_SECONDS = 90 * 60  # R3c: ~4.2 min/update, so 10 updates + startup is ~50 min
MIN_FREE_BYTES = 3 * 1024**3     # one stage writes ~0.41 GB; stop well before the disk fills
ARMS = {"fixed": 1.0, "failure_driven": 0.4}
REMOTE_WORKSPACE = Path("/root/octorl_r3")
REMOTE_ROOT = Path("/root/autodl-tmp/octorl_r4")


def _optimizer_summary(path: Path) -> dict[str, float | int]:
    """Read a saved AdamW state on CPU, without touching model or data shards."""
    import torch

    state = torch.load(path, map_location="cpu", weights_only=False)
    moments = {}
    for key in ("exp_avg", "exp_avg_sq"):
        sum_squares = sum(
            tensor.double().square().sum().item()
            for slot in state["state"].values()
            if (tensor := slot.get(key)) is not None
        )
        moments[f"{key}_norm"] = math.sqrt(sum_squares)
    return {
        **moments,
        "state_entries": len(state["state"]),
        "lr": state["param_groups"][0]["lr"],
    }


def _validation_examples(trainer: Any) -> list[dict[str, Any]]:
    """Use two fixed rows from the stage's train parquet and one fixed response."""
    train_files = trainer.config.data.train_files
    if isinstance(train_files, (list, tuple)):
        train_files = train_files[0]
    rows = pd.read_parquet(train_files)["prompt"].iloc[:2]
    tokenizer = trainer.tokenizer
    response_ids = tokenizer.encode("I will inspect the configuration.", add_special_tokens=False)
    examples = []
    for messages in rows:
        prompt_ids = tokenizer.apply_chat_template(
            list(messages), tokenize=True, add_generation_prompt=True,
            enable_thinking=False,
        )
        examples.append({"prompt_ids": prompt_ids, "response_ids": response_ids})
    return examples


def validate_loaded_trainer(trainer: Any, output_path: str) -> None:
    """Exercise the real load, actor, rollout sync and save; never call fit()."""
    import ray
    import torch
    from verl import DataProto

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    started = time.time()
    resume = Path(trainer.config.trainer.resume_from_path)
    if (resume / "data.pt").exists():
        raise RuntimeError("resume view contains data.pt")
    source_actor = resume / "actor"
    source_summary = _optimizer_summary(source_actor / "optim_world_size_1_rank_0.pt")

    examples = _validation_examples(trainer)
    def actor_logprobs_for_examples() -> list[list[float]]:
        values = []
        for item in examples:
            prompt, response = item["prompt_ids"], item["response_ids"]
            ids = torch.tensor([prompt + response], dtype=torch.long)
            batch = DataProto.from_dict(tensors={
                "input_ids": ids,
                "attention_mask": torch.ones_like(ids),
                "position_ids": torch.arange(ids.shape[1], dtype=torch.long).unsqueeze(0),
                "responses": torch.tensor([response], dtype=torch.long),
            })
            output = trainer.actor_rollout_wg.compute_log_prob(batch)
            values.append(output.batch["old_log_probs"][0].tolist())
        return values

    # init_workers has loaded base + the prior LoRA, exactly the actor init
    # path used by the stage.  Compare it with the same actor after restore.
    preload_adapter_logprobs = actor_logprobs_for_examples()
    trainer.global_steps = 0
    trainer._load_checkpoint()
    if trainer.global_steps != 20:
        raise AssertionError(f"resumed global step {trainer.global_steps}, expected 20")
    actor_logprobs = actor_logprobs_for_examples()
    max_restore_logprob_diff = max(
        abs(a-b)
        for before, after in zip(preload_adapter_logprobs, actor_logprobs, strict=True)
        for a,b in zip(before, after, strict=True)
    )
    if max_restore_logprob_diff > 1e-3:
        raise RuntimeError(f"actor changed during optimizer restore: {max_restore_logprob_diff}")

    # This is the training engine created by init_workers, with its hybrid
    # worker weight sync.  The reference path is run after Ray exits.
    trainer.async_rollout_manager.wake_up()
    vllm_tokens = []
    try:
        server = trainer.async_rollout_manager.server_handles[0]
        for idx, item in enumerate(examples):
            generated = ray.get(server.generate.remote(
                prompt_ids=item["prompt_ids"],
                sampling_params={"temperature": 0.0, "max_tokens": 64,
                                 "ignore_eos": True},
                request_id=f"r4-load-only-{idx}",
            ))
            vllm_tokens.append(list(generated.token_ids))
    finally:
        trainer.async_rollout_manager.sleep()

    trainer._save_checkpoint()
    saved_actor = Path(trainer.config.trainer.default_local_dir) / "global_step_20" / "actor"
    saved_summary = _optimizer_summary(saved_actor / "optim_world_size_1_rank_0.pt")
    saved_extra = saved_actor / "extra_state_world_size_1_rank_0.pt"
    saved_lora = saved_actor / "lora_adapter" / "adapter_model.safetensors"
    if not saved_extra.is_file() or not saved_lora.is_file():
        raise RuntimeError("save omitted extra state or LoRA adapter")
    if list(saved_actor.glob("model_world_size_*.pt")):
        raise RuntimeError("save wrote a full model shard")
    for key in ("exp_avg_norm", "exp_avg_sq_norm", "lr"):
        if not math.isclose(source_summary[key], saved_summary[key], rel_tol=1e-6, abs_tol=1e-12):
            raise RuntimeError(f"optimizer restore failed for {key}: {source_summary[key]} -> {saved_summary[key]}")
    source_scheduler = torch.load(source_actor / "extra_state_world_size_1_rank_0.pt",
                                  map_location="cpu", weights_only=False)["lr_scheduler"]
    saved_scheduler = torch.load(saved_extra, map_location="cpu", weights_only=False)["lr_scheduler"]
    if source_scheduler["last_epoch"] != saved_scheduler["last_epoch"] or saved_scheduler["last_epoch"] != 20:
        raise RuntimeError("LR scheduler did not resume at U20")
    expected_next_lr = 2e-5 * (1 + math.cos(math.pi * (21 - 5) / (100 - 5))) / 2
    expected_current_lr = 2e-5 * (1 + math.cos(math.pi * (20 - 5) / (100 - 5))) / 2
    if not math.isclose(saved_summary["lr"], expected_current_lr, rel_tol=1e-6):
        raise RuntimeError("resumed optimizer LR is not the U20 cosine value")
    # Advance a CPU-only copy of the restored scheduler state.  The training
    # optimizer and scheduler are untouched, and no optimizer step occurs.
    import warnings
    from transformers import get_cosine_schedule_with_warmup
    dummy_optimizer = torch.optim.AdamW([torch.nn.Parameter(torch.zeros(1))], lr=2e-5)
    dummy_scheduler = get_cosine_schedule_with_warmup(
        dummy_optimizer, num_warmup_steps=5, num_training_steps=100)
    dummy_scheduler.load_state_dict(saved_scheduler)
    dummy_optimizer.param_groups[0]["lr"] = saved_summary["lr"]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        dummy_scheduler.step()
    actual_next_lr = dummy_optimizer.param_groups[0]["lr"]
    if not math.isclose(actual_next_lr, expected_next_lr, rel_tol=1e-6):
        raise RuntimeError("restored scheduler's U21 LR differs from cosine schedule")
    result = {
        "started_unix": started, "stopped_unix": time.time(),
        "global_step": trainer.global_steps,
        "resume_view": str(resume), "data_pt_absent": True,
        "source_optimizer": source_summary, "saved_optimizer": saved_summary,
        "scheduler_last_epoch": saved_scheduler["last_epoch"],
        "lr_for_first_resumed_optimizer_step": saved_summary["lr"],
        "lr_after_first_resumed_scheduler_step_expected": expected_next_lr,
        "lr_next_from_restored_scheduler_cpu_copy": actual_next_lr,
        "saved_bytes": {p.name: p.stat().st_size for p in saved_actor.rglob("*") if p.is_file()},
        "examples": examples, "preload_adapter_logprobs": preload_adapter_logprobs,
        "actor_logprobs": actor_logprobs,
        "max_restore_logprob_diff": max_restore_logprob_diff,
        "training_vllm_tokens": vllm_tokens,
    }
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(f"R4 load-only validation: {output}", flush=True)


def verify_reference(validation_path: Path, *, model_path: Path,
                     adapter_path: Path, seed: int) -> dict[str, Any]:
    """Compare a finished load-only run with the U20 adapter in isolation."""
    import gc
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM
    from verl.utils.torch_functional import logprobs_from_logits

    result = json.loads(validation_path.read_text(encoding="utf-8"))
    if result["global_step"] != 20 or not result["data_pt_absent"]:
        raise AssertionError("invalid load-only source record")
    base = AutoModelForCausalLM.from_pretrained(
        model_path, trust_remote_code=True, torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
    ).cuda().eval()
    model = PeftModel.from_pretrained(base, adapter_path).eval()
    reference_logprobs = []
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        for item in result["examples"]:
            prompt, response = item["prompt_ids"], item["response_ids"]
            ids = torch.tensor([prompt + response], device="cuda", dtype=torch.long)
            output = model(input_ids=ids, attention_mask=torch.ones_like(ids),
                           position_ids=torch.arange(ids.shape[1], device="cuda").unsqueeze(0),
                           use_cache=False)
            logits = output.logits[:, -len(response)-1:-1, :].clone()
            logits.div_(1.5)
            labels = torch.tensor([response], device="cuda", dtype=torch.long)
            reference_logprobs.append(logprobs_from_logits(logits, labels)[0].float().cpu().tolist())
    del model, base
    gc.collect()
    torch.cuda.empty_cache()

    from vllm import LLM, SamplingParams
    from vllm.inputs import TokensPrompt
    from vllm.lora.request import LoRARequest

    llm = LLM(model=str(model_path), trust_remote_code=True, max_model_len=8192,
              gpu_memory_utilization=0.85, enable_lora=True, max_lora_rank=16,
              enforce_eager=True, seed=seed)
    prompts = [TokensPrompt(prompt_token_ids=item["prompt_ids"])
               for item in result["examples"]]
    decoded = llm.generate(prompts, sampling_params=SamplingParams(
        temperature=0.0, max_tokens=64, ignore_eos=True),
        lora_request=LoRARequest("r4-reference", 1, str(adapter_path)), use_tqdm=False)
    reference_tokens = [list(item.outputs[0].token_ids) for item in decoded]
    max_abs_diff = max(abs(a-b) for actual, expected in zip(
        result["actor_logprobs"], reference_logprobs, strict=True)
        for a,b in zip(actual, expected, strict=True))
    token_match = result["training_vllm_tokens"] == reference_tokens
    result["reference_logprobs"] = reference_logprobs
    result["reference_vllm_tokens"] = reference_tokens
    result["max_abs_logprob_diff"] = max_abs_diff
    result["vllm_token_match"] = token_match
    result["hf_logprob_threshold_pass"] = max_abs_diff <= 1e-3
    validation_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"seed": seed, "max_abs_logprob_diff": max_abs_diff,
                      "vllm_token_match": token_match,
                      "hf_logprob_threshold_pass": result["hf_logprob_threshold_pass"]}))
    if not token_match:
        raise RuntimeError("training vLLM differs from U20 reference")
    return result


def stage_total_epochs(start_update: int, *, groups: int = 40, batch_size: int = 4) -> int:
    """One epoch beyond Agent-R1's `global_steps // len(stage_dataloader)`."""
    if start_update not in (20, 30, 40, 50, 60, 70) or groups % batch_size:
        raise ValueError("invalid R4 stage boundary or batch geometry")
    return start_update // (groups // batch_size) + 1


def stage_command(seed: int, arm: str, stage: int, *, workspace: Path = REMOTE_WORKSPACE,
                  root: Path = REMOTE_ROOT, previous_checkpoint: Path | None = None,
                  attempt: int = 1) -> list[str]:
    if arm not in ARMS or not 1 <= stage <= 6:
        raise ValueError("invalid arm or stage")
    start_update = 20 + 10 * (stage - 1)
    branch_root = root / f"seed_{seed}" / arm
    stage_dir = branch_root / f"stage_{stage}"
    if previous_checkpoint is None:
        previous_checkpoint = warmup_checkpoint(seed, root) if stage == 1 else (
            branch_root / f"stage_{stage-1}" / "attempt_1" / "checkpoints" / f"global_step_{start_update}")
    return ["bash", str(workspace / "scripts/self_improve/r4_train_stage.sh"), str(seed), arm,
            str(stage), str(start_update), str(stage_dir / "train.parquet"),
            str(previous_checkpoint), str(stage_dir / f"attempt_{attempt}")]


def warmup_checkpoint(seed: int, root: Path = REMOTE_ROOT) -> Path:
    if seed in (42, 137):
        return Path(f"/root/autodl-tmp/octorl_r3c/seed_{seed}/checkpoints/global_step_20")
    if seed == 2718:
        return root / "seed_2718/warmup/checkpoints/global_step_20"
    raise ValueError("R4 train seed must be 42, 137, or 2718")


def dry_run(seed: int, arm: str, *, workspace: Path = REMOTE_WORKSPACE,
            root: Path = REMOTE_ROOT) -> str:
    lines = []
    for stage in range(1, 7):
        command = stage_command(seed, arm, stage, workspace=workspace, root=root)
        lines.append(f"R4_RHO={ARMS[arm]} R4_ARM={arm} " + shlex.join(command))
    return "\n".join(lines) + "\n"


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.open(encoding="utf-8")]


def verify_stage_metrics(run_root: Path, *, start_update: int) -> None:
    """Apply the preregistered numerical stop to all ten completed updates."""
    end_update = start_update + 10
    path = run_root / f"metrics_target_{end_update}.jsonl"
    if not path.is_file():
        raise RuntimeError(f"missing stage metrics: {path}")
    records = _read_jsonl(path)
    by_step = {int(row["step"]): row["data"] for row in records
               if start_update < int(row["step"]) <= end_update}
    expected = set(range(start_update + 1, end_update + 1))
    if set(by_step) != expected:
        raise RuntimeError(f"stage metrics steps {sorted(by_step)} != {sorted(expected)}")
    for step, metrics in sorted(by_step.items()):
        grad_norm = metrics.get("actor/grad_norm")
        if grad_norm is None or not math.isfinite(float(grad_norm)) or float(grad_norm) > 100:
            raise RuntimeError(f"numerical stop at U{step}: actor/grad_norm={grad_norm}")
        losses = {key: value for key, value in metrics.items() if "loss" in key.lower()}
        if not losses:
            raise RuntimeError(f"missing loss metrics at U{step}")
        for key, value in losses.items():
            try:
                finite = math.isfinite(float(value))
            except (TypeError, ValueError):
                finite = False
            if not finite:
                raise RuntimeError(f"numerical stop at U{step}: {key}={value}")


def warmup_records_path(seed: int, *, workspace: Path, root: Path) -> Path:
    if seed in (42, 137):
        return workspace / f"artifacts/self_improve/r4/warmup_records/seed_{seed}.jsonl"
    return root / "seed_2718/warmup/attributed.jsonl"


def prepare_stage(*, seed: int, arm: str, stage: int, workspace: Path, root: Path,
                  record_sources: list[tuple[Path, list[dict[str, Any]]]],
                  cursor: dict[str, int], seen_in_warmup: set[str]) -> tuple[Path, dict[str, int], dict]:
    """Write the frozen selector record and the 40 ordered rows for one stage."""
    start = 20 + 10 * (stage - 1)
    window = (start - 19, start)
    refs = [str(path) for path, rows in record_sources if any(window[0] <= row["global_step"] <= window[1] for row in rows)]
    if not refs:
        raise ValueError("no training records in selector window")
    rows = [row for _, source in record_sources for row in source]
    stats = cell_stats(rows, window)
    distribution = stage_distribution_record(train_seed=seed, window=window, input_record_refs=refs,
                                             stats=stats, rho=ARMS[arm])
    counts = {cell: distribution["cells"][cell]["count"] for cell in CELLS}
    pool = pd.read_parquet(workspace / "artifacts/self_improve/r3/data/train_fault_only.parquet")
    by_id = {row["extra_info"]["task_id"]: row.to_dict() for _, row in pool.iterrows()}
    if len(by_id) != 320:
        raise ValueError("fault-only pool must contain 320 unique tasks")
    orders = {}
    for cell in CELLS:
        instances = [task for task, row in by_id.items() if row["extra_info"]["fault_type"] == cell]
        orders[cell] = within_cell_order(instances, seen_in_warmup, selector_seed(seed, cell))
    selected, next_cursor = build_stage_rows(counts, orders, cursor,
                                             train_seed=seed, stage_index=stage)
    if len(selected) != 40:
        raise AssertionError("stage selector did not return 40 task references")
    stage_dir = root / f"seed_{seed}" / arm / f"stage_{stage}"
    stage_dir.mkdir(parents=True, exist_ok=True)
    write_stage_distribution(stage_dir / "stage_distribution.json", distribution)
    pd.DataFrame([by_id[task] for task in selected]).to_parquet(stage_dir / "train.parquet", index=False)
    (stage_dir / "selection.json").write_text(json.dumps({"tasks": selected, "cursor_before": cursor,
        "cursor_after": next_cursor}, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return stage_dir, next_cursor, distribution


def _cost_so_far(root: Path, branch: Path) -> tuple[float, float]:
    global_seconds = branch_seconds = 0.0
    for path in root.glob("seed_*/**/timing.json"):
        seconds = float(json.loads(path.read_text(encoding="utf-8"))["elapsed_seconds"])
        global_seconds += seconds
        if branch in path.parents:
            branch_seconds += seconds
    return global_seconds * RATE_CNY_PER_HOUR / 3600, branch_seconds * RATE_CNY_PER_HOUR / 3600


def _prune_optimizer_state(checkpoint: Path, root: Path) -> list[str]:
    """Drop optimizer/extra state of a superseded R4 stage checkpoint; keep its LoRA adapter.

    Only checkpoints under the R4 root are eligible, so the R3c and warm-up sources are never touched.
    """
    if root not in checkpoint.parents or "stage_" not in str(checkpoint):
        return []
    if not (checkpoint / "actor/lora_adapter/adapter_model.safetensors").is_file():
        raise RuntimeError(f"refusing to prune {checkpoint}: LoRA adapter missing")
    removed = []
    for pattern in ("optim_world_size_*.pt", "extra_state_world_size_*.pt"):
        for path in (checkpoint / "actor").glob(pattern):
            path.unlink()
            removed.append(str(path))
    return removed


def _run_bounded(command: list[str], *, log_path: Path, max_seconds: float,
                 env: dict[str, str] | None = None) -> tuple[int, float]:
    began = time.monotonic()
    with log_path.open("w", encoding="utf-8") as log:
        process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT,
                                   start_new_session=True, env=env)
        try:
            code = process.wait(timeout=max_seconds)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
            code = 124
    return code, time.monotonic() - began


def run_dev_health(*, seed: int, arm: str, stage: int, checkpoint: Path,
                   stage_dir: Path, workspace: Path, root: Path) -> None:
    """Run the preregistered U50/U80 dev monitor without using it for selection."""
    if stage not in (3, 6):
        return
    eval_dir = stage_dir / "dev_health"
    eval_dir.mkdir(parents=True, exist_ok=True)
    branch = root / f"seed_{seed}" / arm
    global_cost, branch_cost = _cost_so_far(root, branch)
    available_cny = min(GLOBAL_LIMIT_CNY - global_cost, BRANCH_LIMIT_CNY - branch_cost)
    if available_cny <= 0:
        raise RuntimeError(f"R4 budget stop before U{20 + 10*stage} dev health")
    command = [sys.executable, str(workspace / "scripts/self_improve/r3b_eval.py"),
        "--manifest", str(workspace / "artifacts/self_improve/r3/data/dev_manifest.json"),
        "--output", str(eval_dir / "dev.json"),
        "--trajectories", str(eval_dir / "trajectories.jsonl"),
        "--lora-path", str(checkpoint / "actor/lora_adapter"),
        "--eval-seed", "310000"]
    code, elapsed = _run_bounded(command, log_path=eval_dir / "eval.log",
        max_seconds=available_cny * 3600 / RATE_CNY_PER_HOUR,
        env={**{k: v for k, v in os.environ.items() if k.lower() not in ("http_proxy", "https_proxy")},
             "PYTHONPATH": f"{workspace}:/root/Agent-R1",
             "HF_ENDPOINT": "https://hf-mirror.com", "CUDA_VISIBLE_DEVICES": "0"})
    (eval_dir / "timing.json").write_text(json.dumps({
        "elapsed_seconds": elapsed, "cost_cny": elapsed * RATE_CNY_PER_HOUR / 3600,
        "exit_code": code, "command": command,
    }, indent=2) + "\n", encoding="utf-8")
    if code != 0 or not (eval_dir / "dev.json").is_file():
        # Prereg: dev is monitoring_only, never used for selection, so a failed monitor must not
        # stop training. Record it loudly; the U50/U80 adapters stay available for a later re-run.
        (eval_dir / "FAILED").write_text(f"exit_code={code}\n", encoding="utf-8")
        print(f"WARNING: U{20 + 10*stage} dev health evaluation failed (exit {code}); training continues",
              file=sys.stderr, flush=True)


def execute(seed: int, arm: str, *, workspace: Path = REMOTE_WORKSPACE,
            root: Path = REMOTE_ROOT) -> None:
    if arm not in ARMS:
        raise ValueError("unknown arm")
    branch = root / f"seed_{seed}" / arm
    warmup = warmup_records_path(seed, workspace=workspace, root=root)
    records = _read_jsonl(warmup)
    if len(records) != 320:
        raise ValueError("U1-U20 warm-up must contain 320 rollouts")
    sources = [(warmup, records)]
    seen = {row["task_id"] for row in records}
    cursor = dict.fromkeys(CELLS, 0)
    previous = warmup_checkpoint(seed, root)
    for stage in range(1, 7):
        start = 20 + 10 * (stage - 1)
        stage_dir, next_cursor, _ = prepare_stage(seed=seed, arm=arm, stage=stage,
            workspace=workspace, root=root, record_sources=sources, cursor=cursor, seen_in_warmup=seen)
        free = __import__("shutil").disk_usage(root).free
        if free < MIN_FREE_BYTES:
            raise RuntimeError(f"disk stop before stage {stage}: {free / 1024**3:.1f} GB free")
        success = False
        for attempt in (1, 2):
            global_cost, branch_cost = _cost_so_far(root, branch)
            available_cny = min(GLOBAL_LIMIT_CNY - global_cost, BRANCH_LIMIT_CNY - branch_cost)
            if available_cny <= 0:
                raise RuntimeError(f"R4 budget stop before stage {stage}: global ¥{global_cost:.2f}, branch ¥{branch_cost:.2f}")
            command = stage_command(seed, arm, stage, workspace=workspace, root=root,
                                    previous_checkpoint=previous, attempt=attempt)
            attempt_dir = stage_dir / f"attempt_{attempt}"
            attempt_dir.mkdir(parents=True, exist_ok=True)
            code, elapsed = _run_bounded(command, log_path=attempt_dir / "training.log",
                                         max_seconds=min(STAGE_TIMEOUT_SECONDS,
                                                         available_cny * 3600 / RATE_CNY_PER_HOUR))
            (attempt_dir / "timing.json").write_text(json.dumps({"elapsed_seconds": elapsed,
                "cost_cny": elapsed * RATE_CNY_PER_HOUR / 3600, "exit_code": code,
                "command": command}, indent=2) + "\n", encoding="utf-8")
            checkpoint = attempt_dir / "checkpoints" / f"global_step_{start+10}"
            if code == 0 and checkpoint.is_dir():
                verify_stage_metrics(attempt_dir, start_update=start)
                if list(checkpoint.rglob("model_world_size_*.pt")):
                    raise RuntimeError("stage checkpoint unexpectedly contains a full model shard")
                attributed = attribute(attempt_dir, start_step=start+1, end_step=start+10)
                if len(attributed) != 160 or {row.get("stage") for row in attributed} != {stage}:
                    raise ValueError(f"stage {stage}: attribution missing or not 160 rows")
                output = stage_dir / "attributed.jsonl"
                write_records(output, attributed)
                sources.append((output, attributed))
                cursor = next_cursor
                pruned = _prune_optimizer_state(previous, root)
                if pruned:
                    (stage_dir / "pruned_previous.json").write_text(
                        json.dumps(pruned, indent=2) + "\n", encoding="utf-8")
                previous = checkpoint
                run_dev_health(seed=seed, arm=arm, stage=stage, checkpoint=checkpoint,
                               stage_dir=stage_dir, workspace=workspace, root=root)
                success = True
                break
        if not success:
            raise RuntimeError(f"stage {stage} failed twice; branch stopped at U{start}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, choices=(42, 137, 2718), required=True)
    parser.add_argument("--arm", choices=tuple(ARMS))
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--dry-run", action="store_true")
    mode.add_argument("--execute", action="store_true")
    mode.add_argument("--reference-validation", type=Path)
    parser.add_argument("--adapter-path", type=Path)
    args = parser.parse_args()
    if args.reference_validation:
        if args.adapter_path is None:
            parser.error("--reference-validation requires --adapter-path")
        verify_reference(args.reference_validation,
                         model_path=Path("/root/autodl-tmp/models/Qwen3-4B"),
                         adapter_path=args.adapter_path, seed=args.seed)
    elif args.arm is None:
        parser.error("--dry-run and --execute require --arm")
    elif args.dry_run:
        print(dry_run(args.seed, args.arm), end="")
    else:
        execute(args.seed, args.arm)


if __name__ == "__main__":
    main()
