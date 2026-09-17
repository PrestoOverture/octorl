#!/usr/bin/env python3
"""Evaluate base Qwen3-4B or an R3 LoRA checkpoint on a frozen split."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import math
import time
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any

from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest

from agent_r1.env.base import Action
from run_pilot import _prompt, _selected_logprobs
from src.adapters.tool_recovery_agent_r1 import MAX_RESPONSE_LENGTH, ToolRecoveryAgentEnv, append_trajectory


MODEL_REVISION = "1cfa9a7208912126459214e8b04321603b3df60c"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def evaluate(
    *,
    manifest_path: Path,
    output_path: Path,
    trajectories_path: Path,
    model: str,
    lora_path: Path | None,
    eval_seed: int,
    group_size: int = 4,
    temperature: float = 1.5,
    top_p: float = 0.95,
    max_steps: int = 5,
    max_tokens_per_turn: int = 1024,
) -> dict[str, Any]:
    started = time.perf_counter()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    tokenizer = AutoTokenizer.from_pretrained(model, trust_remote_code=True)
    llm = LLM(
        model=model,
        trust_remote_code=True,
        max_model_len=8192,
        gpu_memory_utilization=0.90,
        enable_lora=lora_path is not None,
        max_lora_rank=16,
    )
    lora_request = LoRARequest("r3", 1, str(lora_path)) if lora_path is not None else None
    trajectories_path.parent.mkdir(parents=True, exist_ok=True)
    trajectories_path.write_text("", encoding="utf-8")

    states: list[dict[str, Any]] = []
    for instance_index, instance in enumerate(manifest["instances"]):
        for rollout_index in range(group_size):
            sampling_seed = eval_seed + instance_index * group_size + rollout_index
            env = ToolRecoveryAgentEnv(
                instance=instance,
                model_sampling_seed=sampling_seed,
                trajectory_path=trajectories_path,
            )
            states.append(
                {
                    "env": env,
                    "observation": env.reset(),
                    "rollout_index": rollout_index,
                    "sampling_seed": sampling_seed,
                    "tokens_generated": 0,
                    "done": False,
                }
            )

    for step_index in range(max_steps):
        active = [state for state in states if not state["done"]]
        if not active:
            break
        prompts = [_prompt(tokenizer, state["observation"].messages) for state in active]
        params = [
            SamplingParams(
                temperature=temperature,
                top_p=top_p,
                max_tokens=min(max_tokens_per_turn, MAX_RESPONSE_LENGTH - state["tokens_generated"]),
                seed=state["sampling_seed"],
                logprobs=1,
            )
            for state in active
        ]
        outputs = llm.generate(
            prompts,
            sampling_params=params,
            lora_request=lora_request,
            use_tqdm=True,
        )
        for state, request_output in zip(active, outputs, strict=True):
            generated = request_output.outputs[0]
            _selected_logprobs(generated)
            state["tokens_generated"] += len(generated.token_ids)
            observation, _, done, _ = asyncio.run(
                state["env"].step(Action(text=generated.text, token_ids=list(generated.token_ids)))
            )
            state["observation"] = observation
            state["done"] = done
            if not done and (step_index == max_steps - 1 or state["tokens_generated"] >= MAX_RESPONSE_LENGTH):
                state["env"].force_max_steps()
                state["done"] = True

    rollouts: list[dict[str, Any]] = []
    for state in states:
        env = state["env"]
        if not state["done"]:
            env.force_max_steps()
        continuous, binary = env.final_scores()
        record = env.trajectory()
        append_trajectory(trajectories_path, record)
        instance = env.core.instance
        rollouts.append(
            {
                "task_id": instance.task_id,
                "instance_seed": instance.seed,
                "model_sampling_seed": state["sampling_seed"],
                "rollout_index": state["rollout_index"],
                "family": instance.family,
                "fault_type": instance.fault_type,
                "faulted_field": instance.faulted_field,
                "continuous_reward": continuous,
                "binary_reward": binary,
                "tokens_generated": state["tokens_generated"],
                "tool_calls": env.core.tool_calls,
                "termination_reason": env.core.termination_reason,
                "diagnostic_flags": sorted(env.core.diagnostic_flags),
            }
        )

    expected = len(manifest["instances"]) * group_size
    if len(rollouts) != expected:
        raise AssertionError(f"expected {expected} rollouts, got {len(rollouts)}")
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rollouts:
        grouped[row["task_id"]].append(row)
    fault_rows = [row for row in rollouts if row["fault_type"] != "normal"]
    normal_rows = [row for row in rollouts if row["fault_type"] == "normal"]
    mixed = sum(
        1
        for rows in grouped.values()
        if rows[0]["fault_type"] != "normal"
        and max(row["continuous_reward"] for row in rows) - min(row["continuous_reward"] for row in rows) > 1e-9
    )
    fault_groups = sum(rows[0]["fault_type"] != "normal" for rows in grouped.values())
    result = {
        "model": model,
        "model_revision": MODEL_REVISION,
        "lora_path": str(lora_path) if lora_path is not None else None,
        "manifest_path": str(manifest_path),
        "manifest_sha256": _sha256(manifest_path),
        "trajectory_path": str(trajectories_path),
        "instance_count": len(manifest["instances"]),
        "rollout_count": len(rollouts),
        "group_size": group_size,
        "sampling": {
            "temperature": temperature,
            "top_p": top_p,
            "max_response_length": MAX_RESPONSE_LENGTH,
            "max_steps": max_steps,
            "eval_seed": eval_seed,
        },
        "continuous_reward_mean": mean(row["continuous_reward"] for row in rollouts),
        "continuous_fault_reward_mean": mean(row["continuous_reward"] for row in fault_rows),
        "binary_fcr": mean(row["binary_reward"] for row in fault_rows),
        "binary_npr": mean(row["binary_reward"] for row in normal_rows),
        "mixed_fault_groups": mixed,
        "fault_group_count": fault_groups,
        "mixed_group_fraction": mixed / fault_groups,
        "total_tokens_generated": sum(row["tokens_generated"] for row in rollouts),
        "total_wall_time_seconds": time.perf_counter() - started,
        "all_finite": all(math.isfinite(row["continuous_reward"]) for row in rollouts),
        "rollouts": rollouts,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--trajectories", type=Path, required=True)
    parser.add_argument("--model", default="/root/autodl-tmp/models/Qwen3-4B")
    parser.add_argument("--lora-path", type=Path)
    parser.add_argument("--eval-seed", type=int, default=310000)
    args = parser.parse_args()
    result = evaluate(
        manifest_path=args.manifest,
        output_path=args.output,
        trajectories_path=args.trajectories,
        model=args.model,
        lora_path=args.lora_path,
        eval_seed=args.eval_seed,
    )
    print(json.dumps({key: result[key] for key in (
        "rollout_count",
        "continuous_reward_mean",
        "binary_fcr",
        "total_wall_time_seconds",
    )}, sort_keys=True))


if __name__ == "__main__":
    main()
