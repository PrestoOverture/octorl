#!/usr/bin/env python3
"""Run the inference-only Qwen3-4B tool-recovery pilot with vLLM."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

from agent_r1.env.base import Action
from src.adapters.tool_recovery_agent_r1 import (
    MAX_RESPONSE_LENGTH,
    TOOL_SCHEMAS,
    ToolRecoveryAgentEnv,
    append_trajectory,
)


def _prompt(tokenizer: Any, messages: list[dict[str, Any]]) -> str:
    return tokenizer.apply_chat_template(
        messages,
        tools=TOOL_SCHEMAS,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )


def _selected_logprobs(output: Any) -> list[float]:
    values: list[float] = []
    if output.logprobs is None:
        raise AssertionError("vLLM did not return requested token logprobs")
    for token_id, candidates in zip(output.token_ids, output.logprobs, strict=True):
        selected = candidates.get(token_id)
        if selected is None:
            raise AssertionError(f"chosen token {token_id} absent from vLLM logprobs")
        values.append(float(selected.logprob))
    if len(values) != len(output.token_ids):
        raise AssertionError("response_logprobs must align one-to-one with response_ids")
    return values


def run_pilot(
    manifest_path: Path,
    results_path: Path,
    trajectories_path: Path,
    model: str,
    max_steps: int,
    max_tokens_per_turn: int,
    temperature: float,
    top_p: float,
    reward_mode: str,
) -> dict[str, Any]:
    if reward_mode not in {"binary", "continuous"}:
        raise ValueError(f"unsupported reward mode: {reward_mode}")
    started = time.perf_counter()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    tokenizer = AutoTokenizer.from_pretrained(model, trust_remote_code=True)
    llm = LLM(model=model, trust_remote_code=True, max_model_len=8192, gpu_memory_utilization=0.90)
    trajectories_path.parent.mkdir(parents=True, exist_ok=True)
    trajectories_path.write_text("", encoding="utf-8")

    states: list[dict[str, Any]] = []
    for instance in manifest["instances"]:
        for rollout_index in range(2):
            sampling_seed = rollout_index
            env = ToolRecoveryAgentEnv(
                instance=instance,
                model_sampling_seed=sampling_seed,
                trajectory_path=trajectories_path,
            )
            observation = env.reset()
            states.append(
                {
                    "env": env,
                    "observation": observation,
                    "rollout_index": rollout_index,
                    "sampling_seed": sampling_seed,
                    "tokens_generated": 0,
                    "response_logprobs": [],
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
        outputs = llm.generate(prompts, sampling_params=params, use_tqdm=True)
        for state, request_output in zip(active, outputs, strict=True):
            generated = request_output.outputs[0]
            logprobs = _selected_logprobs(generated)
            state["tokens_generated"] += len(generated.token_ids)
            state["response_logprobs"].append(logprobs)
            observation, _, done, _ = __import__("asyncio").run(
                state["env"].step(Action(text=generated.text, token_ids=list(generated.token_ids)))
            )
            state["observation"] = observation
            state["done"] = done
            if not done and (step_index == max_steps - 1 or state["tokens_generated"] >= MAX_RESPONSE_LENGTH):
                state["env"].force_max_steps()
                state["done"] = True

    wall_time = time.perf_counter() - started
    rollouts: list[dict[str, Any]] = []
    for state in states:
        env = state["env"]
        if not state["done"]:
            env.force_max_steps()
        binary = env.core.reward
        continuous = env.core.continuous_reward
        selected_reward = continuous if reward_mode == "continuous" else binary
        env.core.record.reward = selected_reward
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
                "reward": selected_reward,
                "binary_reward": binary,
                "continuous_reward": continuous,
                "tool_calls": env.core.tool_calls,
                "termination_reason": env.core.termination_reason,
                "diagnostic_flags": sorted(env.core.diagnostic_flags),
                "tokens_generated": state["tokens_generated"],
                "turns": len(state["response_logprobs"]),
            }
        )
    expected = len(manifest["instances"]) * 2
    if len(rollouts) != expected:
        raise AssertionError(f"expected {expected} rollouts, got {len(rollouts)}")
    if sum(1 for _ in trajectories_path.open(encoding="utf-8")) != expected:
        raise AssertionError("trajectory count does not match rollout count")
    results = {
        "model": model,
        "group_size": 2,
        "manifest_path": str(manifest_path),
        "trajectory_path": str(trajectories_path),
        "reward_mode": reward_mode,
        "sampling": {
            "temperature": temperature,
            "top_p": top_p,
            "max_steps": max_steps,
            "model_sampling_seeds": [0, 1],
        },
        "total_wall_time_seconds": wall_time,
        "rollouts": rollouts,
    }
    results_path.parent.mkdir(parents=True, exist_ok=True)
    results_path.write_text(json.dumps(results, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=Path("artifacts/self_improve/pilot/manifest.json"))
    parser.add_argument("--results", type=Path, default=Path("artifacts/self_improve/pilot/results.json"))
    parser.add_argument("--trajectories", type=Path, default=Path("artifacts/self_improve/pilot/trajectories.jsonl"))
    parser.add_argument("--model", default="Qwen/Qwen3-4B")
    parser.add_argument("--max-steps", type=int, default=5)
    parser.add_argument("--max-tokens-per-turn", type=int, default=1024)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.8)
    parser.add_argument("--reward-mode", choices=("binary", "continuous"), default="continuous")
    args = parser.parse_args()
    result = run_pilot(
        args.manifest,
        args.results,
        args.trajectories,
        args.model,
        args.max_steps,
        args.max_tokens_per_turn,
        args.temperature,
        args.top_p,
        args.reward_mode,
    )
    print(json.dumps({"rollouts": len(result["rollouts"]), "wall_time_seconds": result["total_wall_time_seconds"]}))


if __name__ == "__main__":
    main()
