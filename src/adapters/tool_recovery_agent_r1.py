"""Agent-R1 adapter for the frozen r2.0 tool-recovery environment."""

from __future__ import annotations

import json
import os
import time
import uuid
from pathlib import Path
from typing import Any

from agent_r1.agent_flow.agent_env_loop import AgentEnvLoop
from agent_r1.agent_flow.agent_flow import AgentFlowOutput, AgentFlowStep
from agent_r1.env import AgentEnv
from agent_r1.env.base import Action, Observation
from agent_r1.env.tool_format import ToolFormatWrapper

from src.tasks.tool_recovery.environment import ToolRecoveryEnvironment
from src.tasks.tool_recovery.records import Instance
from src.tasks.tool_recovery.reward import continuous_reward


MAX_RESPONSE_LENGTH = 4096
DEFAULT_TRAJECTORY_PATH = Path("artifacts/self_improve/pilot/trajectories.jsonl")

SYSTEM_PROMPT = (
    "You repair a service configuration using the available tools. Read the "
    "configuration, query the relevant field constraint and state variables, "
    "submit only the necessary fix, then respond with a short final message."
)

TOOL_SCHEMAS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "read_config",
            "description": "Read the current service configuration and check for errors.",
            "parameters": {"type": "object", "properties": {}, "additionalProperties": False},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "query_info",
            "description": "Query information about a configuration field, a system state variable, or list all fields.",
            "parameters": {
                "type": "object",
                "properties": {
                    "topic": {
                        "type": "string",
                        "description": "A field name (e.g. 'max_memory_mb'), a state variable name (e.g. 'system.total_memory_mb'), or the literal 'fields' to list all field names.",
                    }
                },
                "required": ["topic"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "submit_fix",
            "description": "Set a configuration field to a new value.",
            "parameters": {
                "type": "object",
                "properties": {
                    "field": {"type": "string", "description": "The configuration field name to update."},
                    "value": {"description": "The new value for the field."},
                },
                "required": ["field", "value"],
                "additionalProperties": False,
            },
        },
    },
]


def _instance(value: Instance | dict[str, Any] | str) -> Instance:
    if isinstance(value, Instance):
        return value
    if isinstance(value, str):
        value = json.loads(value)
    return Instance(**value)


def append_trajectory(path: str | Path, record: dict[str, Any]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(record, sort_keys=True, ensure_ascii=False) + "\n")


@AgentEnv.register("octorl_tool_recovery")
class ToolRecoveryAgentEnv(AgentEnv):
    """Translate Agent-R1 actions into r2.0 environment calls."""

    def __init__(
        self,
        instance: Instance | dict[str, Any] | str | None = None,
        model_sampling_seed: int | None = None,
        trajectory_path: str | Path = DEFAULT_TRAJECTORY_PATH,
        **_: Any,
    ) -> None:
        self.instance_data = instance
        self.model_sampling_seed = model_sampling_seed
        self.trajectory_path = Path(trajectory_path)
        self.format = ToolFormatWrapper.from_name("hermes")
        self.messages: list[dict[str, Any]] = []
        self.core: ToolRecoveryEnvironment | None = None
        self.logged = False

    @property
    def tool_schemas(self) -> list[dict[str, Any]]:
        return TOOL_SCHEMAS

    def reset(self, **kwargs: Any) -> Observation:
        raw = kwargs.get("instance", self.instance_data)
        if raw is None:
            raw = kwargs.get("extra_info", {}).get("instance")
        if raw is None:
            raise ValueError("tool-recovery instance is required")
        sampling_seed = kwargs.get("model_sampling_seed", self.model_sampling_seed)
        self.core = ToolRecoveryEnvironment(_instance(raw), model_sampling_seed=sampling_seed)
        self.messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    f"Repair task {self.core.instance.task_id} for service "
                    f"{self.core.instance.family}. Use the tools to inspect and repair it."
                ),
            },
        ]
        self.logged = False
        return Observation(messages=list(self.messages))

    async def step(self, action: Action) -> tuple[Observation, float | None, bool, dict[str, Any]]:
        if self.core is None:
            raise RuntimeError("reset must be called before step")
        text = action.text or ""
        _, calls = self.format.parse_response(text)
        self.messages.append({"role": "assistant", "content": text})
        observations: list[dict[str, Any]] = []
        reward: int | None = None
        done = False
        if not calls:
            observation, done, reward = self.core.text_response(text)
            observations.append(observation)
        else:
            for call in calls:
                observation, done, reward = self.core.step(call.name, call.arguments)
                observations.append(observation)
                if done:
                    break
        if observations:
            rendered = "\n".join(
                self.format.format_observation(json.dumps(item, sort_keys=True, ensure_ascii=False))
                for item in observations
            )
            self.messages.append({"role": "user", "content": rendered})
        return Observation(messages=list(self.messages)), (float(reward) if reward is not None else None), done, {
            "termination_reason": self.core.termination_reason,
            "diagnostic_flags": sorted(self.core.diagnostic_flags),
        }

    def final_scores(self) -> tuple[float, int]:
        """Return the continuous diagnostic and frozen binary verifier score."""
        if self.core is None:
            raise RuntimeError("reset must be called before reading rewards")
        training_reward = continuous_reward(
            self.core.config,
            self.core.instance.golden_config,
            self.core.family,
        )
        return training_reward, self.core.reward

    def force_max_steps(self) -> float:
        """End an unterminated rollout at the flow cap without mutating the frozen core."""
        if self.core is None:
            raise RuntimeError("reset must be called before force_max_steps")
        if not self.core.terminated:
            self.core.terminated = True
            self.core.termination_reason = "max_steps"
            self.core.record.termination_reason = "max_steps"
            self.core.record.diagnostic_flags = set(self.core.diagnostic_flags)
        continuous, _ = self.final_scores()
        self.core.record.reward = continuous
        return float(self.core.reward)

    def trajectory(self) -> dict[str, Any]:
        if self.core is None:
            raise RuntimeError("reset must be called before trajectory")
        training_reward, binary = self.final_scores()
        self.core.record.reward = training_reward
        record = self.core.record.to_dict()
        record.update(
            {
                "binary_reward": binary,
                "continuous_reward": training_reward,
                "final_config": dict(self.core.config),
            }
        )
        # R4 identity is optional so pre-R4 trajectory bytes remain unchanged.
        for env_name, field in (
            ("R4_RUN_ID", "run_id"),
            ("R4_ARM", "arm"),
            ("R4_SEED", "seed"),
            ("R4_STAGE", "stage"),
        ):
            value = os.environ.get(env_name)
            if value is not None:
                record[field] = int(value) if field in {"seed", "stage"} else value
        return record

    def log_trajectory(self) -> None:
        if not self.logged:
            append_trajectory(self.trajectory_path, self.trajectory())
            self.logged = True


class ToolRecoveryAgentFlow(AgentEnvLoop):
    """Agent-R1 rollout loop with generated-token-only response masks."""

    max_response_length = MAX_RESPONSE_LENGTH

    async def run(self, sampling_params: dict[str, Any], **kwargs: Any) -> AgentFlowOutput:
        env = self._create_env(**kwargs)
        if not isinstance(env, ToolRecoveryAgentEnv):
            raise TypeError("ToolRecoveryAgentFlow requires ToolRecoveryAgentEnv")
        obs = env.reset(**kwargs)
        params = dict(sampling_params)
        if env.core is not None and env.core.model_sampling_seed is not None:
            params["seed"] = env.core.model_sampling_seed
        pending: list[AgentFlowStep] = []
        metrics = {"generate_sequences": 0.0, "tool_calls": 0.0}
        final_reward: float | None = None
        try:
            for step_index in range(self.max_steps):
                prompt_ids = await self._obs_to_prompt(obs, tools=env.tool_schemas)
                if len(prompt_ids) > self.prompt_length:
                    final_reward = float(env.force_max_steps())
                    break
                started = time.perf_counter()
                output = await self.server_manager.generate(
                    request_id=uuid.uuid4().hex, prompt_ids=prompt_ids, sampling_params=params
                )
                metrics["generate_sequences"] += time.perf_counter() - started
                response_ids = output.token_ids[: min(self.response_length, self.max_response_length)]
                logprobs = output.log_probs[: len(response_ids)] if output.log_probs is not None else None
                if logprobs is None or len(logprobs) != len(response_ids):
                    raise AssertionError("response_logprobs must align one-to-one with response_ids")
                response_text = self.tokenizer.decode(response_ids, skip_special_tokens=self.skip_special_tokens)
                tool_started = time.perf_counter()
                obs, reward, done, _ = await env.step(Action(text=response_text, token_ids=response_ids))
                metrics["tool_calls"] += time.perf_counter() - tool_started
                pending.append(
                    AgentFlowStep(
                        prompt_ids=prompt_ids,
                        response_ids=response_ids,
                        response_logprobs=logprobs,
                        response_mask=[1] * len(response_ids),
                        reward_score=None,
                    )
                )
                if done:
                    _, binary = env.final_scores()
                    final_reward = float(binary)
                    break
                if step_index == self.max_steps - 1:
                    final_reward = float(env.force_max_steps())
            if final_reward is None:
                final_reward = float(env.force_max_steps())
            steps = []
            for index, step in enumerate(pending):
                step.reward_score = final_reward if index == len(pending) - 1 else 0.0
                if index == len(pending) - 1:
                    training_reward, binary = env.final_scores()
                    if not hasattr(step, "extra_fields"):
                        step.extra_fields = {}
                    step.extra_fields.update(
                        {
                            "continuous_reward": training_reward,
                            "binary_reward": binary,
                            "task_id": env.core.instance.task_id,
                            "fault_type": env.core.instance.fault_type,
                        }
                    )
                steps.append(await self._postprocess(step, **kwargs))
            return AgentFlowOutput(steps=steps, metrics=metrics)
        finally:
            env.log_trajectory()


__all__ = [
    "MAX_RESPONSE_LENGTH",
    "SYSTEM_PROMPT",
    "TOOL_SCHEMAS",
    "ToolRecoveryAgentEnv",
    "ToolRecoveryAgentFlow",
    "append_trajectory",
]
