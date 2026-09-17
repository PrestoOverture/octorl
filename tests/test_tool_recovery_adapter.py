"""Unit tests for the Agent-R1 tool-recovery bridge."""

from __future__ import annotations

import asyncio
import importlib
import json
import sys
import types
from dataclasses import dataclass

import pytest

from src.tasks.tool_recovery.generator import generate_dataset


@pytest.fixture(scope="module")
def adapter():
    try:
        return importlib.import_module("src.adapters.tool_recovery_agent_r1")
    except ModuleNotFoundError as error:
        if error.name != "agent_r1":
            raise

    class Registry:
        _registry = {}

        @classmethod
        def register(cls, name):
            def decorate(value):
                cls._registry[name] = value
                return value
            return decorate

    class AgentEnv(Registry):
        pass

    class AgentEnvLoop:
        pass

    class ToolFormatWrapper:
        @classmethod
        def from_name(cls, name):
            assert name == "hermes"
            return cls()

        def parse_response(self, text):
            import re
            calls = []
            for raw in re.findall(r"<tool_call>(.*?)</tool_call>", text, re.DOTALL):
                value = json.loads(raw)
                calls.append(types.SimpleNamespace(name=value["name"], arguments=value["arguments"]))
            return text, calls

        def format_observation(self, text):
            return f"<tool_response>\n{text}\n</tool_response>"

    @dataclass
    class Action:
        text: str | None = None
        token_ids: list[int] | None = None

    @dataclass
    class Observation:
        messages: list[dict] | None = None

    @dataclass
    class AgentFlowStep:
        prompt_ids: list[int]
        response_ids: list[int]
        response_logprobs: list[float] | None = None
        response_mask: list[int] | None = None
        reward_score: float | None = None

    @dataclass
    class AgentFlowOutput:
        steps: list
        metrics: dict

    modules = {
        "agent_r1": types.ModuleType("agent_r1"),
        "agent_r1.agent_flow": types.ModuleType("agent_r1.agent_flow"),
        "agent_r1.agent_flow.agent_env_loop": types.ModuleType("agent_r1.agent_flow.agent_env_loop"),
        "agent_r1.agent_flow.agent_flow": types.ModuleType("agent_r1.agent_flow.agent_flow"),
        "agent_r1.env": types.ModuleType("agent_r1.env"),
        "agent_r1.env.base": types.ModuleType("agent_r1.env.base"),
        "agent_r1.env.tool_format": types.ModuleType("agent_r1.env.tool_format"),
    }
    modules["agent_r1.agent_flow.agent_env_loop"].AgentEnvLoop = AgentEnvLoop
    modules["agent_r1.agent_flow.agent_flow"].AgentFlowOutput = AgentFlowOutput
    modules["agent_r1.agent_flow.agent_flow"].AgentFlowStep = AgentFlowStep
    modules["agent_r1.env"].AgentEnv = AgentEnv
    modules["agent_r1.env.base"].Action = Action
    modules["agent_r1.env.base"].Observation = Observation
    modules["agent_r1.env.tool_format"].ToolFormatWrapper = ToolFormatWrapper
    sys.modules.update(modules)
    return importlib.import_module("src.adapters.tool_recovery_agent_r1")


def _call(name, arguments):
    return f'<tool_call>{{"name":"{name}","arguments":{json.dumps(arguments)}}}</tool_call>'


def _constraint_instance():
    return next(
        instance
        for instance in generate_dataset("dev", 50, 100000).instances
        if instance.fault_type == "constraint_violation"
    )


def test_registered_and_frozen_schemas(adapter):
    assert adapter.AgentEnv._registry["octorl_tool_recovery"] is adapter.ToolRecoveryAgentEnv
    assert [item["function"]["name"] for item in adapter.TOOL_SCHEMAS] == ["read_config", "query_info", "submit_fix"]
    assert adapter.MAX_RESPONSE_LENGTH == 4096


def test_scripted_four_call_oracle(adapter, tmp_path):
    instance = _constraint_instance()
    family_field = instance.faulted_field
    from src.tasks.tool_recovery.generator import FAMILY_BY_NAME
    field = FAMILY_BY_NAME[instance.family].get_field(family_field)
    env = adapter.ToolRecoveryAgentEnv(instance=instance, model_sampling_seed=0, trajectory_path=tmp_path / "t.jsonl")
    env.reset()
    for name, arguments in (
        ("read_config", {}),
        ("query_info", {"topic": family_field}),
        ("query_info", {"topic": field.state_var}),
        ("submit_fix", {"field": family_field, "value": instance.golden_config[family_field]}),
    ):
        _, _, done, _ = asyncio.run(env.step(adapter.Action(text=_call(name, arguments))))
        assert not done
    assert env.core.reward == 1


def test_text_only_terminates(adapter, tmp_path):
    env = adapter.ToolRecoveryAgentEnv(instance=_constraint_instance(), trajectory_path=tmp_path / "t.jsonl")
    env.reset()
    _, reward, done, _ = asyncio.run(env.step(adapter.Action(text="I cannot repair it.")))
    assert done and reward == 0
    assert env.core.termination_reason == "text_response"


def test_max_steps_reads_environment_reward(adapter, tmp_path):
    instance = _constraint_instance()
    env = adapter.ToolRecoveryAgentEnv(instance=instance, trajectory_path=tmp_path / "t.jsonl")
    env.reset()
    env.core.config = dict(instance.golden_config)
    assert env.force_max_steps() == 1
    assert env.core.record.reward == env.core.reward == 1
    assert env.core.termination_reason == "max_steps"


def test_generated_token_only_mask_and_logprob_alignment(adapter):
    prompt_ids = [10, 11, 12, 13]
    response_ids = [20, 21, 22]
    logprobs = [-0.1, -0.2, -0.3]
    step = adapter.AgentFlowStep(
        prompt_ids=prompt_ids,
        response_ids=response_ids,
        response_logprobs=logprobs,
        response_mask=[1] * len(response_ids),
    )
    assert step.response_mask == [1, 1, 1]
    assert len(step.response_logprobs) == len(step.response_ids)
    combined_mask = [0] * len(prompt_ids) + step.response_mask
    assert combined_mask == [0, 0, 0, 0, 1, 1, 1]


def test_logprob_mismatch_guard_is_present(adapter):
    source = open(adapter.__file__, encoding="utf-8").read()
    assert 'len(logprobs) != len(response_ids)' in source
    assert "response_logprobs must align one-to-one" in source
