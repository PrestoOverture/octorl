"""Pure-Python state machine for the tool-recovery protocol's three tools."""

from __future__ import annotations

import json
from typing import Any

from .generator import FAMILY_BY_NAME, ConditionalFieldDef, FieldDef
from .records import Instance, TrajectoryRecord, TrajectoryStep
from .reward import binary_reward, continuous_reward


class ToolRecoveryEnvironment:
    TOOL_CALL_LIMIT = 5

    def __init__(self, instance: Instance | None = None, *, model_sampling_seed: int | None = None):
        self.model_sampling_seed = model_sampling_seed
        self.instance: Instance
        if instance is not None:
            self.reset(instance)

    def reset(self, instance: Instance) -> dict[str, Any]:
        self.instance = instance
        self.family = FAMILY_BY_NAME[instance.family]
        self.config = dict(instance.presented_config)
        self.tool_calls = 0
        self.query_calls = 0
        self.has_read_config = False
        self.terminated = False
        self.termination_reason: str | None = None
        self.diagnostic_flags: set[str] = set()
        self._last_call: str | None = None
        self.record = TrajectoryRecord(instance.task_id, instance.seed, self.model_sampling_seed, instance.generator_version)
        return {"task_id": instance.task_id, "service": instance.family}

    @property
    def reward(self) -> int:
        return binary_reward(
            self.config,
            self.instance.golden_config,
            self.instance.presented_config,
            self.instance.fault_type,
            self.instance.faulted_field,
            self.family,
            self.instance.external_state,
        )

    @property
    def continuous_reward(self) -> float:
        return continuous_reward(self.config, self.instance.golden_config, self.family)

    def _finish(self, reason: str) -> tuple[dict[str, Any], bool, int]:
        self.terminated = True
        self.termination_reason = reason
        reward = self.reward
        self.record.reward = reward
        self.record.termination_reason = reason
        self.record.diagnostic_flags = set(self.diagnostic_flags)
        return {"status": "terminated", "reason": reason}, True, reward

    def text_response(self, text: str) -> tuple[dict[str, Any], bool, int]:
        if self.terminated:
            raise RuntimeError("episode already terminated")
        self.record.steps.append(TrajectoryStep(len(self.record.steps), "text", None, None, {"text": text}))
        return self._finish("text_response")

    def step(self, tool_name: str, arguments: dict[str, Any] | None = None) -> tuple[dict[str, Any], bool, int | None]:
        if self.terminated:
            raise RuntimeError("episode already terminated")
        arguments = {} if arguments is None else arguments
        canonical = json.dumps([tool_name, arguments], sort_keys=True, ensure_ascii=True, default=repr)
        if canonical == self._last_call:
            return self._finish("duplicate_tool_call")
        self._last_call = canonical
        if not self._valid_call(tool_name, arguments):
            self.diagnostic_flags.add("invalid_action")
            return self._finish("invalid_action")
        self.tool_calls += 1
        try:
            if tool_name == "read_config":
                observation = self.read_config()
            elif tool_name == "query_info":
                observation = self.query_info(arguments["topic"])
            else:
                observation = self.submit_fix(arguments["field"], arguments["value"])
        except Exception:
            self.diagnostic_flags.add("infra_error")
            return self._finish("infra_error")
        self.record.steps.append(TrajectoryStep(len(self.record.steps), "tool", tool_name, dict(arguments), observation))
        if self.tool_calls >= self.TOOL_CALL_LIMIT:
            terminated = self._finish("tool_call_limit")
            return observation, terminated[1], terminated[2]
        return observation, False, None

    @staticmethod
    def _valid_call(tool_name: str, arguments: Any) -> bool:
        if not isinstance(arguments, dict):
            return False
        if tool_name == "read_config":
            return arguments == {}
        if tool_name == "query_info":
            return set(arguments) == {"topic"} and isinstance(arguments["topic"], str)
        if tool_name == "submit_fix":
            return set(arguments) == {"field", "value"} and isinstance(arguments["field"], str)
        return False

    def _field_error(self, field: FieldDef | ConditionalFieldDef) -> dict[str, str] | None:
        name = field.name
        required = name in self.instance.golden_config
        if required and name not in self.config:
            return {"field": name, "message": "required field is missing", "code": "MISSING_FIELD"}
        if not required or name not in self.config:
            return None
        expected = field.compute(self.instance.external_state)
        if type(self.config[name]) is not type(expected) or self.config[name] != expected:
            if self.instance.fault_type == "stale_version" and name == self.instance.faulted_field:
                return {"field": name, "message": "value is stale after dependency update", "code": "STALE_DEPENDENCY"}
            return {"field": name, "message": "value does not satisfy constraint", "code": "CONSTRAINT_VIOLATION"}
        return None

    def read_config(self) -> dict[str, Any]:
        self.has_read_config = True
        errors = [error for field in [*self.family.fields, *self.family.conditional_fields] if (error := self._field_error(field))]
        return {"config": dict(self.config), "errors": errors, "status": "error" if errors else "ok"}

    def query_info(self, topic: str) -> dict[str, Any]:
        self.query_calls += 1
        if self.query_calls > 3:
            self.diagnostic_flags.add("excessive_queries")
        if topic == "fields":
            return {"fields": [{"name": field.name, "type": field.type} for field in [*self.family.fields, *self.family.conditional_fields]]}
        for field in self.family.fields:
            if field.name == topic:
                return {
                    "name": field.name,
                    "type": field.type,
                    "constraint": field.constraint_text,
                    "dependencies": [field.state_var],
                    "current_value": {field.state_var: self.instance.external_state[field.state_var]},
                }
        for field in self.family.conditional_fields:
            if field.name == topic:
                trigger = repr(field.trigger_value) if isinstance(field.trigger_value, str) else json.dumps(field.trigger_value)
                return {
                    "name": field.name,
                    "type": field.type,
                    "constraint": field.constraint_text,
                    "dependencies": [field.state_var],
                    "current_value": {field.state_var: self.instance.external_state[field.state_var]},
                    "conditional": f"required when {field.trigger_field} == {trigger}",
                }
        for state_var in self.family.state_vars:
            if state_var.name == topic:
                return {"name": state_var.name, "value": self.instance.external_state[state_var.name]}
        return {"error": "unknown_topic", "message": f"No information available for '{topic}'."}

    def submit_fix(self, field: str, value: Any) -> dict[str, str]:
        schema_names = {item.name for item in [*self.family.fields, *self.family.conditional_fields]}
        if field not in schema_names:
            self.diagnostic_flags.add("field_not_in_schema")
            return {"status": "error", "message": f"Field '{field}' does not exist in the schema."}
        if not self.has_read_config:
            self.diagnostic_flags.add("no_read_config")
        self.config[field] = value
        return {"status": "ok", "message": "Field updated."}


ServiceConfigurationRepairEnv = ToolRecoveryEnvironment
