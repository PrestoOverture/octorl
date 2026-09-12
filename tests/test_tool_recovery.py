"""Acceptance and structural tests for the r0.4 tool-recovery protocol."""

from __future__ import annotations

import pytest

from src.tasks.tool_recovery.environment import ToolRecoveryEnvironment
from src.tasks.tool_recovery.generator import (
    CACHE_SERVICE,
    FAMILIES,
    PROTOCOL_VERSION,
    allocate_slots,
    content_fingerprint,
    generate_dataset,
    generate_instance,
    is_modular,
)
from src.tasks.tool_recovery.records import Instance
from src.tasks.tool_recovery.verifier import verify


CACHE_STATE = {
    "system.total_memory_mb": 16384,
    "system.cpu_count": 8,
    "backend.cache_version": 2,
    "cache.ttl_multiplier": 4,
    "infra.max_pool_size": 800,
    "backend.compression_level": 2,
    "cache.freq_base": 30,
}


def cache_instance(fault_type: str, field: str | None, state: dict, golden: dict, presented: dict) -> Instance:
    fingerprint = content_fingerprint("cache_service", presented, state, fault_type, field)
    return Instance(f"walkthrough_{fault_type}", 1101, "cache_service", fault_type, field, golden, presented, state, fingerprint, PROTOCOL_VERSION)


def base_cache(state: dict) -> dict:
    golden = {field.name: field.compute(state) for field in CACHE_SERVICE.fields}
    for field in CACHE_SERVICE.conditional_fields:
        if golden[field.trigger_field] == field.trigger_value:
            golden[field.name] = field.compute(state)
    return golden


def finish(env: ToolRecoveryEnvironment) -> int:
    _, terminated, reward = env.text_response("done")
    assert terminated
    assert reward is not None
    return reward


def test_walkthrough_11_1() -> None:
    golden = base_cache(CACHE_STATE)
    presented = dict(golden, max_memory_mb=8192)
    env = ToolRecoveryEnvironment(cache_instance("constraint_violation", "max_memory_mb", CACHE_STATE, golden, presented))
    read, _, _ = env.step("read_config", {})
    assert read["errors"] == [{"field": "max_memory_mb", "message": "value does not satisfy constraint", "code": "CONSTRAINT_VIOLATION"}]
    info, _, _ = env.step("query_info", {"topic": "max_memory_mb"})
    assert info["dependencies"] == ["system.total_memory_mb"]
    state, _, _ = env.step("query_info", {"topic": "system.total_memory_mb"})
    assert state["value"] == 16384
    result, _, _ = env.step("submit_fix", {"field": "max_memory_mb", "value": 4096})
    assert result == {"status": "ok", "message": "Field updated."}
    assert finish(env) == 1


def test_walkthrough_11_2() -> None:
    golden = base_cache(CACHE_STATE)
    presented = dict(golden, max_memory_mb=8192)
    env = ToolRecoveryEnvironment(cache_instance("constraint_violation", "max_memory_mb", CACHE_STATE, golden, presented))
    env.step("read_config", {})
    env.step("submit_fix", {"field": "max_memory_mb", "value": 1})
    assert finish(env) == 0


def test_walkthrough_11_3() -> None:
    state = dict(CACHE_STATE, **{"backend.cache_version": 3})
    golden = base_cache(state)
    presented = dict(golden)
    del presented["frequency_window_seconds"]
    env = ToolRecoveryEnvironment(cache_instance("missing_dependency", "frequency_window_seconds", state, golden, presented))
    read, _, _ = env.step("read_config", {})
    assert read["errors"][0]["code"] == "MISSING_FIELD"
    info, _, _ = env.step("query_info", {"topic": "frequency_window_seconds"})
    assert info["conditional"] == "required when eviction_policy == 'lfu'"
    queried, _, _ = env.step("query_info", {"topic": "cache.freq_base"})
    assert queried["value"] == 30
    env.step("submit_fix", {"field": "frequency_window_seconds", "value": 120})
    assert finish(env) == 1


def test_walkthrough_11_4() -> None:
    state = dict(CACHE_STATE, **{"backend.cache_version": 3})
    golden = base_cache(state)
    presented = dict(golden)
    del presented["frequency_window_seconds"]
    env = ToolRecoveryEnvironment(cache_instance("missing_dependency", "frequency_window_seconds", state, golden, presented))
    env.step("read_config", {})
    assert finish(env) == 0


def stale_compression_instance() -> Instance:
    old_state = dict(CACHE_STATE, **{"backend.compression_level": 1})
    presented = base_cache(old_state)
    new_state = dict(old_state, **{"backend.compression_level": 2})
    golden = base_cache(new_state)
    return cache_instance("stale_version", "compression_enabled", new_state, golden, presented)


def test_walkthrough_11_5() -> None:
    env = ToolRecoveryEnvironment(stale_compression_instance())
    read, _, _ = env.step("read_config", {})
    assert read["errors"][0]["code"] == "STALE_DEPENDENCY"
    info, _, _ = env.step("query_info", {"topic": "compression_enabled"})
    assert info["dependencies"] == ["backend.compression_level"]
    queried, _, _ = env.step("query_info", {"topic": "backend.compression_level"})
    assert queried["value"] == 2
    env.step("submit_fix", {"field": "compression_enabled", "value": True})
    assert finish(env) == 1


def test_walkthrough_11_6() -> None:
    env = ToolRecoveryEnvironment(stale_compression_instance())
    env.step("read_config", {})
    env.step("submit_fix", {"field": "compression_enabled", "value": 1})
    assert finish(env) == 0


def test_walkthrough_11_7() -> None:
    golden = base_cache(CACHE_STATE)
    env = ToolRecoveryEnvironment(cache_instance("normal", None, CACHE_STATE, golden, golden))
    read, _, _ = env.step("read_config", {})
    assert read["status"] == "ok" and read["errors"] == []
    assert finish(env) == 1


def test_walkthrough_11_8() -> None:
    golden = base_cache(CACHE_STATE)
    env = ToolRecoveryEnvironment(cache_instance("normal", None, CACHE_STATE, golden, golden))
    env.step("read_config", {})
    env.step("submit_fix", {"field": "max_memory_mb", "value": 9999})
    assert finish(env) == 0


def test_isolation_invariant() -> None:
    for family in FAMILIES:
        fields = [*family.fields, *family.conditional_fields]
        assert len({field.state_var for field in fields}) == 7 == len(family.state_vars)
        assert {field.state_var for field in fields} == {state.name for state in family.state_vars}


def test_all_family_contract_shapes_and_patterns() -> None:
    expected_triggers = {
        "storage_service": ("compression_type", "zstd"),
        "network_service": ("proxy_protocol", "v2"),
        "monitoring_service": ("aggregation_method", "histogram"),
        "scheduler_service": ("retry_strategy", "exponential"),
        "deployment_service": ("deploy_strategy", "canary"),
        "notification_service": ("channel_priority", "urgent"),
    }
    for index, family in enumerate(FAMILIES):
        assert (len(family.fields), len(family.conditional_fields), len(family.state_vars)) == (6, 1, 7)
        triggers = [field for field in family.fields if field.is_trigger]
        assert len(triggers) == 1
        assert all(field.values_for_invalid for field in [*family.fields, *family.conditional_fields])
        modular_fields = [field for field in family.fields if is_modular(field)]
        if index < 6:
            assert modular_fields == []
        else:
            assert modular_fields and all(not field.is_trigger for field in modular_fields)
            for field in modular_fields:
                state_var = family.get_state_var(field.state_var)
                assert len({field.compute({field.state_var: value}) for value in state_var.value_range}) >= 2
        if family.name in expected_triggers:
            expected_name, expected_value = expected_triggers[family.name]
            assert triggers[0].name == expected_name
            assert family.conditional_fields[0].trigger_field == expected_name
            assert family.conditional_fields[0].trigger_value == expected_value


def test_trigger_exclusion_for_1000_seeds() -> None:
    for family in FAMILIES:
        trigger_names = {field.name for field in family.fields if field.is_trigger}
        for seed in range(1000):
            for fault in ("constraint_violation", "stale_version"):
                instance = generate_instance(seed, [family], fault)
                if instance is not None:
                    assert instance.faulted_field not in trigger_names


def test_invalid_pool_correctness_after_canonical_filter() -> None:
    # r0.4 defines Boolean pools as [False, True], so correctness is necessarily
    # per-instance: the canonical generator filters the current correct value.
    for family in FAMILIES:
        for field in [*family.fields, *family.conditional_fields]:
            state_var = family.get_state_var(field.state_var)
            for state_value in state_var.value_range:
                state = {item.name: item.value_range[0] for item in family.state_vars}
                state[state_var.name] = state_value
                correct = field.compute(state)
                filtered = [value for value in field.values_for_invalid if value != correct]
                assert all(value != correct for value in filtered)
                assert filtered, f"{family.name}.{field.name} has no usable invalid value"


def test_solvability_200_instances() -> None:
    generated = []
    seed = 0
    while len(generated) < 200:
        fault = ("normal", "constraint_violation", "missing_dependency", "stale_version")[len(generated) % 4]
        instance = generate_instance(seed, FAMILIES, fault)
        seed += 1
        if instance is None:
            continue
        generated.append(instance)
    assert len(generated) == 200
    assert {instance.fault_type for instance in generated} == {
        "normal", "constraint_violation", "missing_dependency", "stale_version"
    }
    for instance in generated:
        fault = instance.fault_type
        env = ToolRecoveryEnvironment(instance)
        env.step("read_config", {})
        if fault == "normal":
            assert finish(env) == 1
            continue
        field = env.family.get_field(instance.faulted_field)
        env.step("query_info", {"topic": field.name})
        env.step("query_info", {"topic": field.state_var})
        env.step("submit_fix", {"field": field.name, "value": instance.golden_config[field.name]})
        assert finish(env) == 1


@pytest.mark.parametrize("count", [0, 1, 3, 4, 50, 100, 1000])
def test_allocator_sums_to_requested_count(count: int) -> None:
    assert sum(allocate_slots(count).values()) == count


def test_allocator_50() -> None:
    assert allocate_slots(50) == {"normal": 10, "constraint_violation": 15, "missing_dependency": 13, "stale_version": 12}


def test_modular_holdout() -> None:
    manifest = generate_dataset("dev", 50, 100000)
    modular_faults = {
        instance.fault_type
        for instance in manifest.instances
        if instance.faulted_field and is_modular(FAMILIES[6 if instance.family == "monitoring_service" else 7].get_field(instance.faulted_field))
    }
    assert {"constraint_violation", "stale_version"} <= modular_faults


def test_content_fingerprint_dedup() -> None:
    manifest = generate_dataset("train", 100, 0)
    fingerprints = [instance.fingerprint for instance in manifest.instances]
    assert len(fingerprints) == len(set(fingerprints)) == 100


def test_strict_type_checking() -> None:
    instance = stale_compression_instance()
    final = dict(instance.presented_config, compression_enabled=1)
    assert verify(final, instance.golden_config, instance.presented_config, instance.fault_type, instance.faulted_field, CACHE_SERVICE, instance.external_state) == 0


def test_unknown_field_does_not_mutate_state() -> None:
    golden = base_cache(CACHE_STATE)
    env = ToolRecoveryEnvironment(cache_instance("normal", None, CACHE_STATE, golden, golden))
    before = dict(env.config)
    result, _, _ = env.step("submit_fix", {"field": "unknown", "value": 3})
    assert result["status"] == "error"
    assert env.config == before
    assert "field_not_in_schema" in env.diagnostic_flags


def test_termination_and_diagnostic_rules() -> None:
    golden = base_cache(CACHE_STATE)
    instance = cache_instance("normal", None, CACHE_STATE, golden, golden)

    duplicate = ToolRecoveryEnvironment(instance)
    duplicate.step("read_config", {})
    _, terminated, reward = duplicate.step("read_config", {})
    assert terminated and reward == 1 and duplicate.termination_reason == "duplicate_tool_call"

    malformed = ToolRecoveryEnvironment(instance)
    _, terminated, reward = malformed.step("query_info", {})
    assert terminated and reward == 1 and "invalid_action" in malformed.diagnostic_flags

    limited = ToolRecoveryEnvironment(instance)
    for topic in ("fields", "max_memory_mb", "system.total_memory_mb", "unknown"):
        _, terminated, _ = limited.step("query_info", {"topic": topic})
        assert not terminated
    _, terminated, reward = limited.step("read_config", {})
    assert terminated and reward == 1 and limited.termination_reason == "tool_call_limit"
    assert "excessive_queries" in limited.diagnostic_flags
