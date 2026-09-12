"""Deterministic Service Configuration Repair instance generation."""

from __future__ import annotations

import dataclasses
import hashlib
import json
import random
from collections.abc import Callable
from typing import Any

from .records import GenerationManifest, Instance, RejectionRecord
from .verifier import verify, verify_golden

PROTOCOL_VERSION = "r0.4"
FAULT_TYPES = ("normal", "constraint_violation", "missing_dependency", "stale_version")


@dataclasses.dataclass(frozen=True)
class FieldDef:
    name: str
    type: str
    constraint_text: str
    compute: Callable[[dict], Any]
    state_var: str
    values_for_invalid: list[Any]
    is_trigger: bool = False


@dataclasses.dataclass(frozen=True)
class ConditionalFieldDef:
    name: str
    type: str
    constraint_text: str
    compute: Callable[[dict], Any]
    state_var: str
    values_for_invalid: list[Any]
    trigger_field: str
    trigger_value: Any


@dataclasses.dataclass(frozen=True)
class StateVarDef:
    name: str
    type: str
    value_range: list[Any]


@dataclasses.dataclass(frozen=True)
class ContentFamily:
    name: str
    fields: list[FieldDef]
    conditional_fields: list[ConditionalFieldDef]
    state_vars: list[StateVarDef]

    def get_state_var(self, name: str) -> StateVarDef:
        return next(state_var for state_var in self.state_vars if state_var.name == name)

    def get_field(self, name: str) -> FieldDef | ConditionalFieldDef:
        return next(field for field in [*self.fields, *self.conditional_fields] if field.name == name)


def _sv(name: str, values: list[Any], type_name: str = "integer") -> StateVarDef:
    return StateVarDef(name, type_name, values)


def _field(
    name: str,
    type_name: str,
    text: str,
    state_var: str,
    function: Callable[[Any], Any],
    invalid: list[Any],
    trigger: bool = False,
) -> FieldDef:
    return FieldDef(name, type_name, text, lambda state, s=state_var, fn=function: fn(state[s]), state_var, invalid, trigger)


def _conditional(
    name: str,
    type_name: str,
    text: str,
    state_var: str,
    function: Callable[[Any], Any],
    invalid: list[Any],
    trigger_field: str,
    trigger_value: Any,
) -> ConditionalFieldDef:
    return ConditionalFieldDef(
        name, type_name, text, lambda state, s=state_var, fn=function: fn(state[s]),
        state_var, invalid, trigger_field, trigger_value
    )


CACHE_SERVICE = ContentFamily(
    "cache_service",
    [
        _field("max_memory_mb", "integer", "must equal floor(system.total_memory_mb / 4)", "system.total_memory_mb", lambda x: x // 4, [512, 2048, 8192, 65536]),
        _field("shard_count", "integer", "must equal system.cpu_count", "system.cpu_count", lambda x: x, [1, 3, 6, 32]),
        _field("eviction_policy", "string", 'must be "lru" if backend.cache_version < 3, otherwise "lfu"', "backend.cache_version", lambda x: "lru" if x < 3 else "lfu", ["fifo", "random", "mru"], True),
        _field("ttl_seconds", "integer", "must equal cache.ttl_multiplier × 60", "cache.ttl_multiplier", lambda x: x * 60, [30, 90, 1800, 7200]),
        _field("max_connections", "integer", "must equal floor(infra.max_pool_size / 2)", "infra.max_pool_size", lambda x: x // 2, [50, 150, 999, 3200]),
        _field("compression_enabled", "boolean", "must be true if backend.compression_level >= 2, otherwise false", "backend.compression_level", lambda x: x >= 2, [False, True]),
    ],
    [_conditional("frequency_window_seconds", "integer", "must equal cache.freq_base × 4", "cache.freq_base", lambda x: x * 4, [5, 15, 100, 300], "eviction_policy", "lfu")],
    [
        _sv("system.total_memory_mb", [4096, 8192, 16384, 32768]), _sv("system.cpu_count", [2, 4, 8, 16]),
        _sv("backend.cache_version", [1, 2, 3]), _sv("cache.ttl_multiplier", [1, 2, 4, 8]),
        _sv("infra.max_pool_size", [200, 400, 800, 1600]), _sv("backend.compression_level", [1, 2, 3]),
        _sv("cache.freq_base", [10, 20, 30, 60]),
    ],
)

QUEUE_SERVICE = ContentFamily(
    "queue_service",
    [
        _field("batch_size", "integer", "must equal floor(backend.max_batch_capacity / 2)", "backend.max_batch_capacity", lambda x: x // 2, [3, 12, 48, 100]),
        _field("retry_count", "integer", "must equal queue.retry_base × 2", "queue.retry_base", lambda x: x * 2, [1, 3, 5, 11]),
        _field("dead_letter_enabled", "boolean", "must be true if backend.api_version >= 3, otherwise false", "backend.api_version", lambda x: x >= 3, [False, True], True),
        _field("timeout_ms", "integer", "must equal queue.timeout_factor × 8", "queue.timeout_factor", lambda x: x * 8, [100, 500, 3000, 9000]),
        _field("max_queue_size", "integer", "must equal queue.capacity_multiplier × 64", "queue.capacity_multiplier", lambda x: x * 64, [100, 700, 1500, 3000]),
        _field("serialization_format", "string", 'must be "json" if backend.protocol_version == 1, "msgpack" if 2, "protobuf" if 3', "backend.protocol_version", lambda x: {1: "json", 2: "msgpack", 3: "protobuf"}[x], ["xml", "yaml", "avro"]),
    ],
    [_conditional("dead_letter_queue_size", "integer", "must equal queue.dlq_factor × 16", "queue.dlq_factor", lambda x: x * 16, [10, 50, 100, 200], "dead_letter_enabled", True)],
    [
        _sv("backend.max_batch_capacity", [16, 32, 64, 128]), _sv("queue.retry_base", [1, 2, 3, 4]),
        _sv("backend.api_version", [1, 2, 3, 4]), _sv("queue.timeout_factor", [100, 250, 500, 1000]),
        _sv("queue.capacity_multiplier", [4, 8, 16, 32]), _sv("backend.protocol_version", [1, 2, 3]),
        _sv("queue.dlq_factor", [2, 4, 8]),
    ],
)

LOGGING_SERVICE = ContentFamily(
    "logging_service",
    [
        _field("max_file_size_mb", "integer", "must equal floor(storage.max_log_quota_mb / 10)", "storage.max_log_quota_mb", lambda x: x // 10, [50, 150, 500, 900]),
        _field("rotation_count", "integer", "must equal logging.rotation_factor × 2", "logging.rotation_factor", lambda x: x * 2, [1, 7, 25, 50]),
        _field("async_enabled", "boolean", "must be true if system.async_threshold <= 2, otherwise false", "system.async_threshold", lambda x: x <= 2, [False, True], True),
        _field("log_level", "string", 'must be "debug" if infra.log_api_version == 1, "info" if 2, "warn" if 3', "infra.log_api_version", lambda x: {1: "debug", 2: "info", 3: "warn"}[x], ["trace", "error", "fatal"]),
        _field("output_format", "string", 'must be "text" if logging.format_version == 1, "json" if 2, "structured" if 3', "logging.format_version", lambda x: {1: "text", 2: "json", 3: "structured"}[x], ["xml", "csv", "binary"]),
        _field("buffer_size_kb", "integer", "must equal logging.buffer_multiplier × 4", "logging.buffer_multiplier", lambda x: x * 4, [5, 20, 50, 200]),
    ],
    [_conditional("flush_interval_ms", "integer", "must equal logging.flush_base × 2", "logging.flush_base", lambda x: x * 2, [50, 300, 750, 1200], "async_enabled", True)],
    [
        _sv("storage.max_log_quota_mb", [1000, 2000, 4000, 8000]), _sv("logging.rotation_factor", [3, 5, 10, 15]),
        _sv("system.async_threshold", [1, 2, 4]), _sv("infra.log_api_version", [1, 2, 3]),
        _sv("logging.format_version", [1, 2, 3]), _sv("logging.buffer_multiplier", [4, 8, 16, 32]),
        _sv("logging.flush_base", [100, 200, 500]),
    ],
)

AUTH_SERVICE = ContentFamily(
    "auth_service",
    [
        _field("hash_algorithm", "string", 'must be "sha256" if security.hash_standard == 1, "sha384" if 2, "sha512" if 3', "security.hash_standard", lambda x: {1: "sha256", 2: "sha384", 3: "sha512"}[x], ["md5", "sha1", "bcrypt"]),
        _field("token_expiry_hours", "integer", "must equal floor(24 / auth.expiry_divisor)", "auth.expiry_divisor", lambda x: 24 // x, [1, 5, 10, 48]),
        _field("require_mfa", "boolean", "must be true if security.policy_version >= 3, otherwise false", "security.policy_version", lambda x: x >= 3, [False, True], True),
        _field("max_sessions", "integer", "must equal floor(infra.session_pool_size / 4)", "infra.session_pool_size", lambda x: x // 4, [8, 40, 100, 200]),
        _field("session_timeout_minutes", "integer", "must equal auth.timeout_base × 2", "auth.timeout_base", lambda x: x * 2, [10, 45, 100, 300]),
        _field("password_min_length", "integer", "must equal security.password_factor × 4", "security.password_factor", lambda x: x * 4, [6, 10, 18, 24]),
    ],
    [_conditional("mfa_timeout_seconds", "integer", "must equal auth.mfa_base × 2", "auth.mfa_base", lambda x: x * 2, [15, 75, 200, 300], "require_mfa", True)],
    [
        _sv("security.hash_standard", [1, 2, 3]), _sv("auth.expiry_divisor", [1, 2, 4, 6]),
        _sv("security.policy_version", [1, 2, 3, 4]), _sv("infra.session_pool_size", [64, 128, 256, 512]),
        _sv("auth.timeout_base", [15, 30, 60, 120]), _sv("security.password_factor", [2, 3, 4, 5]),
        _sv("auth.mfa_base", [30, 60, 90, 120]),
    ],
)


STORAGE_SERVICE = ContentFamily(
    "storage_service",
    [
        _field("block_size_kb", "integer", "must equal floor(storage.volume_capacity_gb / 8)", "storage.volume_capacity_gb", lambda x: x // 8, [4, 24, 48, 100]),
        _field("replica_count", "integer", "must equal storage.redundancy_level × 3", "storage.redundancy_level", lambda x: x * 3, [1, 5, 8, 15]),
        _field("compression_type", "string", 'must be "zstd" if storage.engine_version >= 3, otherwise "lz4"', "storage.engine_version", lambda x: "zstd" if x >= 3 else "lz4", ["gzip", "snappy", "brotli"], True),
        _field("iops_limit", "integer", "must equal storage.throughput_base × 16", "storage.throughput_base", lambda x: x * 16, [500, 1000, 5000, 8000]),
        _field("snapshot_interval_hours", "integer", "must equal floor(48 / storage.snapshot_divisor)", "storage.snapshot_divisor", lambda x: 48 // x, [1, 3, 8, 36]),
        _field("encryption_enabled", "boolean", "must be true if storage.security_tier >= 2, otherwise false", "storage.security_tier", lambda x: x >= 2, [False, True]),
    ],
    [_conditional("dedup_chunk_size_kb", "integer", "must equal storage.dedup_base × 8", "storage.dedup_base", lambda x: x * 8, [16, 48, 96, 300], "compression_type", "zstd")],
    [
        _sv("storage.volume_capacity_gb", [64, 128, 256, 512]), _sv("storage.redundancy_level", [1, 2, 3, 4]),
        _sv("storage.engine_version", [1, 2, 3, 4]), _sv("storage.throughput_base", [50, 100, 200, 400]),
        _sv("storage.snapshot_divisor", [1, 2, 4, 8]), _sv("storage.security_tier", [1, 2, 3]),
        _sv("storage.dedup_base", [4, 8, 16, 32]),
    ],
)

NETWORK_SERVICE = ContentFamily(
    "network_service",
    [
        _field("max_bandwidth_mbps", "integer", "must equal network.link_speed_gbps × 1000", "network.link_speed_gbps", lambda x: x * 1000, [500, 3000, 7500, 15000]),
        _field("connection_pool_size", "integer", "must equal floor(network.socket_pool / 4)", "network.socket_pool", lambda x: x // 4, [10, 75, 150, 300]),
        _field("proxy_protocol", "string", 'must be "v2" if network.proxy_version >= 2, otherwise "v1"', "network.proxy_version", lambda x: "v2" if x >= 2 else "v1", ["http", "socks5", "direct"], True),
        _field("dns_cache_ttl", "integer", "must equal network.ttl_multiplier × 30", "network.ttl_multiplier", lambda x: x * 30, [15, 45, 180, 600]),
        _field("routing_mode", "string", 'must be "round_robin" if network.lb_algorithm == 1, "least_conn" if 2, "ip_hash" if 3', "network.lb_algorithm", lambda x: {1: "round_robin", 2: "least_conn", 3: "ip_hash"}[x], ["random", "weighted", "sticky"]),
        _field("keepalive_enabled", "boolean", "must be true if network.idle_timeout_base <= 30, otherwise false", "network.idle_timeout_base", lambda x: x <= 30, [False, True]),
    ],
    [_conditional("proxy_buffer_kb", "integer", "must equal network.buffer_factor × 32", "network.buffer_factor", lambda x: x * 32, [16, 48, 96, 512], "proxy_protocol", "v2")],
    [
        _sv("network.link_speed_gbps", [1, 2, 5, 10]), _sv("network.socket_pool", [100, 200, 400, 800]),
        _sv("network.proxy_version", [1, 2, 3]), _sv("network.ttl_multiplier", [1, 2, 4, 10]),
        _sv("network.lb_algorithm", [1, 2, 3]), _sv("network.idle_timeout_base", [10, 30, 60, 120]),
        _sv("network.buffer_factor", [1, 2, 4, 8]),
    ],
)

MONITORING_SERVICE = ContentFamily(
    "monitoring_service",
    [
        _field("scrape_interval_seconds", "integer", "must equal (monitoring.scrape_base % 6) + 5", "monitoring.scrape_base", lambda x: (x % 6) + 5, [3, 9, 12, 15]),
        _field("data_retention_days", "integer", "must equal monitoring.retention_factor × 5", "monitoring.retention_factor", lambda x: x * 5, [3, 10, 28, 100]),
        _field("aggregation_method", "string", 'must be "histogram" if monitoring.alert_version >= 3, otherwise "counter"', "monitoring.alert_version", lambda x: "histogram" if x >= 3 else "counter", ["gauge", "summary", "timer"], True),
        _field("sample_rate", "integer", "must equal floor(monitoring.sample_pool / 5)", "monitoring.sample_pool", lambda x: x // 5, [10, 50, 300, 1500]),
        _field("widget_columns", "integer", "must equal monitoring.panel_columns × 3", "monitoring.panel_columns", lambda x: x * 3, [4, 8, 15, 24]),
        _field("alerting_enabled", "boolean", "must be true if monitoring.severity_threshold >= 3, otherwise false", "monitoring.severity_threshold", lambda x: x >= 3, [False, True]),
    ],
    [_conditional("agg_window_minutes", "integer", "must equal monitoring.agg_window_base × 2", "monitoring.agg_window_base", lambda x: x * 2, [3, 15, 45, 90], "aggregation_method", "histogram")],
    [
        _sv("monitoring.scrape_base", [8, 13, 19, 26]), _sv("monitoring.retention_factor", [1, 3, 7, 14]),
        _sv("monitoring.alert_version", [1, 2, 3]), _sv("monitoring.sample_pool", [100, 500, 1000, 5000]),
        _sv("monitoring.panel_columns", [2, 3, 4, 6]), _sv("monitoring.severity_threshold", [1, 2, 3, 4]),
        _sv("monitoring.agg_window_base", [5, 10, 15, 30]),
    ],
)

SCHEDULER_SERVICE = ContentFamily(
    "scheduler_service",
    [
        _field("max_pending_jobs", "integer", "must equal (scheduler.queue_depth % 8) + 2", "scheduler.queue_depth", lambda x: (x % 8) + 2, [1, 5, 9, 12]),
        _field("executor_threads", "integer", "must equal floor(scheduler.worker_pool / 8)", "scheduler.worker_pool", lambda x: x // 8, [1, 3, 6, 24]),
        _field("retry_strategy", "string", 'must be "exponential" if scheduler.retry_version >= 3, otherwise "linear"', "scheduler.retry_version", lambda x: "exponential" if x >= 3 else "linear", ["fibonacci", "constant", "jitter"], True),
        _field("priority_buckets", "integer", "must equal scheduler.priority_levels × 2", "scheduler.priority_levels", lambda x: x * 2, [4, 8, 12, 20]),
        _field("cron_resolution_minutes", "integer", "must equal scheduler.cron_base × 15", "scheduler.cron_base", lambda x: x * 15, [5, 20, 60, 200]),
        _field("max_concurrent", "integer", "must equal floor(scheduler.concurrency_cap / 2)", "scheduler.concurrency_cap", lambda x: x // 2, [1, 3, 8, 16]),
    ],
    [_conditional("grace_period_seconds", "integer", "must equal scheduler.grace_base × 5", "scheduler.grace_base", lambda x: x * 5, [20, 100, 250, 800], "retry_strategy", "exponential")],
    [
        _sv("scheduler.queue_depth", [12, 25, 37, 50]), _sv("scheduler.worker_pool", [16, 32, 64, 128]),
        _sv("scheduler.retry_version", [1, 2, 3]), _sv("scheduler.priority_levels", [3, 5, 7, 9]),
        _sv("scheduler.cron_base", [2, 3, 5, 10]), _sv("scheduler.concurrency_cap", [4, 8, 12, 24]),
        _sv("scheduler.grace_base", [10, 30, 60, 120]),
    ],
)

DEPLOYMENT_SERVICE = ContentFamily(
    "deployment_service",
    [
        _field("parallel_stages", "integer", "must equal (deployment.pipeline_stages % 5) + 1", "deployment.pipeline_stages", lambda x: (x % 5) + 1, [4, 6, 8, 10]),
        _field("max_replicas", "integer", "must equal floor(deployment.instance_capacity / 5)", "deployment.instance_capacity", lambda x: x // 5, [10, 30, 75, 300]),
        _field("deploy_strategy", "string", 'must be "canary" if deployment.strategy_version >= 3, otherwise "rolling"', "deployment.strategy_version", lambda x: "canary" if x >= 3 else "rolling", ["blue_green", "recreate", "shadow"], True),
        _field("rollback_timeout_minutes", "integer", "must equal deployment.rollback_base × 6", "deployment.rollback_base", lambda x: x * 6, [15, 45, 120, 240]),
        _field("health_check_interval", "integer", "must equal deployment.health_factor × 10", "deployment.health_factor", lambda x: x * 10, [5, 15, 40, 100]),
        _field("min_healthy_percent", "integer", "must equal deployment.healthy_pct_base × 25", "deployment.healthy_pct_base", lambda x: x * 25, [10, 30, 60, 90]),
    ],
    [_conditional("canary_weight_percent", "integer", "must equal deployment.canary_weight_base × 2", "deployment.canary_weight_base", lambda x: x * 2, [3, 15, 25, 60], "deploy_strategy", "canary")],
    [
        _sv("deployment.pipeline_stages", [6, 11, 17, 24]), _sv("deployment.instance_capacity", [100, 250, 500, 1000]),
        _sv("deployment.strategy_version", [1, 2, 3, 4]), _sv("deployment.rollback_base", [5, 10, 15, 30]),
        _sv("deployment.health_factor", [2, 3, 5, 8]), _sv("deployment.healthy_pct_base", [1, 2, 3, 4]),
        _sv("deployment.canary_weight_base", [5, 10, 15, 25]),
    ],
)

NOTIFICATION_SERVICE = ContentFamily(
    "notification_service",
    [
        _field("batch_delay_seconds", "integer", "must equal (notification.batch_window % 4) + 1", "notification.batch_window", lambda x: (x % 4) + 1, [4, 5, 7, 10]),
        _field("max_recipients", "integer", "must equal notification.channel_count × 20", "notification.channel_count", lambda x: x * 20, [30, 80, 200, 300]),
        _field("channel_priority", "string", 'must be "urgent" if notification.priority_version >= 3, otherwise "normal"', "notification.priority_version", lambda x: "urgent" if x >= 3 else "normal", ["low", "deferred", "batch"], True),
        _field("rate_limit_per_minute", "integer", "must equal floor(notification.rate_base / 2)", "notification.rate_base", lambda x: x // 2, [3, 15, 75, 300]),
        _field("template_format", "string", 'must be "text" if notification.template_version == 1, "html" if 2, "markdown" if 3, "rich" if 4', "notification.template_version", lambda x: {1: "text", 2: "html", 3: "markdown", 4: "rich"}[x], ["xml", "pdf", "csv"]),
        _field("cooldown_enabled", "boolean", "must be true if notification.cooldown_base >= 120, otherwise false", "notification.cooldown_base", lambda x: x >= 120, [False, True]),
    ],
    [_conditional("escalation_timeout_minutes", "integer", "must equal notification.escalation_base × 4", "notification.escalation_base", lambda x: x * 4, [5, 15, 30, 80], "channel_priority", "urgent")],
    [
        _sv("notification.batch_window", [9, 14, 22, 33]), _sv("notification.channel_count", [3, 5, 8, 12]),
        _sv("notification.priority_version", [1, 2, 3]), _sv("notification.rate_base", [10, 50, 100, 500]),
        _sv("notification.template_version", [1, 2, 3, 4]), _sv("notification.cooldown_base", [30, 60, 120, 300]),
        _sv("notification.escalation_base", [3, 5, 10, 15]),
    ],
)

FAMILIES = [CACHE_SERVICE, QUEUE_SERVICE, LOGGING_SERVICE, AUTH_SERVICE, STORAGE_SERVICE, NETWORK_SERVICE, MONITORING_SERVICE, SCHEDULER_SERVICE, DEPLOYMENT_SERVICE, NOTIFICATION_SERVICE]
FAMILY_BY_NAME = {family.name: family for family in FAMILIES}
SPLIT_FAMILIES = {"train": FAMILIES[:6], "dev": FAMILIES[6:8], "test": FAMILIES[8:]}
SPLIT_SEED_RANGES = {"train": range(0, 100000), "dev": range(100000, 110000), "test": range(200000, 210000)}


def is_modular(field: FieldDef | ConditionalFieldDef) -> bool:
    return "%" in field.constraint_text


def content_fingerprint(family_name: str, config: dict[str, Any], state: dict[str, Any], fault_type: str, faulted_field: str | None) -> str:
    canonical = json.dumps({"family": family_name, "config": dict(sorted(config.items())), "state": dict(sorted(state.items())), "fault_type": fault_type, "faulted_field": faulted_field}, sort_keys=True, ensure_ascii=True)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]


def _oracle_reward(presented: dict[str, Any], golden: dict[str, Any], state: dict[str, Any], fault_type: str, faulted_field: str, family: ContentFamily) -> int:
    repaired = dict(presented)
    repaired[faulted_field] = golden[faulted_field]
    return verify(repaired, golden, presented, fault_type, faulted_field, family, state)


def generate_instance(master_seed: int, families: list[ContentFamily], target_fault: str) -> Instance | None:
    if not isinstance(master_seed, int):
        raise TypeError("master_seed must be an integer")
    if not families:
        raise ValueError("families must not be empty")
    if target_fault not in FAULT_TYPES:
        raise ValueError(f"unknown fault type: {target_fault}")
    rng = random.Random(master_seed)
    family = families[rng.randint(0, len(families) - 1)]
    state = {sv.name: rng.choice(sv.value_range) for sv in family.state_vars}
    golden = {field.name: field.compute(state) for field in family.fields}
    for field in family.conditional_fields:
        if golden[field.trigger_field] == field.trigger_value:
            golden[field.name] = field.compute(state)
    presented = dict(golden)
    faulted_field: str | None = None
    faulted_state = dict(state)

    if target_fault == "constraint_violation":
        eligible = [field for field in family.fields if not field.is_trigger]
        field = eligible[rng.randint(0, len(eligible) - 1)]
        valid_invalids = [value for value in field.values_for_invalid if value != golden[field.name]]
        if not valid_invalids:
            return None
        presented[field.name] = rng.choice(valid_invalids)
        faulted_field = field.name
    elif target_fault == "missing_dependency":
        eligible = [field for field in family.conditional_fields if field.name in golden]
        if not eligible:
            return None
        field = eligible[rng.randint(0, len(eligible) - 1)]
        del presented[field.name]
        faulted_field = field.name
    elif target_fault == "stale_version":
        eligible = [field for field in family.fields if not field.is_trigger]
        field = eligible[rng.randint(0, len(eligible) - 1)]
        state_var = family.get_state_var(field.state_var)
        old_value = state[state_var.name]
        new_values = [value for value in state_var.value_range if value != old_value]
        new_value = new_values[rng.randint(0, len(new_values) - 1)]
        new_state = dict(state)
        new_state[state_var.name] = new_value
        new_golden_value = field.compute(new_state)
        if new_golden_value == golden[field.name]:
            return None
        faulted_field = field.name
        faulted_state = new_state
        golden[field.name] = new_golden_value

    assert verify_golden(golden, family, faulted_state), "golden config invalid"
    if faulted_field:
        assert verify(presented, golden, presented, target_fault, faulted_field, family, faulted_state) == 0, "fault did not break verification"
        assert _oracle_reward(presented, golden, faulted_state, target_fault, faulted_field, family) == 1, "oracle repair failed"
    fingerprint = content_fingerprint(family.name, presented, faulted_state, target_fault, faulted_field)
    return Instance(f"{family.name}_{master_seed}_{target_fault}", master_seed, family.name, target_fault, faulted_field, golden, presented, faulted_state, fingerprint, PROTOCOL_VERSION)


def allocate_slots(count: int) -> dict[str, int]:
    if count < 0:
        raise ValueError("count must be non-negative")
    raw = {"normal": 20, "constraint_violation": 30, "missing_dependency": 25, "stale_version": 25}
    exact = {key: count * value / 100 for key, value in raw.items()}
    slots = {key: int(value) for key, value in exact.items()}
    remainder = count - sum(slots.values())
    by_fraction = sorted(raw, key=lambda key: -(exact[key] - slots[key]))
    for index in range(remainder):
        slots[by_fraction[index]] += 1
    assert sum(slots.values()) == count
    return slots


def generate_dataset(split: str, count: int, start_seed: int | None = None) -> GenerationManifest:
    if split not in SPLIT_FAMILIES:
        raise ValueError(f"unknown split: {split}")
    permitted = SPLIT_SEED_RANGES[split]
    seed = permitted.start if start_seed is None else start_seed
    if seed not in permitted:
        raise ValueError(f"seed {seed} outside {split} range")
    manifest = GenerationManifest(split, count, seed, PROTOCOL_VERSION)
    seen: set[str] = set()
    slot_faults = [fault for fault, slots in allocate_slots(count).items() for _ in range(slots)]
    for slot, fault in enumerate(slot_faults):
        tried: list[int] = []
        accepted = None
        reason = "generator rejection"
        for _ in range(10):
            if seed not in permitted:
                reason = "split seed range exhausted"
                break
            tried.append(seed)
            candidate = generate_instance(seed, SPLIT_FAMILIES[split], fault)
            seed += 1
            if candidate is None:
                continue
            if candidate.fingerprint in seen:
                manifest.duplicate_seeds.append(candidate.seed)
                reason = "duplicate content fingerprint"
                continue
            accepted = candidate
            break
        if accepted is None:
            manifest.rejections.append(RejectionRecord(slot, fault, tuple(tried), reason))
        else:
            seen.add(accepted.fingerprint)
            manifest.instances.append(accepted)
    if len(manifest.rejections) > count * 0.1:
        manifest.warnings.append("more than 10% of slots were rejected")
    if split in {"dev", "test"}:
        combinations = {(instance.fault_type, instance.faulted_field) for instance in manifest.instances if instance.faulted_field and is_modular(FAMILY_BY_NAME[instance.family].get_field(instance.faulted_field))}
        missing = [fault for fault in ("constraint_violation", "stale_version") if not any(item[0] == fault for item in combinations)]
        if missing:
            raise ValueError(f"{split} dataset lacks modular stratum for: {', '.join(missing)}")
    return manifest


def validate_family_structure(family: ContentFamily) -> None:
    if len(family.fields) != 6 or len(family.conditional_fields) != 1 or len(family.state_vars) != 7:
        raise ValueError(f"{family.name}: expected 6 main, 1 conditional, and 7 state variables")
    field_states = {field.state_var for field in [*family.fields, *family.conditional_fields]}
    state_names = {state.name for state in family.state_vars}
    if field_states != state_names or len(field_states) != 7:
        raise ValueError(f"{family.name}: state variables are not isolated")
    if sum(field.is_trigger for field in family.fields) != 1:
        raise ValueError(f"{family.name}: expected exactly one trigger field")


for _family in FAMILIES:
    validate_family_structure(_family)
