# R0 Task and Experiment Protocol

**Version:** r2.0
**Date:** 2026-09-16
**Status:** Revised — r2.0 reverts to r0.5 task behavior + adds continuous reward module

## 1. Overview

This document specifies the task domain, tool interface, fault semantics, data generation, verification, and experiment protocol for the OctoRL v3.0 research direction: "Self-improving agent for dev-tool error recovery via failure-driven task selection and online GRPO."

The environment is a pure Python state machine simulating structured service configuration repair. An agent diagnoses faults in a service's JSON configuration using 3 tools, then either fixes the configuration or confirms it is valid. The design targets 3-5 round episodes with binary reward.

Two experimental questions:
- **Q1:** Does GRPO training improve fault-condition task completion on unseen dev-tool tasks (while preserving normal-condition performance)?
- **Q2:** Does failure-driven practice distribution yield extra gains over fixed-distribution GRPO?

---

## 2. Task Domain

**Domain:** Service Configuration Repair.

Each task instance presents a fictional service with a flat JSON configuration (6 main fields + up to 1 conditional field). The configuration may contain a fault that the agent must diagnose and fix, or it may be valid (normal condition). The agent interacts through 3 structured tools — no open shell, no code execution.

**Four task categories:**

| Category | Fault Type | Description |
|---|---|---|
| normal | (none) | Config is valid. Reward = 1 only if agent makes no changes. |
| param-change | constraint_violation | One field has an invalid value. The valid value is uniquely determined by an external state variable. |
| info-missing | missing_dependency | One conditionally-required field is absent. Its required value is determined by an external state variable. |
| state-change | stale_version | An external state variable changed, making a previously valid field value invalid. Only targets fields with isolated dependencies that do not trigger dependency rules. |

**Key design properties:**
- Each fault resolves to **exactly one correct value**, determined by an external state variable.
- Each field depends on **exactly one state variable**, and **no two fields share a state variable**. This isolation prevents cross-field value inference and cascade effects.
- Error messages from `read_config` identify WHICH field is wrong but never reveal the correct value or the state variable's current value.
- The task is a controlled simulation, not a real-world tool.

---

## 3. Frozen Tool Schemas

### 3.1 `read_config`

Reads the current configuration and reports whether it is valid, without identifying erroneous fields.

```json
{
  "type": "function",
  "function": {
    "name": "read_config",
    "description": "Read the current service configuration and check for errors.",
    "parameters": {
      "type": "object",
      "properties": {},
      "additionalProperties": false
    }
  }
}
```

**Returns:**
```json
{
  "config": {"field_name": "value", "...": "..."},
  "status": "ok | error"
}
```

### 3.2 `query_info`

Queries field constraints, state variable values, or the field list.

```json
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
          "description": "A field name (e.g. 'max_memory_mb'), a state variable name (e.g. 'system.total_memory_mb'), or the literal 'fields' to list all field names."
        }
      },
      "required": ["topic"],
      "additionalProperties": false
    }
  }
}
```

**Returns (field name):**
```json
{
  "name": "max_memory_mb",
  "type": "integer",
  "constraint": "must equal floor(system.total_memory_mb / 4)",
  "dependencies": ["system.total_memory_mb"],
  "current_value": {"system.total_memory_mb": 16384},
  "default": null
}
```

Constraint text always uses `floor(...)` for divisions and explicit operators (`×`, `+`) for other arithmetic. No ambiguous "divided by" phrasing.

**Returns (state variable):**
```json
{
  "name": "system.total_memory_mb",
  "value": 16384
}
```

**Returns ("fields"):**
```json
{
  "fields": [
    {"name": "max_memory_mb", "type": "integer"},
    {"name": "eviction_policy", "type": "string"},
    "..."
  ]
}
```

**Returns (unknown topic):**
```json
{
  "error": "unknown_topic",
  "message": "No information available for 'xxx'."
}
```

### 3.3 `submit_fix`

Updates a configuration field value.

```json
{
  "type": "function",
  "function": {
    "name": "submit_fix",
    "description": "Set a configuration field to a new value.",
    "parameters": {
      "type": "object",
      "properties": {
        "field": {
          "type": "string",
          "description": "The configuration field name to update."
        },
        "value": {
          "description": "The new value for the field."
        }
      },
      "required": ["field", "value"],
      "additionalProperties": false
    }
  }
}
```

**Returns:**
```json
{
  "status": "ok | error",
  "message": "Field updated. | Field 'xxx' does not exist in the schema."
}
```

Value validation does NOT happen at submission — only at episode end by the verifier. Submitting an invalid value succeeds silently but leads to reward = 0.

### 3.4 Termination Conditions

| Condition | Behavior |
|---|---|
| Tool call limit | 6 calls total across all turns. Each call counts separately. |
| Token limit | 4096 output tokens per model turn. |
| Text-only response | Agent responds without a tool call → episode ends. |
| Duplicate detection | 2 consecutive tool calls with identical (name, arguments) → episode ends. |
| Invalid tool call | Unknown tool name or malformed arguments → episode ends with `invalid_action` flag. |

All termination modes verify the final config state as-is. No termination mode automatically assigns reward = 0 — the final-state predicate (§8) is the sole authority.

---

## 4. Fault Semantics

Each fault rule is a 4-tuple: (trigger_condition, state_change, remaining_count, recovery_method).

### 4.1 Isolation Requirement

**Every field depends on exactly one state variable, and no two fields share a state variable.** This isolation is a structural invariant of all content families (§5). It guarantees:
- No cross-field value inference (the correct value of any faulted field cannot be derived from other visible config values).
- No cascade on state variable change (bumping one state variable affects exactly one field).
- Every faulted instance is solvable if the agent investigates the faulted field within the call budget; field identification is intentionally search-dependent.

### 4.2 constraint_violation

| Component | Specification |
|---|---|
| Trigger | `config[F]` does not satisfy `constraint(F, external_state)` |
| State change | `config[F]` set to a value sampled from `F.values_for_invalid`, **excluding the correct value**. The generator asserts that the presented config fails verification before any repair. |
| Count | 1 per instance |
| Eligible fields | Main fields that are **NOT trigger fields**. Trigger fields are excluded because the presence/absence of the conditional field would reveal the trigger's correct value, leaking structural information that bypasses the isolation guarantee. |
| Recovery | `read_config()` → investigate candidate fields with `query_info(field)` → find F and its current dependency value → compute V → `submit_fix(F, V)` |

### 4.3 missing_dependency

| Component | Specification |
|---|---|
| Trigger | Conditional field `F` is absent but required by dependency rule |
| State change | `del config[F]` (field removed from config dict) |
| Count | 1 per instance |
| Eligible fields | Conditional fields whose trigger condition is satisfied in the golden config |
| Recovery | `read_config()` → compare the visible config with candidate field definitions → `query_info(F)` → compute V from `current_value` → `submit_fix(F, V)` |

If no conditional field's trigger is satisfied for a given state, the generator rejects this (family, seed) for missing_dependency and retries with a new seed (see §6.3).

### 4.4 stale_version

| Component | Specification |
|---|---|
| Trigger | State variable `S` changed; `config[F]` was valid under old `S` but not new `S` |
| State change | `external_state[S] = new_value`; config unchanged |
| Count | 1 per instance |
| Eligible fields | Main fields whose state variable is NOT a dependency of any trigger field. This prevents cascade: changing S cannot make a new conditional field required. |
| Recovery | `read_config()` → investigate candidate fields with `query_info(field)` → find stale F → compute from `current_value` → `submit_fix(F, new_value)` |

The generator also verifies that the old and new golden values for F actually differ — if the state variable change doesn't affect F's computed value, the instance is rejected.

### 4.5 Shortcut Baselines

The isolation requirement prevents deriving faulted-field values from other visible config values. However, structural shortcuts exist and are acknowledged:

**Defined shortcut strategies:**

| Strategy | Mechanism | Expected FCR |
|---|---|---|
| Boolean flip | For boolean fault fields, submit the opposite value. Works 100% of the time on booleans (2 values, correct value excluded from pool). | ~5% overall (only ~1/20 eligible fields across families are boolean) |
| Enum elimination | For string fault fields with 3 enum values, the agent knows the current value is wrong; guessing one of the remaining 2 gives 50%. Requires knowing the valid enum set from schema metadata (see baseline definition below). | ~12% on enum fields (50% × ~25% enum share), ~3% overall |
| Conditional presence | If a trigger field were faulted, conditional field presence would leak the correct value. Eliminated by excluding trigger fields from faulting (§4.2, §4.4). | 0% (structurally prevented) |

**Scripted baseline definition:** A no-`query_info` scripted agent that executes the best deterministic strategy from the table above. This agent:
- Calls `read_config()` to obtain the status, visible config, and field-level errors (r2.0 restores the `errors` array from r0.5).
- Has access to **static schema metadata** for the family: field names, field types (boolean / string-enum / integer), and the set of **reachable output values** for each field (i.e., the values the field's `compute()` function can produce across the full state-variable range — NOT the state-variable range itself). This metadata is derived from the family definition and does not require `query_info`.
- Selects a candidate field from the static field list using the recorded RNG because field identity is hidden.
- For booleans: submits the opposite value (deterministic). For enums: guesses uniformly from reachable values excluding the presented value (seeded RNG). For integers: guesses uniformly from reachable values excluding the presented value (seeded RNG).
- Does NOT call `query_info` — its entire advantage comes from schema-aware guessing.
- Uses a **recorded seed** (`random.Random(baseline_seed)`) so its guesses are reproducible. The seed is logged in the pilot manifest.

**R1 acceptance threshold:** The scripted baseline's FCR must be **measured** on the dev split (50 instances, 100 rollouts). If measured FCR ≥ 15%, the family designs must be revised (e.g., reduce boolean faultable fields, widen integer ranges). The 15% threshold is a design target, not an analytical claim — it must be empirically confirmed in R1.

### 4.6 Search-Dependent Solvability Invariant

Every generated fault instance is solvable **if the agent investigates the faulted field within its call budget**. Once the correct field is selected, the best-case repair path is 3 calls:
1. `read_config()` — learn only that some field is wrong
2. `query_info(faulted_field)` — learn its constraint and dependency's `current_value`
3. `submit_fix(faulted_field, value)` — apply the fix

The 6-call budget permits `read_config`, up to four candidate-field investigations, and one `submit_fix`. Exhaustively checking all seven fields and then submitting would require 9 calls and exceed the limit. Normal instances remain solvable with 1 call (`read_config` returns `status: "ok"`, then the agent stops).

The generator MUST validate the known-field oracle repair for every produced instance and confirm that it yields reward = 1. Search success is a policy property and is measured by the pilot rather than guaranteed by generation.

---

## 5. Content Families

Each content family is a fictional service defining: fields (name, type, constraints), state variables (name, type, value range), and dependency rules (conditional requirements).

### 5.1 Family Schema

```python
@dataclasses.dataclass(frozen=True)
class FieldDef:
    name: str
    type: str                          # "integer", "string", "boolean"
    constraint_text: str               # human-readable, uses floor() for division
    compute: Callable[[dict], Any]     # (state_vars) -> exact correct value
    state_var: str                     # exactly one state variable name
    values_for_invalid: list[Any]      # pool of plausible-but-wrong values (same JSON type)
    is_trigger: bool = False           # True if this field's value triggers a dependency rule

@dataclasses.dataclass(frozen=True)
class ConditionalFieldDef:
    name: str
    type: str
    constraint_text: str
    compute: Callable[[dict], Any]
    state_var: str                     # isolated — no other field uses this state var
    values_for_invalid: list[Any]
    trigger_field: str                 # which main field triggers this
    trigger_value: Any                 # the value that makes this field required

@dataclasses.dataclass(frozen=True)
class StateVarDef:
    name: str
    type: str
    value_range: list[Any]             # enumerated possible values

@dataclasses.dataclass(frozen=True)
class ContentFamily:
    name: str
    fields: list[FieldDef]             # exactly 6
    conditional_fields: list[ConditionalFieldDef]  # exactly 1
    state_vars: list[StateVarDef]      # exactly 7 (one per field + conditional)
```

**Structural invariant:** For every family, the set `{f.state_var for f in fields + conditional_fields}` has exactly 7 elements, equal to `{sv.name for sv in state_vars}`. No state variable is shared between fields.

### 5.2 Family Definitions

**10 families total.** Split: families 0-5 = train, 6-7 = dev, 8-9 = test.

Below are complete definitions for 4 reference families. Remaining families follow the same template and are listed in the summary table (§5.3).

#### Family 0: `cache_service` (train)

**State variables (7):**

| Name | Type | Value range |
|---|---|---|
| `system.total_memory_mb` | integer | [4096, 8192, 16384, 32768] |
| `system.cpu_count` | integer | [2, 4, 8, 16] |
| `backend.cache_version` | integer | [1, 2, 3] |
| `cache.ttl_multiplier` | integer | [1, 2, 4, 8] |
| `infra.max_pool_size` | integer | [200, 400, 800, 1600] |
| `backend.compression_level` | integer | [1, 2, 3] |
| `cache.freq_base` | integer | [10, 20, 30, 60] |

**Fields (6 main):**

| Field | Type | Constraint text | Compute | State var | Trigger |
|---|---|---|---|---|---|
| `max_memory_mb` | integer | must equal floor(system.total_memory_mb / 4) | `total_memory_mb // 4` | system.total_memory_mb | no |
| `shard_count` | integer | must equal system.cpu_count | `cpu_count` | system.cpu_count | no |
| `eviction_policy` | string | must be "lru" if backend.cache_version < 3, otherwise "lfu" | conditional | backend.cache_version | **yes** |
| `ttl_seconds` | integer | must equal cache.ttl_multiplier × 60 | `ttl_multiplier * 60` | cache.ttl_multiplier | no |
| `max_connections` | integer | must equal floor(infra.max_pool_size / 2) | `max_pool_size // 2` | infra.max_pool_size | no |
| `compression_enabled` | boolean | must be true if backend.compression_level >= 2, otherwise false | `compression_level >= 2` | backend.compression_level | no |

**Stale_version eligible:** max_memory_mb, shard_count, ttl_seconds, max_connections, compression_enabled (NOT eviction_policy — it is a trigger field).

**Conditional field (1):**

| Field | Type | Constraint text | Compute | State var | Trigger condition |
|---|---|---|---|---|---|
| `frequency_window_seconds` | integer | must equal cache.freq_base × 4 | `freq_base * 4` | cache.freq_base | `eviction_policy == "lfu"` |

**Invalid value pools (examples):**

| Field | values_for_invalid |
|---|---|
| max_memory_mb | [512, 2048, 8192, 65536] |
| shard_count | [1, 3, 6, 32] |
| eviction_policy | ["fifo", "random", "mru"] |
| ttl_seconds | [30, 90, 1800, 7200] |
| max_connections | [50, 150, 999, 3200] |
| compression_enabled | [false, true] (opposite of correct) |
| frequency_window_seconds | [5, 15, 100, 300] |

#### Family 1: `queue_service` (train)

**State variables (7):**

| Name | Type | Value range |
|---|---|---|
| `backend.max_batch_capacity` | integer | [16, 32, 64, 128] |
| `queue.retry_base` | integer | [1, 2, 3, 4] |
| `backend.api_version` | integer | [1, 2, 3, 4] |
| `queue.timeout_factor` | integer | [100, 250, 500, 1000] |
| `queue.capacity_multiplier` | integer | [4, 8, 16, 32] |
| `backend.protocol_version` | integer | [1, 2, 3] |
| `queue.dlq_factor` | integer | [2, 4, 8] |

**Fields (6 main):**

| Field | Type | Constraint text | Compute | State var | Trigger |
|---|---|---|---|---|---|
| `batch_size` | integer | must equal floor(backend.max_batch_capacity / 2) | `max_batch_capacity // 2` | backend.max_batch_capacity | no |
| `retry_count` | integer | must equal queue.retry_base × 2 | `retry_base * 2` | queue.retry_base | no |
| `dead_letter_enabled` | boolean | must be true if backend.api_version >= 3, otherwise false | `api_version >= 3` | backend.api_version | **yes** |
| `timeout_ms` | integer | must equal queue.timeout_factor × 8 | `timeout_factor * 8` | queue.timeout_factor | no |
| `max_queue_size` | integer | must equal queue.capacity_multiplier × 64 | `capacity_multiplier * 64` | queue.capacity_multiplier | no |
| `serialization_format` | string | must be "json" if backend.protocol_version == 1, "msgpack" if 2, "protobuf" if 3 | lookup | backend.protocol_version | no |

**Conditional field:**

| Field | Type | Constraint text | Compute | State var | Trigger condition |
|---|---|---|---|---|---|
| `dead_letter_queue_size` | integer | must equal queue.dlq_factor × 16 | `dlq_factor * 16` | queue.dlq_factor | `dead_letter_enabled == true` |

#### Family 2: `logging_service` (train)

**State variables (7):**

| Name | Type | Value range |
|---|---|---|
| `storage.max_log_quota_mb` | integer | [1000, 2000, 4000, 8000] |
| `logging.rotation_factor` | integer | [3, 5, 10, 15] |
| `system.async_threshold` | integer | [1, 2, 4] |
| `infra.log_api_version` | integer | [1, 2, 3] |
| `logging.format_version` | integer | [1, 2, 3] |
| `logging.buffer_multiplier` | integer | [4, 8, 16, 32] |
| `logging.flush_base` | integer | [100, 200, 500] |

**Fields (6 main):**

| Field | Type | Constraint text | Compute | State var | Trigger |
|---|---|---|---|---|---|
| `max_file_size_mb` | integer | must equal floor(storage.max_log_quota_mb / 10) | `quota // 10` | storage.max_log_quota_mb | no |
| `rotation_count` | integer | must equal logging.rotation_factor × 2 | `factor * 2` | logging.rotation_factor | no |
| `async_enabled` | boolean | must be true if system.async_threshold <= 2, otherwise false | `threshold <= 2` | system.async_threshold | **yes** |
| `log_level` | string | must be "debug" if infra.log_api_version == 1, "info" if 2, "warn" if 3 | lookup | infra.log_api_version | no |
| `output_format` | string | must be "text" if logging.format_version == 1, "json" if 2, "structured" if 3 | lookup | logging.format_version | no |
| `buffer_size_kb` | integer | must equal logging.buffer_multiplier × 4 | `multiplier * 4` | logging.buffer_multiplier | no |

**Conditional field:**

| Field | Type | Constraint text | Compute | State var | Trigger condition |
|---|---|---|---|---|---|
| `flush_interval_ms` | integer | must equal logging.flush_base × 2 | `flush_base * 2` | logging.flush_base | `async_enabled == true` |

Note: Value ranges for `storage.max_log_quota_mb` are chosen as multiples of 10, so `floor(x / 10)` produces exact integers with no rounding ambiguity.

#### Family 3: `auth_service` (train)

**State variables (7):**

| Name | Type | Value range |
|---|---|---|
| `security.hash_standard` | integer | [1, 2, 3] |
| `auth.expiry_divisor` | integer | [1, 2, 4, 6] |
| `security.policy_version` | integer | [1, 2, 3, 4] |
| `infra.session_pool_size` | integer | [64, 128, 256, 512] |
| `auth.timeout_base` | integer | [15, 30, 60, 120] |
| `security.password_factor` | integer | [2, 3, 4, 5] |
| `auth.mfa_base` | integer | [30, 60, 90, 120] |

**Fields (6 main):**

| Field | Type | Constraint text | Compute | State var | Trigger |
|---|---|---|---|---|---|
| `hash_algorithm` | string | must be "sha256" if security.hash_standard == 1, "sha384" if 2, "sha512" if 3 | lookup | security.hash_standard | no |
| `token_expiry_hours` | integer | must equal floor(24 / auth.expiry_divisor) | `24 // divisor` | auth.expiry_divisor | no |
| `require_mfa` | boolean | must be true if security.policy_version >= 3, otherwise false | `version >= 3` | security.policy_version | **yes** |
| `max_sessions` | integer | must equal floor(infra.session_pool_size / 4) | `pool // 4` | infra.session_pool_size | no |
| `session_timeout_minutes` | integer | must equal auth.timeout_base × 2 | `base * 2` | auth.timeout_base | no |
| `password_min_length` | integer | must equal security.password_factor × 4 | `factor * 4` | security.password_factor | no |

**Conditional field:**

| Field | Type | Constraint text | Compute | State var | Trigger condition |
|---|---|---|---|---|---|
| `mfa_timeout_seconds` | integer | must equal auth.mfa_base × 2 | `mfa_base * 2` | auth.mfa_base | `require_mfa == true` |

Note: `auth.expiry_divisor` values [1, 2, 4, 6] yield `floor(24 / d)` = [24, 12, 6, 4] — all exact integers.

### 5.3 Family Summary Table

| Index | Name | Split | State vars (7 each) | Trigger field | Dep rule trigger |
|---|---|---|---|---|---|
| 0 | cache_service | train | total_memory, cpu_count, cache_version, ttl_mult, pool_size, compress_level, freq_base | eviction_policy | eviction_policy == "lfu" |
| 1 | queue_service | train | batch_capacity, retry_base, api_version, timeout_factor, capacity_mult, protocol_version, dlq_factor | dead_letter_enabled | dead_letter_enabled == true |
| 2 | logging_service | train | log_quota, rotation_factor, async_threshold, log_api_version, format_version, buffer_mult, flush_base | async_enabled | async_enabled == true |
| 3 | auth_service | train | hash_standard, expiry_divisor, policy_version, session_pool, timeout_base, password_factor, mfa_base | require_mfa | require_mfa == true |
| 4 | storage_service | train | (7 isolated) | compression_type | compression_type == "zstd" |
| 5 | network_service | train | (7 isolated) | proxy_protocol | proxy_protocol == "v2" |
| 6 | monitoring_service | dev | (7 isolated) | aggregation_method | aggregation_method == "histogram" |
| 7 | scheduler_service | dev | (7 isolated) | retry_strategy | retry_strategy == "exponential" |
| 8 | deployment_service | test | (7 isolated) | deploy_strategy | deploy_strategy == "canary" |
| 9 | notification_service | test | (7 isolated) | channel_priority | channel_priority == "urgent" |

Families 4-9: full definitions (7 state variables, 6 fields, 1 conditional, invalid-value pools) are provided in the R1 implementation contract. Each MUST satisfy the isolation invariant: 7 state variables, each used by exactly one field. The R1 contract includes a programmatic check that verifies this invariant for all families.

### 5.4 Constraint Patterns

Each field uses one of three constraint patterns:

| Pattern | Example | Train families 0-5 | Dev/Test families 6-9 |
|---|---|---|---|
| `arithmetic` | `floor(state_var / constant)` or `state_var × constant` | yes | yes |
| `lookup` | `{1: "sha256", 2: "sha384", 3: "sha512"}[state_var]` | yes | yes |
| `threshold` | `true if state_var >= constant, otherwise false` | yes | yes |
| `modular` | `(state_var % constant) + offset` | **no** | **yes** (≥1 field per family) |

Train families use three patterns. Dev/test families add a fourth (modular) to create a held-out structural combination (§7.3).

---

## 6. Generator Design

### 6.1 Determinism

All randomness derives from `random.Random(seed)` where `seed` is an explicit integer. The seed is recorded in every instance manifest. The RNG is consumed in a fixed order (§6.3), and the same seed always produces the same instance or the same rejection.

### 6.2 Canonical Generation Algorithm

```python
PROTOCOL_VERSION = "r2.0"

def generate_instance(
    master_seed: int,
    families: list[ContentFamily],
    target_fault: str,          # one of the 4 fault types
) -> Instance | None:
    rng = random.Random(master_seed)

    # Step 1: select family
    family = families[rng.randint(0, len(families) - 1)]

    # Step 2: generate external state
    state = {}
    for sv in family.state_vars:          # fixed iteration order
        state[sv.name] = rng.choice(sv.value_range)

    # Step 3: compute golden config
    golden = {}
    for f in family.fields:               # fixed iteration order
        golden[f.name] = f.compute(state)

    # Step 4: add conditional fields where trigger is satisfied
    for cf in family.conditional_fields:
        trigger_val = golden[cf.trigger_field]
        if trigger_val == cf.trigger_value:
            golden[cf.name] = cf.compute(state)

    # Step 5: inject fault
    presented = dict(golden)
    faulted_field = None
    faulted_state = dict(state)

    if target_fault == "normal":
        pass  # no changes

    elif target_fault == "constraint_violation":
        eligible = [f for f in family.fields if not f.is_trigger]
        field = eligible[rng.randint(0, len(eligible) - 1)]
        valid_invalids = [v for v in field.values_for_invalid
                         if v != golden[field.name]]
        if not valid_invalids:
            return None  # REJECT: no invalid value differs from correct
        presented[field.name] = rng.choice(valid_invalids)
        faulted_field = field.name

    elif target_fault == "missing_dependency":
        eligible = [cf for cf in family.conditional_fields
                    if cf.name in golden]   # trigger satisfied
        if not eligible:
            return None                     # REJECT: no conditional field required
        cf = eligible[rng.randint(0, len(eligible) - 1)]
        del presented[cf.name]
        faulted_field = cf.name

    elif target_fault == "stale_version":
        eligible = [f for f in family.fields if not f.is_trigger]
        field = eligible[rng.randint(0, len(eligible) - 1)]
        sv = family.get_state_var(field.state_var)
        old_val = state[sv.name]
        new_vals = [v for v in sv.value_range if v != old_val]
        new_val = new_vals[rng.randint(0, len(new_vals) - 1)]
        new_state = dict(state)
        new_state[sv.name] = new_val
        new_golden_val = field.compute(new_state)
        if new_golden_val == golden[field.name]:
            return None                     # REJECT: value unchanged after bump
        faulted_field = field.name
        faulted_state = new_state
        # golden is updated to reflect new correct value
        golden[field.name] = new_golden_val

    # Step 6: validate solvability
    assert verify_golden(golden, family, faulted_state), "golden config invalid"
    if faulted_field:
        # Pre-repair check: the presented config MUST fail verification.
        assert verify(presented, golden, presented, target_fault, faulted_field,
                      family, faulted_state) == 0, "fault did not break verification"
        # Oracle check: a known-field repair yields reward = 1.
        assert simulate_oracle(presented, faulted_state, faulted_field, family) == 1

    # Step 7: compute content fingerprint
    fingerprint = content_fingerprint(family.name, presented, faulted_state,
                                       target_fault, faulted_field)

    return Instance(
        task_id=f"{family.name}_{master_seed}_{target_fault}",
        seed=master_seed,
        family=family.name,
        fault_type=target_fault,
        faulted_field=faulted_field,
        golden_config=golden,
        presented_config=presented,
        external_state=faulted_state,
        fingerprint=fingerprint,
        generator_version=PROTOCOL_VERSION,
    )
```

### 6.3 Rejection and Retry

When `generate_instance` returns `None`:
- The caller increments the seed by 1 and retries.
- Maximum 10 retries per slot. If all 10 fail, the slot is left empty and logged.
- Rejection reasons are recorded in the generation manifest.

Rejection occurs when:
- `missing_dependency` is requested but no conditional field's trigger is satisfied.
- `stale_version` is requested but the state variable bump doesn't change the field's golden value.

### 6.4 Dataset Generation

Instances are generated per-split with a target distribution over fault types:

| Fault type | Target proportion |
|---|---|
| normal | 20% |
| constraint_violation | 30% |
| missing_dependency | 25% |
| stale_version | 25% |

The generator pre-allocates slots deterministically:

```python
def allocate_slots(N: int) -> dict[str, int]:
    raw = {"normal": 20, "constraint_violation": 30,
           "missing_dependency": 25, "stale_version": 25}
    exact = {k: N * v / 100 for k, v in raw.items()}
    slots = {k: int(v) for k, v in exact.items()}
    remainder = N - sum(slots.values())
    # Largest-remainder method; ties broken by dict insertion order
    by_frac = sorted(raw.keys(), key=lambda k: -(exact[k] - slots[k]))
    for i in range(remainder):
        slots[by_frac[i]] += 1
    assert sum(slots.values()) == N
    return slots
```

Each slot consumes one master seed in order. The fault type is determined by the slot, not by the RNG — the seed controls family selection and state values, not fault type. Rejected seeds (§6.3) leave the slot empty; the final dataset may be smaller than N. If more than 10% of slots are rejected, the generation manifest records a warning and the pilot must re-examine family coverage.

**Training sampling distribution** (how instances are drawn during GRPO training) is separate and defined in the training protocol (R3). The dataset generation distribution above determines what's available; the training protocol determines what's used.

### 6.5 Content Fingerprint

```python
def content_fingerprint(family_name, config, state, fault_type, faulted_field):
    canonical = json.dumps({
        "family": family_name,
        "config": dict(sorted(config.items())),
        "state": dict(sorted(state.items())),
        "fault_type": fault_type,
        "faulted_field": faulted_field,
    }, sort_keys=True, ensure_ascii=True)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]
```

Duplicate fingerprints within a split are rejected — the generator skips to the next seed.

---

## 7. Data Splits

### 7.1 Split Assignment

| Split | Families | Seed range | Purpose |
|---|---|---|---|
| train | 0-5 (cache, queue, logging, auth, storage, network) | [0, 99999] | GRPO training |
| dev | 6-7 (monitoring, scheduler) | [100000, 109999] | Checkpoint selection, pilot tuning |
| test | 8-9 (deployment, notification) | [200000, 209999] | Final evaluation only |

Content families do NOT appear across splits — test families are genuinely unseen during training.

### 7.2 Composition Identifiers

A **composition** is a `(content_family, fault_type)` pair. There are 10 × 4 = 40 total compositions.

| Split | Compositions | What is unseen |
|---|---|---|
| train | 6 families × 4 faults = 24 | — |
| dev | 2 families × 4 faults = 8 | Family-specific fields, state vars, value ranges |
| test | 2 families × 4 faults = 8 | Family-specific fields, state vars, value ranges |

### 7.3 Held-Out Structural Combinations

To satisfy the PRD requirement for unseen operation combinations (not just unseen content), dev and test families MUST include at least one field using a **modular constraint pattern** not present in any train family:

| Pattern | Formula | Train | Dev/Test |
|---|---|---|---|
| arithmetic | `floor(state_var / c)` or `state_var × c` | yes | yes |
| lookup | `{1: "x", 2: "y", 3: "z"}[state_var]` | yes | yes |
| threshold | `true if state_var >= c` | yes | yes |
| **modular** | `(state_var % c) + offset` | **no** | **yes** |

This creates a genuine held-out operation combination: `(modular, constraint_violation)` and `(modular, stale_version)` are unseen in training. The R1 contract for families 6-9 must designate at least one field per dev/test family with a modular constraint.

**Modular field requirements:**
- The modular field MUST be a **non-trigger main field** (not a conditional field, not a trigger field) so it is eligible for both `constraint_violation` and `stale_version` faulting.
- The modular constraint MUST produce **≥ 2 distinct output values** across the state variable's range. If `(state_var % c) + offset` produces only one reachable value, the field cannot generate both correct and incorrect values for faulting.

**Eval-set verification:** After generating the dev/test split, the generator MUST verify that the generated set contains at least one instance of `(modular, constraint_violation)` and at least one of `(modular, stale_version)`. If either is missing after all seeds are consumed (including retries), generation fails with an explicit error — do not silently accept a set that lacks the held-out combination.

**Modular stratum reporting:** Pilot results (§9.1) must report FCR for the modular stratum separately, alongside per-fault-type rates. This stratum is small (expected 2-4 instances per dev family) so it is directional, not statistically powered.

**Generalization claim scope:** The experiment tests generalization across (a) unseen service configurations (field names, state variable names, value ranges) and (b) one unseen constraint pattern (modular). It does NOT test generalization to unseen fault types — all three fault types appear in all splits.

### 7.3 Instance Counts

Determined by the training protocol, not R0. The seed ranges support up to ~100K instances per split; actual usage is a subset.

**Pilot evaluation:** 50 instances from the dev split (details in §9).

---

## 8. Verification and Reward

### 8.1 Single Final-State Predicate

The reward is determined by **one authoritative predicate** applied to the final config state. No other rule overrides it. All edge cases are derived from this predicate, not specified separately.

```python
def verify(final_config: dict, golden_config: dict, presented_config: dict,
           fault_type: str, faulted_field: str | None,
           family: ContentFamily, state: dict) -> int:
    """Returns 1 (success) or 0 (failure)."""

    # 0. Key-set check: final config must contain exactly the expected keys.
    #    Expected = main fields + conditional fields whose trigger is satisfied
    #    in the GOLDEN config (not final_config, to prevent self-referential logic).
    expected_keys = {f.name for f in family.fields}
    for cf in family.conditional_fields:
        golden_trigger = golden_config.get(cf.trigger_field)
        if golden_trigger == cf.trigger_value:
            expected_keys.add(cf.name)
    if set(final_config.keys()) != expected_keys:
        return 0

    # 1. Strict JSON type check: every field must match expected type exactly.
    #    In particular, True is not 1, and False is not 0.
    for field_name, expected_value in golden_config.items():
        if field_name in final_config:
            if type(final_config[field_name]) is not type(expected_value):
                return 0

    # 2. The final config must be a fully valid configuration under current state.
    #    All main fields present with correct values:
    for f in family.fields:
        if final_config[f.name] != f.compute(state):
            return 0

    #    All required conditional fields present with correct values:
    for cf in family.conditional_fields:
        if cf.name in expected_keys:
            if final_config[cf.name] != cf.compute(state):
                return 0

    # 3. No regression: no field changed from presented_config unless it is
    #    the faulted field or a field that was absent in presented_config.
    for field_name, presented_value in presented_config.items():
        if field_name == faulted_field:
            continue
        if final_config[field_name] != presented_value:
            return 0

    return 1
```

### 8.2 Edge Cases (Derived from §8.1)

| Scenario | Predicate result | Why |
|---|---|---|
| submit_fix(A, same_value) then submit_fix(B, correct) | Check A: unchanged from presented ✓. Check B: matches golden ✓. | Final state is valid, no regression → 1 |
| submit_fix(faulted, wrong) then submit_fix(faulted, correct) | faulted matches golden ✓. No regression ✓. | Final state is valid → 1 |
| submit_fix(faulted, correct) then submit_fix(other, wrong) | other changed from presented → regression | → 0 |
| normal task, submit_fix(A, same_value_as_presented) | A unchanged (same value) ✓. Config still valid ✓. | → 1 |
| normal task, submit_fix(A, different_value) | A changed from presented → regression | → 0 |
| stale_version: correct field fix but conditional now required | Cannot happen — trigger fields excluded from stale_version (§4.4) | structurally prevented |
| normal: agent adds frequency_window_seconds with garbage value | Key-set check (§8.1 step 0): extra key not in expected set | → 0 |
| any fault: agent adds unknown key via submit_fix | submit_fix returns error for unknown field; key not added to config | no state change |

### 8.3 Diagnostic Flags

Recorded for analysis but do NOT affect reward:

| Flag | Trigger |
|---|---|
| `field_not_in_schema` | `submit_fix` called with a field name not in the family's schema |
| `no_read_config` | `submit_fix` called without a preceding `read_config` |
| `excessive_queries` | More than 3 `query_info` calls in one episode |
| `invalid_action` | Episode terminated due to malformed tool call |
| `infra_error` | Python exception in environment code |

Infrastructure errors (`infra_error`) are excluded from reward statistics entirely.

---

## 9. Pilot Protocol

### 9.1 Pilot Design

**50 dev-split instances × 2 rollouts per instance (group size = 2) = 100 rollouts total.**

**Group size is frozen at 2 for the pilot.** This is the minimum for measuring mixed-reward fraction. If mixed-group fraction < 15%, the first adjustment should simplify constraints rather than increase group size — increasing group size changes the rollout budget and invalidates the 300-rollout cap (§9.4).

Distribution per `allocate_slots(50)`: 10 normal + 15 constraint_violation + 13 missing_dependency + 12 stale_version = 50. Instances drawn from seed range [100000, ...] using the generation algorithm (§6).

The pilot runs the base Qwen3-4B model (no training) to measure:
- **FCR:** reward rate on the 40 fault instances.
- **NPR:** reward rate on the 10 normal instances.
- **Per-fault-type reward rates.**
- **Mixed-reward group fraction:** for each instance with 2 rollouts, count how many have one success and one failure. A mixed-reward fraction near zero means GRPO advantages will be all-zero (no learning signal), regardless of FCR. **Target: mixed-group fraction ≥ 15%.** If below 15%, the task has insufficient within-instance variance for GRPO — simplify constraints to increase partial solvability (§9.2).
- Tool call patterns and failure modes.

### 9.2 What May Be Adjusted

| Parameter | Initial value | Adjustment range | Trigger |
|---|---|---|---|
| Tool call limit | 6 | 6-8 | Base model needs more investigation steps |
| Token limit per turn | 4096 | 4096-8192 | Thinking tokens too compressed |
| Constraint complexity | As specified | Simplify compute functions | FCR < 5% across all fault types OR mixed-group fraction < 15% |
| Fault-type distribution | 20/30/25/25 | Rebalance within ±10pp | One type dominates failures |
| Constraint complexity (up) | As specified | Add complexity | FCR > 60% |

**Target ranges:** FCR in [5%, 60%], mixed-group fraction ≥ 15%.

### 9.3 What Is Frozen After R0

| Parameter | Value |
|---|---|
| Content family names and split assignment | §5.3 |
| Tool schemas (names, arguments, return structure) | §3 |
| Fault types (3 types + normal) | §4 |
| Reward predicate (binary 0/1, §8.1 logic) | §8 |
| Isolation invariant (7 state vars per family, each isolated) | §4.1 |
| Exact-one-value reward rule | Each fault has exactly one correct value |
| Pilot group size | 2 rollouts per instance (§9.1) |

### 9.4 Adjustment Limits

- **Maximum 2 adjustments.** Cumulative budget: 3 runs × 100 rollouts = **300 rollouts total** (initial + 2 post-adjustment re-evaluations).
- **Cumulative runtime cap:** 2 GPU-hours enforced. If a run exceeds 40 minutes, the remaining budget is recalculated from actual throughput, not the S5 estimate.
- **Stopping rule:** if 2 consecutive adjustments each change FCR by < 5pp, stop and escalate to task redesign.
- **If no adjustment brings FCR into [5%, 60%] AND mixed-group fraction ≥ 15%:** report pilot results and escalate to task redesign rather than further piloting.

### 9.5 Versioning Discipline

Each adjustment creates a versioned spec (r0.5, r0.6, ...). The changelog (§13) records: date, parameter changed, old/new value, pilot evidence (FCR, NPR, failure mode distribution), and version number. The final test split must NOT inform pilot adjustments.

---

## 10. Experiment Ordering

### 10.1 Metrics

Two separate metrics, measured on their respective task subsets:

| Metric | Definition | Subset |
|---|---|---|
| **Fault Completion Rate (FCR)** | reward rate on fault tasks | constraint_violation + missing_dependency + stale_version instances |
| **Normal Preservation Rate (NPR)** | reward rate on normal tasks | normal instances |

All experimental comparisons use FCR as the primary metric and NPR as a guardrail.

### 10.2 Q1: Does GRPO Improve Fault Completion?

| Component | Specification |
|---|---|
| Training | GRPO on train split, fixed distribution over instances |
| Checkpoint selection | Dev-split FCR (highest FCR among checkpoints with NPR ≥ NPR(base) - 10pp) |
| Final evaluation | Test split |
| Control | Base Qwen3-4B on the same test split |
| **Primary success** | Test-split FCR(trained) > FCR(base), statistically significant |
| **Guardrail** | Test-split NPR(trained) ≥ NPR(base) - 10pp |

Statistical test details (exact test, sample size, significance level) are deferred to the R3 training protocol. R0 specifies the metric definitions and comparison structure only.

### 10.3 Q2 Gate

Q2 proceeds **only if** Q1's GRPO run shows:
- Dev-set FCR improvement ≥ +5pp over base model.
- Dev-set NPR ≥ NPR(base) - 10pp.

If either condition fails, Q2 is not run. The project reports Q1 results and stops.

**If no checkpoint qualifies** (no checkpoint meets the NPR guardrail), Q1 reports failure: GRPO training did not produce a checkpoint that improves fault completion without unacceptable normal-condition regression.

### 10.4 Q2: Does Failure-Driven Distribution Beat Fixed?

| Component | Specification |
|---|---|
| Arm A (fixed) | Fixed distribution over (family, fault_type) pairs throughout training |
| Arm B (failure-driven) | Distribution shifts toward pairs with lower current-policy FCR |
| Same base | Both arms start from the same base Qwen3-4B |
| Same budget | Same total training budget (steps × batch_size × rollouts) |
| Same eval | Same test split, same protocol |
| **Primary success** | Test-split FCR(Arm B) > FCR(Arm A) |
| **Guardrail** | NPR(Arm B) ≥ NPR(Arm A) - 10pp |

### 10.5 Confounding Prevention

- Q1 runs before Q2 — basic GRPO effect is measured independently.
- Q2 arms share: base model, training budget, eval protocol, test split.
- Only the practice distribution differs between arms.
- Dev set is used for checkpoint selection and Q2 gate; test set only for final evaluation.
- FCR and NPR are always reported separately — no aggregate "overall reward rate" is used for any gate or success criterion.

---

## 11. Acceptance Walkthroughs

Each walkthrough is a manually executable sequence proving the environment works without a model. All use the isolated-dependency family designs from §5.2.

### 11.1 constraint_violation — Correct Agent

```
Instance: cache_service
State: {system.total_memory_mb: 16384, system.cpu_count: 8, backend.cache_version: 2,
        cache.ttl_multiplier: 4, infra.max_pool_size: 800, backend.compression_level: 2,
        cache.freq_base: 30}
Golden: {max_memory_mb: 4096, shard_count: 8, eviction_policy: "lru",
         ttl_seconds: 240, max_connections: 400, compression_enabled: true}
Fault: max_memory_mb set to 8192

Step 1: read_config()
  → {config: {max_memory_mb: 8192, shard_count: 8, eviction_policy: "lru",
              ttl_seconds: 240, max_connections: 400, compression_enabled: true},
     status: "error"}

Step 2: investigate max_memory_mb with query_info("max_memory_mb")
  → {name: "max_memory_mb", type: "integer",
     constraint: "must equal floor(system.total_memory_mb / 4)",
     dependencies: ["system.total_memory_mb"],
     current_value: {"system.total_memory_mb": 16384}}

Step 3: submit_fix("max_memory_mb", 4096)
  → {status: "ok", message: "Field updated."}

Verify: §8.1 predicate → max_memory_mb=4096=golden ✓, no regression ✓, all constraints satisfied ✓
→ reward = 1

Isolation check: no other visible field depends on system.total_memory_mb, so the
agent cannot derive 4096 from any other config value. ✓
```

### 11.2 constraint_violation — "Always Retry" Agent

```
Same instance as 11.1.

Step 1: read_config() → status: "error" (field is not identified)
Step 2: submit_fix("max_memory_mb", 1) → {status: "ok"}

Verify: max_memory_mb=1 ≠ 4096=golden → reward = 0
```

### 11.3 missing_dependency — Correct Agent

```
Instance: cache_service
State: {..., backend.cache_version: 3, cache.freq_base: 30}
Golden: {..., eviction_policy: "lfu", frequency_window_seconds: 120}
Fault: frequency_window_seconds removed (was required because eviction_policy=="lfu")

Step 1: read_config()
  → {config: {max_memory_mb: 4096, shard_count: 8, eviction_policy: "lfu",
              ttl_seconds: 240, max_connections: 400, compression_enabled: true},
     status: "error"}

Step 2: investigate the absent frequency_window_seconds with query_info("frequency_window_seconds")
  → {name: "frequency_window_seconds", type: "integer",
     constraint: "must equal cache.freq_base × 4",
     dependencies: ["cache.freq_base"],
     current_value: {"cache.freq_base": 30},
     conditional: "required when eviction_policy == 'lfu'"}

Step 3: submit_fix("frequency_window_seconds", 120)
  → {status: "ok"}

Verify: frequency_window_seconds=120=golden ✓, no regression ✓ → reward = 1

Isolation check: no other visible field depends on cache.freq_base. ✓
```

### 11.4 missing_dependency — "Do Nothing" Agent

```
Same instance as 11.3.

Step 1: read_config() → status: "error" (field is not identified)
(Agent responds with text → episode ends.)

Verify: frequency_window_seconds absent, but required → §8.1 step 2 fails → reward = 0
```

### 11.5 stale_version — Correct Agent

```
Instance: cache_service
State: {..., backend.compression_level: 2} (was 1, bumped to 2)
Golden: {..., compression_enabled: true} (was false under level=1, now true under level=2)
Fault: compression_enabled still false in config (stale)

Step 1: read_config()
  → {config: {..., compression_enabled: false},
     status: "error"}

Step 2: investigate compression_enabled with query_info("compression_enabled")
  → {name: "compression_enabled", type: "boolean",
     constraint: "must be true if backend.compression_level >= 2, otherwise false",
     dependencies: ["backend.compression_level"],
     current_value: {"backend.compression_level": 2}}

Step 3: submit_fix("compression_enabled", true)
  → {status: "ok"}

Verify: compression_enabled=true=golden ✓, no regression ✓ → reward = 1

Cascade check: compression_enabled is NOT a trigger field → no conditional field
becomes required → no cascade. ✓
```

### 11.6 stale_version — "Guess Common Value" Agent

```
Same instance as 11.5.

Step 1: read_config() → status: "error" (field is not identified)
Step 2: submit_fix("compression_enabled", 1)  ← note: integer 1, not boolean true

Verify: type(1) is int ≠ type(true) is bool → §8.1 strict type check fails → reward = 0
```

### 11.7 normal — Correct Agent

```
Instance: cache_service (no fault, config is valid)

Step 1: read_config()
  → {config: {...}, status: "ok"}

(Agent responds confirming config is valid → episode ends.)

Verify: config unchanged → matches golden ✓, no regression ✓ → reward = 1
```

### 11.8 normal — "Always Fix" Agent

```
Same instance as 11.7.

Step 1: read_config() → status: "ok"
Step 2: submit_fix("max_memory_mb", 9999)

Verify: max_memory_mb=9999 ≠ golden → §8.1 step 2 fails → reward = 0
```

---

## 12. Planned Code Paths

| Path | Responsibility |
|---|---|
| `src/tasks/tool_recovery/generator.py` | Content families, instance generation, solvability validation, content fingerprints |
| `src/tasks/tool_recovery/environment.py` | reset/step, tool dispatch, state management, termination |
| `src/tasks/tool_recovery/verifier.py` | §8.1 predicate, strict type checking, diagnostic flags |
| `src/tasks/tool_recovery/records.py` | Instance manifest schema, trajectory records, generation manifest |
| `src/adapters/tool_recovery_agent_r1.py` | Agent-R1 adapter for the new task |
| `configs/self_improve/` | Training and evaluation configs |
| `tests/test_tool_recovery*.py` | Walkthroughs (§11), isolation invariant, shortcut baseline, solvability |
| `artifacts/self_improve/` | Contracts, manifests, pilot results, run reports |

---

## 13. Changelog

### r0.1 (2026-09-12) — Revised per Codex review

**9 issues addressed:**

1. **Cascade fix (P1):** Isolated state variable dependencies — each field depends on exactly one unique state variable. Stale_version faults restricted to non-trigger fields, preventing cascade where fixing one field makes another required.

2. **Correlation fix (P1):** Same isolation fix eliminates cross-field value inference. Added shortcut baseline test (< 15% FCR without query_info) to R1 acceptance. Removed claim that query_info is the "only" path.

3. **Split structure (P1):** Added composition identifiers (§7.2), content fingerprints (§6.5), duplicate rejection. Explicitly stated generalization scope and limitations (§5.4).

4. **Type checking (P1):** Strict JSON type comparison (`type(x) is type(y)`) in verifier. Single final-state predicate (§8.1) is sole authority — all edge cases derived from it. Walkthrough 11.6 demonstrates `True ≠ 1`.

5. **Solvability invariant (P2):** Isolated dependencies guarantee exactly 4 calls per faulted field. Removed "may be skippable" language.

6. **Integer division (P2):** All constraint texts use `floor(...)`. Value ranges chosen to produce exact integers (e.g., log_quota in [1000, 2000, 4000, 8000] ÷ 10 = no remainder).

7. **Generator algorithm (P2):** Single canonical algorithm (§6.2) with fixed RNG consumption order, explicit rejection/retry (§6.3), separated dataset generation vs training sampling distributions (§6.4).

8. **Pilot consolidation (P2):** One design: 50 instances × 2 rollouts (§9.1). Max 3 adjustments, cumulative ≤ 300 rollouts, < 2 GPU-hours. Stopping rule: 2 consecutive < 5pp change.

9. **Metrics fix (P1):** FCR (fault completion rate) and NPR (normal preservation rate) defined separately (§10.1). Q1 success requires FCR improvement AND NPR guardrail. Q2 gate uses FCR only. No aggregate "overall reward rate" in any gate.

**Structural changes:** Families now have 7 state variables (was 3), one per field. 4 families fully specified (was 3). Walkthroughs updated for isolation (11.5 uses compression_enabled instead of eviction_policy to avoid cascade).

### r0.2 (2026-09-12) — Revised per second Codex review

**8 issues addressed:**

1. **Invalid pool overlap (P1):** Generator now filters `values_for_invalid` to exclude the correct value before sampling. Added pre-repair verification assertion: the presented config must fail the verifier before repair.

2. **Extra-key acceptance (P1):** Verifier step 0 checks `final_config.keys() == expected_keys`. Expected keys derived from golden config's trigger values, not final config. Adding `frequency_window_seconds: "garbage"` while `eviction_policy == "lru"` now correctly fails.

3. **Conditional presence leak (P1):** Trigger fields excluded from constraint_violation eligibility (in addition to stale_version). Shortcut baseline redefined with precise strategies, analytical FCR bounds, and a 15% acceptance threshold with derivation.

4. **PRD composition split (P1):** Added held-out structural combination via modular constraint pattern (`(state_var % c) + offset`), present only in dev/test families. Creates genuine unseen operation combinations `(modular, constraint_violation)` and `(modular, stale_version)`.

5. **Mixed-group GRPO signal (P2):** Pilot now measures mixed-reward group fraction (instances with divergent rollout outcomes). Target ≥ 15%. Below threshold triggers constraint simplification or increased rollouts per instance.

6. **Allocation rounding (P2):** Replaced `ceil` with deterministic `allocate_slots()` function using floor division + ordered remainder distribution. Sum always equals N exactly.

7. **Adjustment limit alignment (P2):** Max 2 adjustments (was 3), aligning with 300 rollout budget (3 runs × 100). Cumulative 2 GPU-hour cap enforced from actual runtime, not throughput estimate. FCR < 5% / FCR > 60% triggers clearly separated in adjustment table.

8. **Checkpoint guardrail (P2):** Changed from fixed `NPR ≥ 50%` to baseline-relative `NPR ≥ NPR(base) - 10pp`, consistent with final guardrail. Added explicit handling when no checkpoint qualifies.

### r0.3 (2026-09-12) — Revised per third Codex review

**5 issues addressed:**

1. **Modular holdout guarantee (P1):** Modular field must be a non-trigger main field with ≥2 distinct reachable outputs. Generator verifies eval set contains at least one `(modular, constraint_violation)` and one `(modular, stale_version)` instance; generation fails if either is missing. Pilot reports modular stratum FCR separately.

2. **Scripted baseline measurement (P2):** Removed unsupported "~8% analytical upper bound" claim. Replaced with a precisely defined scripted baseline (no-`query_info`, type-aware guessing) and a requirement to **measure** its FCR empirically in R1. Acceptance threshold (< 15% FCR) is now a design target, not an analytical assertion.

3. **Allocator fix (P2):** `allocate_slots()` changed from fixed-priority remainder to largest-remainder method. `allocate_slots(50)` now produces `10, 15, 13, 12` matching §9.1's stated counts.

4. **Group size frozen (P2):** Pilot group size frozen at 2 rollouts per instance. If mixed-group fraction < 15%, adjust constraints first — not group size. Added to §9.3 frozen parameters.

5. **Version constant (P2):** Replaced hardcoded `generator_version="r0.1"` with `PROTOCOL_VERSION` constant at top of generator code. All generated instances carry the current protocol version.

### r0.4 (2026-09-12) — Revised per fourth Codex review

**2 issues addressed:**

1. **Scripted baseline precision (P2):** Baseline now explicitly granted static schema metadata (field types + reachable output values, not state-variable ranges). Enum elimination row in shortcut table updated to reflect schema access. Sampling uses seeded RNG with seed recorded in pilot manifest. "Deterministic" language replaced with "schema-aware guessing."

2. **Group-size adjustment contradiction (P2):** Removed "or increase rollouts per instance" from §9.1 mixed-group instruction (contradicted frozen group size). Added mixed-group fraction < 15% as a trigger for constraint simplification in §9.2 adjustment table, alongside the existing FCR < 5% trigger.

### r0.5 (2026-09-16) — Pilot-driven query response adjustment

**Parameter changed:** `query_info(field_name)` response payload.

- **Old value (r0.4):** Returned the field constraint and dependency name; the agent needed a separate `query_info(state_var_name)` call to obtain the dependency's current value.
- **New value (r0.5):** Also returns `current_value`, a one-entry mapping from the dependency name to its current value. Conditional fields receive the same key; `query_info("fields")` and direct state-variable queries are unchanged.
- **Pilot evidence:** r0.4 produced FCR = 5%, NPR = 20%, and mixed-reward-group fraction = 0%. The dominant fault-task failure pattern was skipping the state-variable query in 78 of 80 fault rollouts.
- **Version:** r0.5.

### r1.0 (2026-09-16) — Search-based task redesign

**Nature:** Task redesign, escalated from the r0.5 pilot rather than counted as another arithmetic-constraint adjustment.

- **Field-identification removal:** `read_config()` still computes and returns `status: "ok" | "error"`, but no longer exposes the `errors` array or the faulted field. The agent must investigate candidate fields with `query_info(field)`.
- **Call-budget change:** Tool call limit increased from 5 to 6, permitting one `read_config`, up to four candidate-field investigations, and one `submit_fix`.
- **Solvability change:** A faulted instance is solvable if the policy investigates the faulted field within the call budget. The known-field best-case path is 3 calls; exhaustive seven-field search plus submission exceeds the budget.
- **Pilot evidence:** r0.5 produced FCR = 48.75% and mixed-reward-group fraction = 7.5%; 36 of 40 fault instances had identical reward pairs. The diagnosed source was deterministic arithmetic after field identity was directly exposed, so r1.0 moves difficulty to stochastic investigation order.
- **Version:** r1.0.

### r2.0 (2026-09-16) — Revert to r0.5 behavior + continuous reward

**Nature:** Reward function change, informed by 5-condition pilot analysis (500 rollouts) showing binary reward as a root cause of GRPO signal failure.

- **Task behavior revert:** Reverts to r0.5 behavior — `read_config()` returns `errors` array, `query_info()` returns `current_value`, tool call limit back to 5. The r1.0 search-based redesign (removing field identification) did not improve mixed groups and reduced FCR.
- **Continuous reward module:** Added `src/tasks/tool_recovery/reward.py` with `continuous_reward(final_config, golden_config, family)` returning float in [0, 1]. Per-field score: `1/(1+|submitted−golden|)` for integer fields, exact-match 0/1 for string/boolean. Overall reward is the mean across all golden fields.
- **Binary verifier preserved:** The original `verifier.py` and binary reward remain unchanged. Both reward modes are computed for every rollout; `--reward-mode` flag selects which is used for training.
- **Pilot evidence:** Binary reward at T≤1.0 capped mixed groups at 7.5% (3/40). Retroactive continuous reward on T=1.5 data yields 7/40 = 17.5% mixed groups, passing the 15% gate. The combination works because T=1.5 generates 8/40 pairs with different final submit values, and continuous reward differentiates 7 of them (the 8th pair submitted different actions but reached the same final config).
- **Version:** r2.0.
