"""Independent final-state verifier from protocol r0.4 section 8.1."""

from __future__ import annotations

from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from .generator import ContentFamily


DIAGNOSTIC_FLAGS = frozenset(
    {"field_not_in_schema", "no_read_config", "excessive_queries", "invalid_action", "infra_error"}
)


def expected_keys(golden_config: dict[str, Any], family: "ContentFamily") -> set[str]:
    keys = {field.name for field in family.fields}
    for field in family.conditional_fields:
        if golden_config.get(field.trigger_field) == field.trigger_value:
            keys.add(field.name)
    return keys


def verify_golden(
    golden_config: dict[str, Any], family: "ContentFamily", state: dict[str, Any]
) -> bool:
    return bool(
        verify(golden_config, golden_config, golden_config, "normal", None, family, state)
    )


def verify(
    final_config: dict[str, Any],
    golden_config: dict[str, Any],
    presented_config: dict[str, Any],
    fault_type: str,
    faulted_field: str | None,
    family: "ContentFamily",
    state: dict[str, Any],
) -> int:
    """Return one exactly when the r0.4 final-state predicate succeeds."""
    required = expected_keys(golden_config, family)
    if set(final_config) != required:
        return 0

    for name, expected in golden_config.items():
        if name in final_config and type(final_config[name]) is not type(expected):
            return 0

    for field in family.fields:
        if final_config[field.name] != field.compute(state):
            return 0
    for field in family.conditional_fields:
        if field.name in required and final_config[field.name] != field.compute(state):
            return 0

    for name, value in presented_config.items():
        if name == faulted_field:
            continue
        if final_config[name] != value:
            return 0
    return 1
