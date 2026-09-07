"""Typed configuration parser with dot-path access, merging, and validation."""
from __future__ import annotations

import re
from collections.abc import Iterable, Mapping
from copy import deepcopy


class ConfigError(Exception):
    pass


def _is_identifier(key: str) -> bool:
    return bool(key) and re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", key) is not None


_sentinel = object()


def get_path(data: dict, path: str, default=_sentinel):
    if not path:
        raise ConfigError("empty path")
    parts = path.split(".")
    current = data
    for part in parts:
        if not isinstance(current, dict) or part not in current:
            if default is _sentinel:
                raise KeyError(path)
            return default
        current = current[part]
    return current


def set_path(data: dict, path: str, value) -> None:
    parts = path.split(".")
    if not all(_is_identifier(p) for p in parts):
        raise ConfigError(f"invalid path: {path}")
    current = data
    for part in parts[:-1]:
        if part not in current or not isinstance(current[part], dict):
            current[part] = {}
        current = current[part]
    current[parts[-1]] = value


def delete_path(data: dict, path: str) -> object:
    parts = path.split(".")
    current = data
    for part in parts[:-1]:
        if not isinstance(current, dict) or part not in current:
            raise KeyError(path)
        current = current[part]
    if not isinstance(current, dict) or parts[-1] not in current:
        raise KeyError(path)
    return current.pop(parts[-1])


def flatten(data: dict, prefix: str = "") -> dict[str, object]:
    result = {}
    for key, value in data.items():
        full = f"{prefix}.{key}" if prefix else key
        if isinstance(value, dict):
            result.update(flatten(value, full))
        else:
            result[full] = value
    return result


def unflatten(flat: dict[str, object]) -> dict:
    result: dict = {}
    for path, value in sorted(flat.items()):
        set_path(result, path, value)
    return result


def merge(base: dict, override: dict) -> dict:
    result = deepcopy(base)
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = merge(result[key], value)
        else:
            result[key] = deepcopy(value)
    return result


def diff(old: dict, new: dict) -> dict[str, tuple[object, object]]:
    old_flat = flatten(old)
    new_flat = flatten(new)
    changes = {}
    for key in sorted(set(old_flat) | set(new_flat)):
        old_val = old_flat.get(key)
        new_val = new_flat.get(key)
        if old_val != new_val:
            changes[key] = (old_val, new_val)
    return changes


VALIDATORS = {
    "int": lambda v: isinstance(v, int) and not isinstance(v, bool),
    "float": lambda v: isinstance(v, (int, float)) and not isinstance(v, bool),
    "str": lambda v: isinstance(v, str),
    "bool": lambda v: isinstance(v, bool),
    "list": lambda v: isinstance(v, list),
    "dict": lambda v: isinstance(v, dict),
    "positive": lambda v: isinstance(v, (int, float)) and not isinstance(v, bool) and v > 0,
    "nonempty": lambda v: bool(v) if isinstance(v, (str, list, dict)) else True,
}


def validate(data: dict, schema: dict[str, str | list[str]]) -> list[str]:
    errors = []
    for path, rules in sorted(schema.items()):
        if isinstance(rules, str):
            rules = [rules]
        try:
            value = get_path(data, path)
        except KeyError:
            errors.append(f"missing: {path}")
            continue
        for rule in rules:
            if rule == "required":
                continue
            if rule.startswith("oneof:"):
                allowed = rule[6:].split(",")
                if str(value) not in allowed:
                    errors.append(f"{path}: must be one of {allowed}")
            elif rule in VALIDATORS:
                if not VALIDATORS[rule](value):
                    errors.append(f"{path}: failed {rule}")
            else:
                errors.append(f"{path}: unknown rule {rule}")
    return errors


def parse_env(text: str) -> dict[str, str]:
    result = {}
    for number, line in enumerate(text.splitlines(), 1):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            raise ConfigError(f"line {number}: missing '='")
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip()
        if not _is_identifier(key):
            raise ConfigError(f"line {number}: invalid key {key!r}")
        if value.startswith('"') and value.endswith('"'):
            value = value[1:-1]
        elif value.startswith("'") and value.endswith("'"):
            value = value[1:-1]
        result[key] = value
    return result


def to_env(data: dict) -> str:
    flat = flatten(data)
    lines = []
    for key, value in sorted(flat.items()):
        env_key = key.replace(".", "_").upper()
        if isinstance(value, bool):
            lines.append(f"{env_key}={'true' if value else 'false'}")
        elif isinstance(value, str) and (" " in value or "=" in value):
            lines.append(f'{env_key}="{value}"')
        else:
            lines.append(f"{env_key}={value}")
    return "\n".join(lines) + "\n" if lines else ""


def interpolate(data: dict, context: dict | None = None) -> dict:
    context = context or {}
    flat = flatten(data)
    combined = {**flatten(context), **flat}
    result = {}
    for key, value in flat.items():
        if isinstance(value, str):
            result[key] = _resolve(value, combined)
        else:
            result[key] = value
    return unflatten(result)


def _resolve(template: str, values: dict, depth: int = 10) -> str:
    if depth <= 0:
        raise ConfigError("circular interpolation")
    def replacer(match):
        ref = match.group(1)
        if ref not in values:
            return match.group(0)
        resolved = values[ref]
        if isinstance(resolved, str) and "${" in resolved:
            return _resolve(resolved, values, depth - 1)
        return str(resolved)
    return re.sub(r"\$\{([^}]+)}", replacer, template)


def select(data: dict, paths: Iterable[str]) -> dict:
    result: dict = {}
    for path in paths:
        try:
            value = get_path(data, path)
        except KeyError:
            continue
        set_path(result, path, deepcopy(value))
    return result


def coerce(value: str) -> int | float | bool | str:
    if value.lower() in ("true", "yes", "on"):
        return True
    if value.lower() in ("false", "no", "off"):
        return False
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        pass
    return value
