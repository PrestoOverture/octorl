"""Pure CPU operations for the preregistered R6 model chooser.

The module deliberately contains no model-runtime imports.  Prompt text is
loaded from the frozen preregistration at runtime and all stochastic query
seeds are explicit SHA-256 derivations.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import yaml

from src.curriculum.failure_driven import CELLS, PI0, allocate, apply_cap, failure_rates


PREREG_PATH = Path("artifacts/self_improve/contracts/r6_model_chooser_prereg.yaml")
WINDOW_UPDATES = 20
GROUPS = 40
SAMPLES = 8
MIN_VALID = 6
RHO = 0.4
MAX_P = 0.6


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def load_prompt_template(path: str | Path = PREREG_PATH) -> str:
    prereg = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    return prereg["chooser"]["prompt_template_v1"]


def stats_table(stats: Mapping[str, Mapping[str, int]]) -> str:
    rates = failure_rates(stats)
    lines = [
        "| fault type | attempts | failures | failure rate |",
        "|---|---|---|---|",
    ]
    for cell in CELLS:
        lines.append(f"| {cell} | {int(stats[cell]['n'])} | {int(stats[cell]['fail'])} | {rates[cell]:.3f} |")
    return "\n".join(lines)


def render_prompt(
    stats: Mapping[str, Mapping[str, int]],
    *,
    prereg_path: str | Path = PREREG_PATH,
) -> dict[str, str]:
    """Render using literal replacement so the template's JSON braces survive."""
    template = load_prompt_template(prereg_path)
    prompt = template.replace("{window_updates}", str(WINDOW_UPDATES))
    prompt = prompt.replace("{stats_table}", stats_table(stats))
    prompt = prompt.replace("{groups}", str(GROUPS))
    return {
        "template": template,
        "prompt": prompt,
        "template_sha256": sha256_text(template),
        "prompt_sha256": sha256_text(prompt),
    }


def query_seed(train_seed: int, stage: int, sample_index: int, *, noise: bool = False) -> int:
    if not 0 <= sample_index < SAMPLES:
        raise ValueError("sample_index must be in 0..7")
    prefix = "noise" if noise else "chooser"
    payload = f"r6|{prefix}|{int(train_seed)}|{int(stage)}|{int(sample_index)}"
    return int(hashlib.sha256(payload.encode("utf-8")).hexdigest()[:8], 16)


def _balanced_objects(text: str) -> list[str]:
    """Return balanced brace substrings while respecting JSON string escaping."""
    results: list[str] = []
    starts: list[int] = []
    in_string = False
    escaped = False
    for index, char in enumerate(text):
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == "{":
            starts.append(index)
        elif char == "}" and starts:
            start = starts.pop()
            results.append(text[start:index + 1])
    return results


def parse_completion(completion: str) -> dict[str, Any]:
    parsed: dict[str, Any] | None = None
    for candidate in reversed(_balanced_objects(completion)):
        try:
            value = json.loads(candidate)
        except (json.JSONDecodeError, TypeError):
            continue
        if isinstance(value, dict):
            parsed = value
            break
    invalid = {"valid": False, "values": parsed, "q": None, "unnormalised": False}
    if parsed is None or set(parsed) != set(CELLS):
        return invalid
    values: dict[str, float] = {}
    for cell in CELLS:
        value = parsed[cell]
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            return invalid
        number = float(value)
        if not math.isfinite(number) or number < 0:
            return invalid
        values[cell] = number
    total = sum(values.values())
    if total <= 0:
        return invalid
    return {
        "valid": True,
        "values": parsed,
        "q": {cell: values[cell] / total for cell in CELLS},
        "unnormalised": total != 100,
    }


def aggregate_samples(completions: Sequence[str]) -> dict[str, Any]:
    if len(completions) != SAMPLES:
        raise ValueError(f"a chooser stage requires exactly {SAMPLES} samples")
    samples = [parse_completion(item) for item in completions]
    valid = [item["q"] for item in samples if item["valid"]]
    fallback = len(valid) < MIN_VALID
    if fallback:
        q = dict(PI0)
    else:
        q = {cell: sum(item[cell] for item in valid) / len(valid) for cell in CELLS}
    return {"samples": samples, "valid_count": len(valid), "fallback": fallback, "q": q}


def distribution_from_q(
    q: Mapping[str, float], *, rho: float = RHO, max_p: float = MAX_P, groups: int = GROUPS,
) -> tuple[dict[str, float], dict[str, int]]:
    values = [float(q[cell]) for cell in CELLS]
    if any(not math.isfinite(value) or value < 0 for value in values):
        raise ValueError("q must contain finite nonnegative values")
    if not math.isclose(sum(values), 1.0, abs_tol=1e-12):
        raise ValueError("q must sum to one")
    before_cap = {cell: rho * PI0[cell] + (1 - rho) * float(q[cell]) for cell in CELLS}
    p = apply_cap(before_cap, PI0, max_p)
    return p, allocate(p, groups)


def tv_distance(left: Mapping[str, float], right: Mapping[str, float]) -> float:
    return 0.5 * sum(abs(float(left[cell]) - float(right[cell])) for cell in CELLS)


def cyclic_relabellings(stats: Mapping[str, Mapping[str, int]]) -> list[dict[str, dict[str, int]]]:
    """Move the stats at cell c to sigma(c) for the two non-identity cycles."""
    outputs = []
    for shift in (1, 2):
        relabelled = {}
        for index, cell in enumerate(CELLS):
            destination = CELLS[(index + shift) % len(CELLS)]
            relabelled[destination] = dict(stats[cell])
        outputs.append({cell: relabelled[cell] for cell in CELLS})
    return outputs
