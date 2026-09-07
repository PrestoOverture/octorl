"""Deterministic prompt construction, independent of model sampling."""
import json
from pathlib import Path
from typing import Any


def build_prompt(instance: dict[str, Any], history: list[dict[str, Any]], seed: int, *, prompt_path: Path | None = None) -> list[dict[str, str]]:
    if type(seed) is not int:
        raise TypeError("seed must be an explicit integer")
    prompt_path = prompt_path or Path(__file__).with_name("prompt.md")
    return [{"role": "system", "content": prompt_path.read_text()}, {"role": "user", "content": json.dumps(instance, sort_keys=True, ensure_ascii=False)}] + [{"role": entry["role"], "content": entry["content"]} for entry in history]
