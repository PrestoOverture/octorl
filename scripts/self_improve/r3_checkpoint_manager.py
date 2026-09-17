#!/usr/bin/env python3
"""Bound R3 checkpoint disk use while preserving LoRA and optimizer artifacts."""

from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path


STEP_RE = re.compile(r"global_step_(\d+)$")


def checkpoints(root: Path) -> list[tuple[int, Path]]:
    found = []
    for path in root.glob("global_step_*"):
        match = STEP_RE.fullmatch(path.name)
        if match and path.is_dir():
            found.append((int(match.group(1)), path))
    return sorted(found)


def _complete(path: Path) -> bool:
    actor = path / "actor"
    return (
        (actor / "lora_adapter" / "adapter_model.safetensors").is_file()
        and (actor / "lora_adapter" / "adapter_config.json").is_file()
        and any(actor.glob("optim_world_size_*_rank_*.pt"))
        and (path / "data.pt").is_file()
    )


def prune(root: Path, log_path: Path) -> list[dict]:
    found = checkpoints(root)
    if len(found) < 2:
        return []
    latest = found[-1][0]
    events = []
    for step, path in found:
        if step == latest or not _complete(path):
            continue
        for shard in (path / "actor").glob("model_world_size_*_rank_*.pt"):
            size = shard.stat().st_size
            shard.unlink()
            events.append({"step": step, "removed": str(shard), "bytes": size})
    if events:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("a", encoding="utf-8") as stream:
            for event in events:
                stream.write(json.dumps(event, sort_keys=True) + "\n")
    return events


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--log", type=Path, required=True)
    parser.add_argument("--watch", action="store_true")
    parser.add_argument("--interval", type=float, default=15.0)
    args = parser.parse_args()
    if not args.watch:
        print(json.dumps(prune(args.root, args.log)))
        return
    while True:
        prune(args.root, args.log)
        time.sleep(args.interval)


if __name__ == "__main__":
    main()

