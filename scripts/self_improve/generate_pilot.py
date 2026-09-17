#!/usr/bin/env python3
"""Generate the frozen r0.4 dev manifest for the base-model pilot."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.tasks.tool_recovery.generator import allocate_slots, generate_dataset


def generate_manifest(output: Path, count: int = 50, start_seed: int = 100000) -> dict:
    manifest = generate_dataset(split="dev", count=count, start_seed=start_seed)
    data = manifest.to_dict()
    data["generated_count"] = manifest.generated_count
    data["slot_allocation"] = allocate_slots(count)
    if manifest.generated_count < int(count * 0.9):
        warning = "generated fewer than 90% of requested instances"
        if warning not in data["warnings"]:
            data["warnings"].append(warning)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return data


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("artifacts/self_improve/pilot/manifest.json"))
    parser.add_argument("--count", type=int, default=50)
    parser.add_argument("--start-seed", type=int, default=100000)
    args = parser.parse_args()
    data = generate_manifest(args.output, args.count, args.start_seed)
    print(json.dumps({"path": str(args.output), "generated_count": len(data["instances"]), "warnings": data["warnings"]}))


if __name__ == "__main__":
    main()
