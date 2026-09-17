#!/usr/bin/env python3
"""Generate a normal-only dev guardrail manifest for R3c evaluation."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from src.tasks.tool_recovery.generator import (
    SPLIT_FAMILIES,
    SPLIT_SEED_RANGES,
    generate_instance,
)


GUARDRAIL_COUNT = 30
GUARDRAIL_START_SEED = 100200
EVAL_SEED = 320000

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("artifacts/self_improve/r3c/data"),
    )
    parser.add_argument("--count", type=int, default=GUARDRAIL_COUNT)
    parser.add_argument("--start-seed", type=int, default=GUARDRAIL_START_SEED)
    args = parser.parse_args()

    permitted = SPLIT_SEED_RANGES["dev"]
    families = SPLIT_FAMILIES["dev"]
    seed = args.start_seed
    instances = []
    fingerprints: set[str] = set()

    for _ in range(args.count):
        accepted = None
        for _ in range(20):
            if seed not in permitted:
                raise RuntimeError(f"seed {seed} outside dev range")
            candidate = generate_instance(seed, families, "normal")
            seed += 1
            if candidate is None:
                continue
            if candidate.fingerprint in fingerprints:
                continue
            accepted = candidate
            break
        if accepted is None:
            raise RuntimeError("exhausted retries generating normal instance")
        fingerprints.add(accepted.fingerprint)
        instances.append(accepted)

    manifest = {
        "split": "dev_normal_guardrail",
        "count": len(instances),
        "start_seed": args.start_seed,
        "eval_seed": EVAL_SEED,
        "generator_version": instances[0].generator_version,
        "instances": [inst.to_dict() for inst in instances],
    }

    args.output_root.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output_root / "dev_normal_guardrail_manifest.json"

    encoded = json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    manifest_path.write_text(encoded, encoding="utf-8")

    sha = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    print(json.dumps({
        "manifest": str(manifest_path),
        "manifest_sha256": sha,
        "instance_count": len(instances),
        "seed_range": f"{args.start_seed}-{seed - 1}",
        "families": sorted({inst.family for inst in instances}),
    }, indent=2))


if __name__ == "__main__":
    main()
