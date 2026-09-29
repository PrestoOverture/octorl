#!/usr/bin/env python3
"""Generate and seal the preregistered R6 test3 sample."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd

from scripts.self_improve.r3_prepare import _row
from src.tasks.tool_recovery.generator import FAMILY_BY_NAME, PROTOCOL_VERSION, generate_instance


COUNTS = {"constraint_violation": 60, "missing_dependency": 52, "stale_version": 48, "normal": 40}
FAMILIES = ["deployment_service", "notification_service"]
START_SEED = 320000
MAX_SEED = 420000


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_identical_or_new(path: Path, content: bytes) -> None:
    if path.exists() and path.read_bytes() != content:
        raise RuntimeError(f"refusing to overwrite sealed output with different bytes: {path}")
    path.write_bytes(content)


def generate(workspace: Path, output_dir: Path | None = None) -> dict[str, Any]:
    target = output_dir or workspace / "artifacts/self_improve/r6/data"
    sources = {
        "train": workspace / "artifacts/self_improve/r3/data/train_manifest.json",
        "dev": workspace / "artifacts/self_improve/r3/data/dev_manifest.json",
        "test": workspace / "artifacts/self_improve/r3/data/test_manifest.json",
        "test2": workspace / "artifacts/self_improve/r4r/data/test2_manifest.json",
    }
    prior = {name: json.loads(path.read_text(encoding="utf-8")) for name, path in sources.items()}
    if PROTOCOL_VERSION != "r2.0" or any(data["generator_version"] != PROTOCOL_VERSION for data in prior.values()):
        raise RuntimeError("test3 requires generator r2.0 and matching prior manifests")
    used = {item["fingerprint"] for data in prior.values() for item in data["instances"]}
    instances: list[dict[str, Any]] = []
    rejected: list[int] = []
    duplicates: list[int] = []
    seed = START_SEED
    families = [FAMILY_BY_NAME[name] for name in FAMILIES]
    for fault_type, count in COUNTS.items():
        accepted = 0
        while accepted < count and seed < MAX_SEED:
            candidate_seed = seed
            instance = generate_instance(candidate_seed, families, fault_type)
            seed += 1
            if instance is None:
                rejected.append(candidate_seed)
            elif instance.fingerprint in used:
                duplicates.append(candidate_seed)
            else:
                used.add(instance.fingerprint)
                instances.append(instance.to_dict())
                accepted += 1
        if accepted != count:
            raise RuntimeError(f"cannot reach {fault_type}={count}; generated {accepted}")
    manifest = {
        "split": "test3", "requested_count": 200, "generated_count": len(instances),
        "start_seed": START_SEED, "end_seed_exclusive": seed, "generator_version": PROTOCOL_VERSION,
        "instances": instances, "rejected_seeds": rejected, "duplicate_seeds": duplicates,
    }
    target.mkdir(parents=True, exist_ok=True)
    manifest_path = target / "test3_manifest.json"
    parquet_path = target / "test3.parquet"
    manifest_bytes = (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode()
    _write_identical_or_new(manifest_path, manifest_bytes)

    temporary = target / ".test3.parquet.generated"
    pd.DataFrame([_row(item, index) for index, item in enumerate(instances)]).to_parquet(temporary, index=False)
    parquet_bytes = temporary.read_bytes()
    temporary.unlink()
    _write_identical_or_new(parquet_path, parquet_bytes)

    fingerprints = {item["fingerprint"] for item in instances}
    dedup = {
        name: {"manifest_sha256": sha256(path),
               "overlap_count": len(fingerprints & {item["fingerprint"] for item in data["instances"]})}
        for (name, path), data in zip(sources.items(), prior.values(), strict=True)
    }
    if any(item["overlap_count"] for item in dedup.values()):
        raise RuntimeError(f"test3 fingerprint overlap detected: {dedup}")
    seal = {
        "manifest_sha256": sha256(manifest_path), "parquet_sha256": sha256(parquet_path),
        "generator_version": PROTOCOL_VERSION, "start_seed": START_SEED,
        "end_seed_exclusive": seed, "seed_range": [START_SEED, seed - 1],
        "counts": dict(Counter(item["fault_type"] for item in instances)),
        "families": FAMILIES, "unique_fingerprints": len(fingerprints),
        "model_evaluated": False, "dedup_evidence": dedup,
    }
    seal_path = target / "test3_seal.json"
    _write_identical_or_new(seal_path, (json.dumps(seal, indent=2, sort_keys=True) + "\n").encode())
    return seal


if __name__ == "__main__":
    workspace = Path(__file__).resolve().parents[2]
    print(json.dumps(generate(workspace), indent=2, sort_keys=True))
