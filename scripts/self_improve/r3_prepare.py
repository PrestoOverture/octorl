#!/usr/bin/env python3
"""Freeze R3 train/dev/test manifests and Agent-R1 parquet inputs."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import pandas as pd

from src.tasks.tool_recovery.generator import allocate_slots, generate_dataset


SPLITS = {
    "train": (400, 0),
    "dev": (50, 100000),
    "test": (50, 200000),
}

SYSTEM_PROMPT = (
    "You repair a service configuration using the available tools. Read the "
    "configuration, query the relevant field constraint and state variables, "
    "submit only the necessary fix, then respond with a short final message."
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _row(instance: dict[str, Any], index: int) -> dict[str, Any]:
    return {
        "data_source": "octorl_tool_recovery_r2",
        "prompt": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    f"Repair task {instance['task_id']} for service {instance['family']}. "
                    "Use the tools to inspect and repair it."
                ),
            },
        ],
        "agent_name": "octorl_tool_recovery",
        "env_kwargs": json.dumps({"instance": instance}, sort_keys=True),
        "reward_model": {"style": "rule", "ground_truth": instance["task_id"]},
        "extra_info": {
            "task_id": instance["task_id"],
            "index": index,
            "fault_type": instance["fault_type"],
            "instance_seed": instance["seed"],
        },
    }


def freeze(output_root: Path) -> dict[str, Any]:
    output_root.mkdir(parents=True, exist_ok=True)
    summary: dict[str, Any] = {"splits": {}}
    fingerprints: dict[str, set[str]] = {}
    for split, (count, start_seed) in SPLITS.items():
        generated = generate_dataset(split, count, start_seed)
        if generated.generated_count != count or generated.rejections:
            raise AssertionError(
                f"{split}: expected {count} accepted instances; got "
                f"{generated.generated_count} with {len(generated.rejections)} rejections"
            )
        data = generated.to_dict()
        data["generated_count"] = generated.generated_count
        data["slot_allocation"] = allocate_slots(count)
        manifest_path = output_root / f"{split}_manifest.json"
        parquet_path = output_root / f"{split}.parquet"
        encoded = json.dumps(data, indent=2, sort_keys=True) + "\n"
        if manifest_path.exists() and manifest_path.read_text(encoding="utf-8") != encoded:
            raise RuntimeError(f"refusing to overwrite frozen manifest with different data: {manifest_path}")
        manifest_path.write_text(encoded, encoding="utf-8")
        pd.DataFrame([_row(item, index) for index, item in enumerate(data["instances"])]).to_parquet(
            parquet_path, index=False
        )
        fingerprints[split] = {item["fingerprint"] for item in data["instances"]}
        summary["splits"][split] = {
            "count": count,
            "start_seed": start_seed,
            "manifest": str(manifest_path),
            "manifest_sha256": _sha256(manifest_path),
            "parquet": str(parquet_path),
            "parquet_sha256": _sha256(parquet_path),
        }

    for left, right in (("train", "dev"), ("train", "test"), ("dev", "test")):
        overlap = fingerprints[left] & fingerprints[right]
        if overlap:
            raise AssertionError(f"content leakage between {left} and {right}: {len(overlap)} fingerprints")
    summary["content_fingerprint_overlap"] = 0
    summary_path = output_root / "dataset_manifest.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, default=Path("artifacts/self_improve/r3/data"))
    args = parser.parse_args()
    print(json.dumps(freeze(args.output_root), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
