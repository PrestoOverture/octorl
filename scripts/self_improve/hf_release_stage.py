#!/usr/bin/env python3
"""Stage the LoRA adapters for Hugging Face release.

Weights are hard-linked, so they are byte-identical to the archived adapters and their manifest
sha256. Only two fields of adapter_config.json change:

- target_modules: the training stack serialised the regex as a list of single characters, which
  PEFT cannot match. It is restored to the regex string; r5_load_adapter.normalize_modules does
  the same repair in memory.
- base_model_name_or_path and revision: the remote path is replaced by the public model id and
  the exact base revision used in training.

Model cards are written by hand and are not generated here.
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts/self_improve"))
from r5_load_adapter import normalize_modules  # noqa: E402

MANIFEST = ROOT / "artifacts/self_improve/r5/checkpoint_manifest.json"
OUT = ROOT / "artifacts/self_improve/hf_release"
BASE_ID = "Qwen/Qwen3-4B"
BASE_REVISION = "1cfa9a7208912126459214e8b04321603b3df60c"
PUBLIC = {f"r4r_{arm}_{seed}_u80" for arm in ("fixed", "failure_driven") for seed in (42, 137, 2718)}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def stage() -> dict:
    rows = json.loads(MANIFEST.read_text())
    if len(rows) != 20 or not PUBLIC <= {row["name"] for row in rows}:
        raise SystemExit("manifest does not hold the expected 20 adapters")
    release = {"public": [], "archive": []}
    for row in rows:
        src = ROOT / row["local_path"]
        for name, expected in row["files"].items():
            if sha256(src / name) != expected:
                raise SystemExit(f"local file differs from manifest: {src / name}")
        target = "public" if row["name"] in PUBLIC else "archive"
        dst = OUT / target / row["name"]
        dst.mkdir(parents=True, exist_ok=True)
        weights = dst / "adapter_model.safetensors"
        if weights.exists():
            weights.unlink()
        os.link(src / "adapter_model.safetensors", weights)
        config = json.loads((src / "adapter_config.json").read_text())
        config["target_modules"] = normalize_modules(config["target_modules"])
        config["base_model_name_or_path"] = BASE_ID
        config["revision"] = BASE_REVISION
        (dst / "adapter_config.json").write_text(json.dumps(config, indent=2, sort_keys=True) + "\n")
        release[target].append({
            "name": row["name"], "global_step": row["global_step"], "lineage_parent": row["lineage_parent"],
            "path_in_repo": row["name"],
            "adapter_model.safetensors": sha256(weights),
            "adapter_config.json": sha256(dst / "adapter_config.json"),
            "original_adapter_config_sha256": row["files"]["adapter_config.json"],
            "test_eval_json": row.get("test_eval_json"),
        })
    for target, entries in release.items():
        record = {"base_model": BASE_ID, "base_revision": BASE_REVISION,
                  "source_manifest": "artifacts/self_improve/r5/checkpoint_manifest.json",
                  "config_changes": ["target_modules: char-list -> regex string",
                                     "base_model_name_or_path/revision: remote path -> public id + revision"],
                  "adapters": sorted(entries, key=lambda e: e["name"])}
        (OUT / target / "release_manifest.json").write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
    return {target: len(entries) for target, entries in release.items()}


if __name__ == "__main__":
    print(json.dumps(stage()))
