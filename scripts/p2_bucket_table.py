"""Compile P2 bucket baselines from the frozen P1.24 evaluation."""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = ROOT / "artifacts/p1/baseline_eval.json"
DEFAULT_OUTPUT = ROOT / "artifacts/p2/bucket_table.json"
EASY_BUCKET = "c1-L0-single-function"


def compile_bucket_table(data: dict) -> dict:
    """Return bucket-level pass rates and population counts."""
    specs_by_cell = {
        (spec["repo_name"], spec["bucket"]): spec for spec in data["specs"]
    }
    grouped: dict[str, list[tuple[str, dict]]] = defaultdict(list)
    for cell_key, result in data["pass_at_1"].items():
        repo, bucket = cell_key.split("|", 1)
        if (repo, bucket) not in specs_by_cell:
            raise ValueError(f"result has no matching spec: {cell_key}")
        grouped[bucket].append((repo, result))

    buckets = {}
    for bucket, cells in sorted(grouped.items()):
        total_rollouts = sum(cell["rollouts"] for _, cell in cells)
        total_successes = sum(cell["mean_reward"] * cell["rollouts"]
                              for _, cell in cells)
        n_instances = len(cells)
        n_train = sum(specs_by_cell[(repo, bucket)]["repo"].find("/repos/train/") >= 0
                      for repo, _ in cells)
        n_held_out = sum(specs_by_cell[(repo, bucket)]["repo"].find("/repos/held_out/") >= 0
                         for repo, _ in cells)
        if n_train + n_held_out != n_instances:
            raise ValueError(f"cannot classify repository split for bucket {bucket}")
        buckets[bucket] = {
            "pass_at_1": total_successes / total_rollouts if total_rollouts else 0.0,
            "pass_at_8": sum(cell["mean_reward"] > 0 for _, cell in cells) / n_instances,
            "mixed_group_ratio": sum(0 < cell["mean_reward"] < 1 for _, cell in cells)
                                 / n_instances,
            "n_instances": n_instances,
            "n_train": n_train,
            "n_held_out": n_held_out,
        }

    if EASY_BUCKET not in buckets:
        raise ValueError(f"easy bucket missing: {EASY_BUCKET}")
    return {
        "source": str(DEFAULT_INPUT.relative_to(ROOT)),
        "rollouts_per_instance": data["model"]["rollouts_per_instance"],
        "buckets": buckets,
        "easy_bucket": {"bucket": EASY_BUCKET, **buckets[EASY_BUCKET]},
    }


def display_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT))
    except ValueError:
        return str(path)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    result = compile_bucket_table(json.loads(args.input.read_text()))
    result["source"] = display_path(args.input)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(f"wrote {len(result['buckets'])} buckets to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
