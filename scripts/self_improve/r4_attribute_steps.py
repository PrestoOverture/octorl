#!/usr/bin/env python3
"""Join trajectory records to numbered trainer rollout dumps; abort on any mismatch."""

from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path


FAULT_TYPES = ("constraint_violation", "missing_dependency", "stale_version")
FIELDS = ("global_step", "instance_seed", "task_id", "fault_type", "binary_reward", "diagnostic_flags")


def attribute(run_dir: Path, *, start_step: int | None = None, end_step: int | None = None) -> list[dict]:
    files = sorted((run_dir / "rollouts").glob("*.jsonl"), key=lambda p: int(p.stem))
    if start_step is not None:
        files = [p for p in files if int(p.stem) >= start_step]
    if end_step is not None:
        files = [p for p in files if int(p.stem) <= end_step]
    if not files:
        raise ValueError("no numbered rollout dumps in requested step range")
    steps = [int(p.stem) for p in files]
    if steps != list(range(steps[0], steps[-1] + 1)):
        raise ValueError(f"rollout step gap: {steps}")
    trajectories = [json.loads(line) for line in (run_dir / "trajectories.jsonl").open(encoding="utf-8")]
    offset = 0
    output = []
    for path, step in zip(files, steps):
        dumped = [json.loads(line) for line in path.open(encoding="utf-8")]
        batch = trajectories[offset:offset + len(dumped)]
        if len(batch) != len(dumped):
            raise ValueError(f"step {step}: trajectory count below rollout dump count")
        actual = collections.Counter((row["task_id"], row["binary_reward"]) for row in batch)
        expected = collections.Counter((row["gts"], int(row["score"])) for row in dumped)
        if actual != expected:
            raise ValueError(f"step {step}: trajectory/dump task and binary-score multiset mismatch")
        for row in batch:
            task_id = row["task_id"]
            fault_type = next((cell for cell in FAULT_TYPES if task_id.endswith("_" + cell)), None)
            if fault_type is None:
                raise ValueError(f"step {step}: unknown fault_type in task_id {task_id}")
            record = {"global_step": step, "instance_seed": row["instance_seed"], "task_id": task_id,
                      "fault_type": fault_type, "binary_reward": row["binary_reward"],
                      "diagnostic_flags": row["diagnostic_flags"]}
            for field in ("run_id", "arm", "seed", "stage"):
                if field in row:
                    record[field] = row[field]
            output.append(record)
        offset += len(dumped)
    if end_step is None and offset != len(trajectories):
        raise ValueError(f"{len(trajectories) - offset} extra trajectory records after last dump")
    return output


def write_records(path: Path, rows: list[dict]) -> None:
    # Validate fully before opening the output, so a bad step cannot leave a partial selector input.
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, sort_keys=True) + "\n")
    tmp.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--start-step", type=int)
    parser.add_argument("--end-step", type=int)
    args = parser.parse_args()
    rows = attribute(args.run_dir, start_step=args.start_step, end_step=args.end_step)
    write_records(args.output, rows)
    print(json.dumps({"rows": len(rows), "steps": sorted({row["global_step"] for row in rows})}))


if __name__ == "__main__":
    main()
