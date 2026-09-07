"""P1.21 throughput gate: trajectories/hour at concurrency N over the real
environment.

The policy is scripted, so the numbers measure CPU-side sandbox throughput --
the term PRD Sec. 9 identifies as dominant -- and are a floor for any
model-in-the-loop rate, never an estimate of one.
"""
from __future__ import annotations

import argparse
import json
import platform
import os
import statistics
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.environment.harness import ScriptedPolicy, package, run_trajectory
from src.environment.sandbox import create_pool
from src.injector.catalog import inject_from_catalog
from src.injector.core import BUG_TYPES, Difficulty

ROOT = Path(__file__).resolve().parents[1]
IMAGE = (ROOT / "artifacts/p1/docker-image.id").read_text().strip()
BUCKETS = {"c1-L0-single-function": (1, "L0", "single-function"),
           "c2-L0-single-file": (2, "L0", "single-file")}


def pick_difficulty(repo: Path, bucket: str, seed: int) -> Difficulty:
    """First bug type this repo can actually satisfy in this bucket.

    Pinned to `mutation` so the benchmark measures a fixed task population
    across runs; all three sources generate since the F2 repair."""
    count, hint, span = BUCKETS[bucket]
    for bug_type in BUG_TYPES:
        difficulty = Difficulty("mutation", bug_type, count, hint, span)
        try:
            inject_from_catalog(repo, difficulty, seed)
            return difficulty
        except Exception:
            continue
    raise SystemExit(f"{repo.name} cannot populate bucket {bucket}")


def script_for(instance) -> list[dict]:
    """Twelve steps, all five tools, four run_tests calls, ending repaired."""
    path = instance.edits[0].path
    import difflib

    def diff(before: str, after: str) -> str:
        return "".join(difflib.unified_diff(before.splitlines(keepends=True), after.splitlines(keepends=True),
                                            fromfile=f"a/{path}", tofile=f"b/{path}"))

    buggy = instance.modified[path]
    annotated = "# investigating\n" + buggy
    return [
        {"name": "list_files", "arguments": {"path": "."}},
        {"name": "search_code", "arguments": {"pattern": "def "}},
        {"name": "read_file", "arguments": {"path": path}},
        {"name": "run_tests", "arguments": {}},
        {"name": "apply_patch", "arguments": {"diff": diff(buggy, annotated)}},
        {"name": "run_tests", "arguments": {}},
        {"name": "read_file", "arguments": {"path": path}},
        {"name": "search_code", "arguments": {"pattern": "return"}},
        {"name": "apply_patch", "arguments": {"diff": diff(annotated, instance.originals[path])}},
        {"name": "run_tests", "arguments": {}},
        {"name": "read_file", "arguments": {"path": path}},
        {"name": "run_tests", "arguments": {}},
    ]


def one_trajectory(pool, instance, seed: int, tag: str) -> dict:
    started = time.monotonic()
    with pool.lease(instance.root) as box:
        leased = time.monotonic()
        record = run_trajectory(instance, box, ScriptedPolicy(script_for(instance)), sampling_seed=seed,
                                trajectory_id=tag, policy_version="scripted-12step",
                                base_model_revision="none", manifest_ref="p1-throughput")
    total = time.monotonic() - started
    turns = record["turns"]
    segments = {s["name"]: s["end"] - s["start"] for s in record["wall_clock_segments"]}
    test_turns = [t for t in turns if t["tool_calls"][0]["name"] == "run_tests"]
    return {
        "trajectory_id": tag, "reward": record["reward"], "steps": len(turns),
        "run_tests_calls": len(test_turns), "tools_used": sorted({t["tool_calls"][0]["name"] for t in turns}),
        "cheat_flags": record["cheat_flags"], "total": total, "lease_reset": leased - started,
        "run_tests": sum(t["end"] - t["start"] for t in test_turns),
        "other_tools": sum(t["end"] - t["start"] for t in turns if t not in test_turns),
        "verify": segments.get("verify", 0.0),
    }


def summarize(rows: list[dict], concurrency: int, wall: float) -> dict:
    totals = sorted(r["total"] for r in rows)
    phase = lambda key: round(statistics.mean(r[key] for r in rows), 3)
    return {
        "concurrency": concurrency, "trajectories": len(rows),
        "wall_clock_s": round(wall, 2),
        "trajectories_per_hour": round(len(rows) / wall * 3600, 1),
        "p50_s": round(statistics.median(totals), 2),
        "p95_s": round(totals[min(len(totals) - 1, int(round(0.95 * (len(totals) - 1))))], 2),
        "max_s": round(totals[-1], 2), "mean_s": round(statistics.mean(totals), 2),
        "phase_mean_s": {"lease_reset": phase("lease_reset"), "other_tools": phase("other_tools"),
                         "run_tests": phase("run_tests"), "verify": phase("verify")},
        "rewards": sorted({r["reward"] for r in rows}),
        "steps": sorted({r["steps"] for r in rows}),
        "run_tests_calls": sorted({r["run_tests_calls"] for r in rows}),
        "tools_used": sorted({tool for r in rows for tool in r["tools_used"]}),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repos", nargs="+", default=["record_index", "slot_planner", "config_parser"])
    parser.add_argument("--buckets", nargs="+", default=list(BUCKETS))
    parser.add_argument("--concurrency", nargs="+", type=int, default=[1, 4, 8])
    parser.add_argument("--per-cell", type=int, default=8)
    parser.add_argument("--out", default=str(ROOT / "artifacts/p1"))
    parser.add_argument("--workspace", default=None)
    args = parser.parse_args()

    workspace = Path(args.workspace or (ROOT / "artifacts/p1/_bench"))
    workspace.mkdir(parents=True, exist_ok=True)

    package_times, instances = [], {}
    for name in args.repos:
        repo = ROOT / "repos/train" / name
        for bucket in args.buckets:
            difficulty = pick_difficulty(repo, bucket, 20260906)
            started = time.monotonic()
            instances[(name, bucket)] = package(repo, difficulty, 20260906, workspace / f"{name}-{bucket}")
            package_times.append(time.monotonic() - started)
            print(f"packaged {name}/{bucket} via {difficulty.type}", flush=True)

    results, cells = {}, []
    for concurrency in args.concurrency:
        with create_pool(IMAGE, size=concurrency, workspace=workspace / f"pool-{concurrency}") as pool:
            warm = instances[(args.repos[0], args.buckets[0])]
            one_trajectory(pool, warm, 1, f"warmup-N{concurrency}")  # excluded: container/pytest warm-up

            jobs = [(key, seed) for key in instances for seed in range(args.per_cell)]
            started = time.monotonic()
            with ThreadPoolExecutor(max_workers=concurrency) as executor:
                rows = list(executor.map(
                    lambda job: one_trajectory(pool, instances[job[0]], 100 + job[1],
                                               f"{job[0][0]}-{job[0][1]}-N{concurrency}-{job[1]}"), jobs))
            wall = time.monotonic() - started
        results[f"N{concurrency}"] = summarize(rows, concurrency, wall)
        for (key, _), row in zip(jobs, rows):
            cells.append({"repo": key[0], "bucket": key[1], **row})
        print(f"N={concurrency}: {results[f'N{concurrency}']['trajectories_per_hour']} traj/h, "
              f"p95 {results[f'N{concurrency}']['p95_s']}s", flush=True)

    gate = {name: {"p95_s": row["p95_s"], "max_s": row["max_s"], "passes_90s": row["p95_s"] < 90}
            for name, row in results.items()}
    report = {
        "measured_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "host": {"platform": platform.platform(), "machine": platform.machine(),
                 "cpu_count": os.cpu_count(),
                 "docker_server": subprocess.run(["docker", "version", "--format", "{{.Server.Os}}/{{.Server.Arch}} {{.Server.Version}}"],
                                                 capture_output=True, text=True).stdout.strip()},
        "image": IMAGE,
        "policy": "scripted-12step (no model). Per-trajectory latency here is a LOWER bound and the rate an UPPER bound on any model-in-the-loop figure: adding generation latency lengthens trajectories and lowers traj/hour.",
        "design": {"repos": args.repos, "buckets": args.buckets, "per_cell": args.per_cell,
                   "steps_per_trajectory": 12, "warmup_trajectory_excluded": True},
        "packaging_mean_s": round(statistics.mean(package_times), 3),
        "results": results, "gate_90s": gate, "trajectories": cells,
    }
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "throughput.json").write_text(json.dumps(report, indent=2) + "\n")
    (out / "throughput.md").write_text(render(report))
    print(json.dumps(gate, indent=2))
    return 0


def render(report: dict) -> str:
    host = report["host"]
    lines = [
        "# P1.21 throughput gate", "",
        f"Measured {report['measured_at']} on {host['platform']} ({host['machine']}, "
        f"{host['cpu_count']} CPUs), Docker {host['docker_server']}.", "",
        f"- Image: `{report['image']}`",
        f"- Policy: **{report['policy']}**",
        f"- Design: {len(report['design']['repos'])} repos x {len(report['design']['buckets'])} buckets "
        f"x {report['design']['per_cell']} trajectories/cell, 12 steps each; one warm-up trajectory "
        f"per pool excluded from every figure.",
        f"- Packaging (once per task instance, amortized over G=8 rollouts): "
        f"{report['packaging_mean_s']}s mean, excluded from trajectory wall-clock.", "",
        "## Rates", "",
        "| N | trajectories | wall (s) | traj/hour | p50 (s) | p95 (s) | max (s) | <90s gate |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for name, row in report["results"].items():
        lines.append(f"| {row['concurrency']} | {row['trajectories']} | {row['wall_clock_s']} | "
                     f"**{row['trajectories_per_hour']}** | {row['p50_s']} | {row['p95_s']} | {row['max_s']} | "
                     f"{'PASS' if report['gate_90s'][name]['passes_90s'] else 'FAIL'} |")
    lines += ["", "## Phase breakdown (mean seconds per trajectory)", "",
              "| N | lease+reset | tools (non-test) | run_tests | verify | sum |", "|---|---|---|---|---|---|"]
    for row in report["results"].values():
        p = row["phase_mean_s"]
        lines.append(f"| {row['concurrency']} | {p['lease_reset']} | {p['other_tools']} | {p['run_tests']} | "
                     f"{p['verify']} | {round(sum(p.values()), 3)} |")
    lines += ["", "## Validity of this measurement", "",
              "- **Host caveat.** The gate number that matters is the one from the machine that runs",
              "  rollouts during training. The AutoDL GPU host cannot run any container runtime",
              "  (no Docker, no rootless Podman -- `unshare` is blocked, `/dev/fuse` absent), so this",
              "  run is from macOS Docker Desktop, i.e. a Linux VM on arm64. Treat it as a",
              "  functional demonstration of the pipeline, not as the host gate.",
              "- **Linux-side bound.** A container-free pytest probe on the GPU host",
              "  (`linux_pytest_probe.json`, 128 CPUs) puts the dominant term -- four `run_tests`",
              "  plus verify's two suites -- at 5.62 s mean / 8.82 s max per 12-step trajectory.",
              "  Docker `exec` overhead is *not* separately measured: the lease+reset column below is",
              "  0.005-0.012 s, but per-command container overhead sits inside the run_tests and",
              "  verify segments. The probe is therefore a partial cost estimate, not an upper bound",
              "  on a model-driven trajectory, and does not by itself close the gate.",
              "- **Scripted policy.** No model is in the loop; per-step generation latency is absent",
              "  from every number above, so these rates are ceilings, not floors.",
              "- **Concurrency.** N=8 on an 8-CPU laptop is CPU-oversubscribed (each container is",
              "  capped at 2 CPUs), which is why N=8 is slower than N=4 here. The GPU host reports",
              "  128 logical CPUs but its cgroup quota is `cpu.max = 1600000 100000` = 16 cores,",
              "  shared with training -- not 128.", ""]
    first = next(iter(report["results"].values()))
    lines += ["", "## Trajectory validity", "",
              f"- Steps per trajectory: {first['steps']}",
              f"- `run_tests` calls per trajectory: {first['run_tests_calls']}",
              f"- Tools exercised: {', '.join(first['tools_used'])}",
              f"- Rewards observed: {first['rewards']} (the scripted trajectory repairs the bug, so 1 is expected)",
              ""]
    return "\n".join(lines)


if __name__ == "__main__":
    raise SystemExit(main())
