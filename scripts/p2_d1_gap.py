"""P2.5 / D1 prequential aggregation-gap measurement.

The task pool is frozen before any model rollout is evaluated.  Group records
are appended durably so the 4,400-rollout run can resume without changing the
prequential history or resampling tasks.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import shutil
import subprocess
import sys
import tempfile
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

try:
    from scripts.baseline_eval import IMAGE, evaluate_instance, satisfiable
except ModuleNotFoundError:  # direct execution where an installed `scripts` shadows this directory
    from baseline_eval import IMAGE, evaluate_instance, satisfiable
from src.environment.harness import admit, package
from src.environment.sandbox import create_pool
from src.injector.buckets import BUCKETS
from src.injector.core import Difficulty
from src.sampling.betabinom import est_betabinom
from src.sampling.bucket import est_plugin
from src.sampling.common import G, f
from src.sampling.empirical import est_empirical

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = ROOT / "artifacts/p2/d1"
SOURCES = ("mutation", "commit-rollback", "feat-add")
BUCKET_PLAN = {
    "c2-L0-single-function": 200,
    "c3-L0-single-file": 200,
    "c1-L0-single-function": 75,
    "c3-L0-single-function": 75,
}
HARD_BUCKETS = {"c2-L0-single-function", "c3-L0-single-file"}
POOL_SEEDS = range(50)
BOOTSTRAP_SEED = 20260911
SAMPLING_MASTER_SEED = 20260910


def stable_seed(*parts: object) -> int:
    payload = "|".join(map(str, parts)).encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


def repos() -> list[Path]:
    paths = sorted(p for p in (ROOT / "repos/train").iterdir() if p.is_dir())
    paths += sorted(p for p in (ROOT / "repos/held_out").iterdir() if p.is_dir())
    return paths


def atomic_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    tmp.replace(path)


def prepare_catalog_history(out_dir: Path) -> dict[str, str]:
    """Provide semantic replacements for catalog commits lost to Git GC.

    The catalog's historical commits are provenance validators: injection uses
    its recorded before/after fragments, while ``git show`` proves those
    fragments belonged to the named transition.  The original unreachable
    objects are no longer in the project clone, so construct deterministic
    two-commit repositories from those frozen fragments in a separate bare
    object database.  Git replacement refs retain the catalog IDs without
    modifying a repository, catalog, injector, or task source file.
    """
    replacements_path = out_dir / "catalog_history_replacements.json"
    history_git = out_dir / "catalog_history.git"
    if replacements_path.exists() and history_git.exists():
        mappings = json.loads(replacements_path.read_text())
        os.environ["GIT_DIR"] = str(history_git.resolve())
        return mappings
    if history_git.exists():
        shutil.rmtree(history_git)
    subprocess.run(["git", "init", "--bare", "-q", str(history_git)], check=True)
    mappings: dict[str, str] = {}
    grouped: dict[tuple[Path, str], list[dict]] = {}
    for repo in repos():
        catalog = json.loads((repo / "injector_sources.json").read_text())
        for entry in catalog["history"]:
            grouped.setdefault((repo, entry["fix_commit"]), []).append(entry)

    commit_env = dict(os.environ)
    commit_env.update({
        "GIT_AUTHOR_DATE": "2000-01-01T00:00:00+00:00",
        "GIT_COMMITTER_DATE": "2000-01-01T00:00:00+00:00",
    })
    for (repo, catalog_commit), entries in sorted(grouped.items(), key=lambda item: str(item[0])):
        with tempfile.TemporaryDirectory(prefix="octorl-catalog-history-") as temp_name:
            temp = Path(temp_name)
            subprocess.run(["git", "init", "-q", str(temp)], check=True)
            subprocess.run(["git", "-C", str(temp), "config", "user.name", "OctoRL"], check=True)
            subprocess.run(["git", "-C", str(temp), "config", "user.email", "octorl@local"], check=True)
            # Only provenance membership is queried from these commits.  Use a
            # fragment ledger rather than today's source tree because a later
            # catalog transition may intentionally touch the same function.
            current = {
                path: "\n\n".join(entry["after"] for entry in entries if entry["path"] == path)
                for path in sorted({entry["path"] for entry in entries})
            }
            parent = {
                path: "\n\n".join(entry["before"] for entry in entries if entry["path"] == path)
                for path in sorted({entry["path"] for entry in entries})
            }
            for path, content in parent.items():
                destination = temp / path
                destination.parent.mkdir(parents=True, exist_ok=True)
                destination.write_text(content)
            subprocess.run(["git", "-C", str(temp), "add", "."], check=True)
            subprocess.run(["git", "-C", str(temp), "commit", "-qm", "buggy"],
                           check=True, env=commit_env)
            for path, content in current.items():
                (temp / path).write_text(content)
            subprocess.run(["git", "-C", str(temp), "add", "."], check=True)
            subprocess.run(["git", "-C", str(temp), "commit", "-qm", "fix"],
                           check=True, env=commit_env)
            replacement = subprocess.check_output(
                ["git", "-C", str(temp), "rev-parse", "HEAD"], text=True).strip()
            subprocess.run(["git", f"--git-dir={history_git}", "fetch", "-q", str(temp), replacement],
                           check=True)
            subprocess.run(["git", f"--git-dir={history_git}", "update-ref",
                            f"refs/replace/{catalog_commit}", replacement], check=True)
            mappings[catalog_commit] = replacement

    history_env = dict(os.environ)
    history_env["GIT_DIR"] = str(history_git.resolve())
    for (repo, catalog_commit), entries in grouped.items():
        for entry in entries:
            before = subprocess.check_output(
                ["git", "-C", str(repo), "show", f"{catalog_commit}^:{entry['path']}"],
                text=True, env=history_env)
            after = subprocess.check_output(
                ["git", "-C", str(repo), "show", f"{catalog_commit}:{entry['path']}"],
                text=True, env=history_env)
            if entry["before"] not in before or entry["after"] not in after:
                raise ValueError(f"catalog history reconstruction failed: {repo.name} {catalog_commit}")
    os.environ["GIT_DIR"] = str(history_git.resolve())
    atomic_json(out_dir / "catalog_history_replacements.json", mappings)
    return mappings


def build_pool_manifest(out_dir: Path, pool_size: int) -> list[dict]:
    """Enumerate and admit every pre-registered (repo, source, seed) triple."""
    wanted = {bucket.id: bucket for bucket in BUCKETS}
    workspace = Path(tempfile.mkdtemp(prefix="octorl-d1-admission-"))
    tasks: list[dict] = []
    enumeration_path = out_dir / "pool_enumeration.jsonl"
    all_work = [(bucket_id, repo, source, seed)
                for bucket_id in BUCKET_PLAN
                for repo in repos()
                for source in SOURCES
                for seed in POOL_SEEDS]
    # Candidate existence is seed-invariant: the seed only shuffles candidates
    # before combinations are searched.  Prove feasibility once per structural
    # cell, then still package/admit every one of its 50 seeded tasks.
    structural_cells = [(bucket_id, repo, source)
                        for bucket_id in BUCKET_PLAN
                        for repo in repos()
                        for source in SOURCES]

    def check_feasibility(cell: tuple[str, Path, str]):
        bucket_id, repo, source = cell
        difficulty = satisfiable(repo, wanted[bucket_id], 0, source)
        return (bucket_id, repo.name, source), (
            difficulty.type if difficulty is not None else None)

    with ThreadPoolExecutor(max_workers=pool_size) as executor:
        feasibility = dict(executor.map(check_feasibility, structural_cells))
    print(f"structural feasibility: {sum(value is not None for value in feasibility.values())}/"
          f"{len(feasibility)} cells satisfiable", flush=True)
    expected_keys = {f"{bucket}|{repo.name}|{source}|{seed}"
                     for bucket, repo, source, seed in all_work}
    prior: list[dict] = []
    if enumeration_path.exists():
        prior = [json.loads(line) for line in enumeration_path.read_text().splitlines()
                 if line.strip()]
        prior_keys = [row["key"] for row in prior]
        if len(prior_keys) != len(set(prior_keys)) or not set(prior_keys) <= expected_keys:
            raise ValueError("pool enumeration checkpoint is incompatible")
        tasks.extend({name: row[name] for name in
                      ("bucket", "repo", "source", "seed", "difficulty_type")}
                     for row in prior if row["status"] == "admitted")
        done = set(prior_keys)
        work = [item for item in all_work
                if f"{item[0]}|{item[1].name}|{item[2]}|{item[3]}" not in done]
        print(f"resuming admission: {len(prior)}/{len(all_work)} triples complete", flush=True)
    else:
        work = all_work

    def inspect(item: tuple[str, Path, str, int], sandbox_pool) -> dict:
        bucket_id, repo, source, seed = item
        key = f"{bucket_id}|{repo.name}|{source}|{seed}"
        difficulty_type = feasibility[(bucket_id, repo.name, source)]
        if difficulty_type is None:
            return {"key": key, "bucket": bucket_id, "repo": repo.name,
                    "source": source, "seed": seed, "status": "unsatisfiable"}
        bucket = wanted[bucket_id]
        difficulty = Difficulty(source, difficulty_type, bucket.count, bucket.hint, bucket.span)
        dest = workspace / key.replace("|", "-")
        try:
            instance = package(repo, difficulty, seed, dest)
            admission = admit(instance, sandbox_pool)
            return {
                "key": key, "bucket": bucket_id, "repo": repo.name,
                "source": source, "seed": seed,
                "difficulty_type": difficulty.type,
                "status": "admitted" if admission["admitted"] else "rejected",
                "reasons": admission["reasons"],
            }
        finally:
            shutil.rmtree(dest, ignore_errors=True)

    try:
        with create_pool(IMAGE, size=pool_size) as sandbox_pool:
            with ThreadPoolExecutor(max_workers=pool_size) as executor:
                results = executor.map(lambda item: inspect(item, sandbox_pool), work)
                with enumeration_path.open("a") as audit:
                    for seen, row in enumerate(results, len(prior) + 1):
                        audit.write(json.dumps(row, sort_keys=True) + "\n")
                        audit.flush()
                        if row["status"] == "admitted":
                            tasks.append({name: row[name] for name in
                                          ("bucket", "repo", "source", "seed", "difficulty_type")})
                        if seen % 100 == 0:
                            print(f"admission {seen}/{len(all_work)}; admitted={len(tasks)}", flush=True)
    finally:
        shutil.rmtree(workspace, ignore_errors=True)

    counts = Counter(task["bucket"] for task in tasks)
    atomic_json(out_dir / "pool_manifest.json", tasks)
    print("pool sizes:", dict(counts), flush=True)
    too_small = {bucket: counts[bucket] for bucket in BUCKET_PLAN if counts[bucket] < 10}
    if too_small:
        raise RuntimeError(f"admitted pool too small; stopping before evaluation: {too_small}")
    return tasks


def load_and_validate_manifest(path: Path) -> list[dict]:
    tasks = json.loads(path.read_text())
    required = {"bucket", "repo", "source", "seed", "difficulty_type"}
    if not isinstance(tasks, list) or any(set(task) != required for task in tasks):
        raise ValueError("pool manifest has an incompatible schema")
    counts = Counter(task["bucket"] for task in tasks)
    missing = {bucket: counts[bucket] for bucket in BUCKET_PLAN if counts[bucket] < 10}
    if missing:
        raise RuntimeError(f"admitted pool too small: {missing}")
    return tasks


def choose_task(tasks: list[dict], bucket_id: str, group_index: int) -> dict:
    candidates = [task for task in tasks if task["bucket"] == bucket_id]
    rng = random.Random(stable_seed("d1-task", bucket_id, group_index))
    return candidates[rng.randrange(len(candidates))]


def predictions(history: list[int]) -> dict[str, float | None]:
    if not history:
        return {"B1_pred": None, "B_emp_pred": None, "B2_pred": None}
    return {
        "B1_pred": est_plugin(history),
        "B_emp_pred": est_empirical(history),
        "B2_pred": est_betabinom(history),
    }


def annotate_rollout(index: int, detail: dict) -> dict:
    """Make infrastructure completion explicit for every zero/one outcome."""
    row = dict(detail)
    termination = row.get("termination")
    seeds = row.get("sampling_seeds", [])
    turn_count = row.get("turn_count", 0)
    seed_trace_complete = len(seeds) == turn_count
    completed = termination in {"stop", "max_steps"} and seed_trace_complete
    row.update({
        "rollout_index": index,
        "completed": completed,
        "seed_trace_complete": seed_trace_complete,
        "failure_kind": None if completed else (
            "api_sandbox_or_grader_exception" if termination == "api_error"
            else "incomplete_sampling_trace"),
    })
    return row


def append_group(path: Path, row: dict) -> None:
    with path.open("a") as handle:
        handle.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def read_groups(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    keys = [(row["bucket_id"], row["group_index"]) for row in rows]
    if len(keys) != len(set(keys)):
        raise ValueError("duplicate group records in checkpoint")
    return rows


def task_spec(task: dict) -> dict:
    repo_path = next(path for path in repos() if path.name == task["repo"])
    bucket = next(bucket for bucket in BUCKETS if bucket.id == task["bucket"])
    difficulty = {
        "source": task["source"], "type": task["difficulty_type"],
        "count": bucket.count, "hint": bucket.hint, "span": bucket.span,
    }
    return {
        "repo": str(repo_path), "repo_name": task["repo"],
        "bucket": task["bucket"], "injector_seed": task["seed"],
        "difficulty": difficulty,
    }


def group_file(out_dir: Path, bucket_id: str) -> Path:
    return out_dir / f"prequential_groups_{bucket_id}.jsonl"


def load_all_groups(out_dir: Path, plan: dict[str, int] | None = None) -> list[dict]:
    plan = plan or BUCKET_PLAN
    rows = []
    for bucket_id in plan:
        rows.extend(read_groups(group_file(out_dir, bucket_id)))
    return rows


def run_groups(args, tasks: list[dict], out_dir: Path,
               plan: dict[str, int] | None = None) -> list[dict]:
    from openai import OpenAI

    plan = plan or BUCKET_PLAN
    all_rows: list[dict] = []
    workspace = Path(tempfile.mkdtemp(prefix="octorl-d1-eval-"))
    client = OpenAI(base_url=args.base_url, api_key=args.api_key)
    print(f"[run_groups] buckets={list(plan.keys())} concurrency={args.concurrency}", flush=True)
    try:
        with create_pool(IMAGE, size=args.concurrency) as sandbox_pool:
            print(f"[run_groups] sandbox pool ready", flush=True)
            for bucket_id, target in plan.items():
                gpath = group_file(out_dir, bucket_id)
                rows = read_groups(gpath)
                completed = {(row["bucket_id"], row["group_index"]): row for row in rows}
                history: list[int] = []
                for group_index in range(1, target + 1):
                    key = (bucket_id, group_index)
                    expected_task = choose_task(tasks, bucket_id, group_index)
                    if key in completed:
                        row = completed[key]
                        if row["task"] != expected_task:
                            raise ValueError(f"checkpoint task mismatch at {key}")
                        if row["predictions"] != predictions(history):
                            raise ValueError(f"checkpoint prequential mismatch at {key}")
                        history.append(row["K"])
                        all_rows.append(row)
                        continue

                    if group_index == len(completed) + 1:
                        print(f"[{bucket_id}] checkpoint OK ({len(completed)} groups), "
                              f"resuming at group {group_index}/{target}", flush=True)

                    pred = predictions(history)
                    group_master_seed = stable_seed(
                        SAMPLING_MASTER_SEED, bucket_id, group_index) % (2**31)
                    print(f"[{bucket_id}] evaluating group {group_index}/{target} "
                          f"repo={expected_task['repo']}...", flush=True)
                    started = time.monotonic()
                    raw_details = evaluate_instance(
                        client, args.model, task_spec(expected_task), sandbox_pool,
                        workspace / f"{bucket_id}-g{group_index}", G,
                        concurrency=args.concurrency,
                        sampling_master_seed=group_master_seed,
                        top_k=args.top_k, top_p=args.top_p,
                    )
                    details = [annotate_rollout(index, detail)
                               for index, detail in enumerate(raw_details)]
                    rewards = [float(detail["reward"]) for detail in details]
                    if len(rewards) != G or any(reward not in (0.0, 1.0) for reward in rewards):
                        raise RuntimeError(f"invalid rollout rewards at {key}: {rewards}")
                    row = {
                        "bucket_id": bucket_id,
                        "group_index": group_index,
                        "task": expected_task,
                        "group_sampling_master_seed": group_master_seed,
                        "predictions": pred,
                        "K": int(sum(rewards)),
                        "is_mixed": 0 < sum(rewards) < G,
                        "rollouts": details,
                        "infrastructure_ok": all(detail["completed"] for detail in details),
                        "wall_time_s": round(time.monotonic() - started, 3),
                    }
                    append_group(gpath, row)
                    all_rows.append(row)
                    history.append(row["K"])
                    print(f"{bucket_id} {group_index}/{target} K={row['K']} "
                          f"infra_ok={row['infrastructure_ok']}", flush=True)
    finally:
        shutil.rmtree(workspace, ignore_errors=True)
    return all_rows


def gap(rows: list[dict]) -> float:
    mu = sum(row["K"] for row in rows) / (G * len(rows))
    return float(f(mu) - np.mean([row["is_mixed"] for row in rows]))


def cluster_bootstrap(rows: list[dict], bucket_id: str, B: int = 10_000) -> tuple[float, float]:
    by_repo: dict[str, list[dict]] = {}
    for row in rows:
        by_repo.setdefault(row["task"]["repo"], []).append(row)
    repo_names = sorted(by_repo)
    rng = np.random.default_rng(stable_seed(BOOTSTRAP_SEED, bucket_id))
    values = np.empty(B)
    for index in range(B):
        sampled = rng.choice(repo_names, size=len(repo_names), replace=True)
        resample = [row for name in sampled for row in by_repo[str(name)]]
        values[index] = gap(resample)
    return tuple(float(x) for x in np.percentile(values, [2.5, 97.5]))


def summarize(rows: list[dict], tasks: list[dict]) -> list[dict]:
    output = []
    for bucket_id, target in BUCKET_PLAN.items():
        bucket_rows = sorted(
            (row for row in rows if row["bucket_id"] == bucket_id),
            key=lambda row: row["group_index"])
        if len(bucket_rows) != target:
            raise RuntimeError(f"{bucket_id}: expected {target} groups, found {len(bucket_rows)}")
        infra_bad = sum(not row["infrastructure_ok"] for row in bucket_rows)
        if infra_bad:
            print(f"WARNING: {bucket_id}: {infra_bad}/{target} groups have infrastructure failures "
                  f"({infra_bad / target:.1%}); included as-is (prequential, non-replayable)", flush=True)
        mu = sum(row["K"] for row in bucket_rows) / (G * target)
        f_mu = float(f(mu))
        q_emp = float(np.mean([row["is_mixed"] for row in bucket_rows]))
        gap_hat = f_mu - q_emp
        lo, hi = cluster_bootstrap(bucket_rows, bucket_id)
        output.append({
            "bucket_id": bucket_id, "T": target,
            "n_pool": sum(task["bucket"] == bucket_id for task in tasks),
            "n_repos_observed": len({row["task"]["repo"] for row in bucket_rows}),
            "mu_hat": mu, "f_mu_hat": f_mu, "q_emp": q_emp,
            "gap_hat": gap_hat, "bootstrap_ci_lo": lo,
            "bootstrap_ci_hi": hi,
            "a_bar_pass": bool(bucket_id in HARD_BUCKETS and gap_hat >= 0.05 and lo > 0),
            "infra_failures": infra_bad,
        })
    return output


def make_figure(rows: list[dict], out_path: Path) -> None:
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(13, 9), sharey=True)
    for axis, bucket_id in zip(axes.flat, BUCKET_PLAN):
        bucket_rows = sorted(
            (row for row in rows if row["bucket_id"] == bucket_id),
            key=lambda row: row["group_index"])
        x = np.arange(1, len(bucket_rows) + 1)
        for key, label in (("B1_pred", "B1"), ("B_emp_pred", "B_emp"), ("B2_pred", "B2")):
            y = [math.nan if row["predictions"][key] is None else row["predictions"][key]
                 for row in bucket_rows]
            axis.plot(x, y, label=label, linewidth=1.5)
        measured = np.cumsum([row["is_mixed"] for row in bucket_rows]) / x
        axis.plot(x, measured, label="measured mixed rate", color="black", linewidth=2)
        axis.set_title(bucket_id)
        axis.set_xlabel("group")
        axis.set_ylabel("probability")
        axis.grid(alpha=0.25)
        axis.legend(fontsize=8)
    fig.suptitle("D1: prequential aggregation predictions vs cumulative measured mixed rate")
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def write_run_metadata(args, out_dir: Path, history_replacements: dict[str, str]) -> None:
    git_env = dict(os.environ)
    git_env.pop("GIT_DIR", None)
    metadata = {
        "git_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True, env=git_env).strip(),
        "code_sha256": {
            name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
            for name in ("scripts/p2_d1_gap.py", "scripts/baseline_eval.py")
        },
        "catalog_history_replacements": history_replacements,
        "bucket_plan": BUCKET_PLAN,
        "pool_sources": list(SOURCES),
        "pool_seeds": [min(POOL_SEEDS), max(POOL_SEEDS)],
        "G": G, "bootstrap_resamples": 10_000,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "sampling_master_seed": SAMPLING_MASTER_SEED,
        "model": args.model, "base_url": args.base_url,
        "top_k": args.top_k, "top_p": args.top_p,
    }
    path = out_dir / "run_metadata.json"
    skip = {"concurrency", "code_sha256", "git_commit"}
    if path.exists():
        existing = {k: v for k, v in json.loads(path.read_text()).items() if k not in skip}
        proposed = {k: v for k, v in metadata.items() if k not in skip}
        if existing != proposed:
            raise ValueError("existing run metadata is incompatible")
    else:
        atomic_json(path, metadata)


def finalize(rows: list[dict], tasks: list[dict], out_dir: Path) -> int:
    summary = summarize(rows, tasks)
    atomic_json(out_dir / "d1_summary.json", summary)
    make_figure(rows, out_dir / "d1_headline.png")
    overall = any(row["a_bar_pass"] for row in summary if row["bucket_id"] in HARD_BUCKETS)
    atomic_json(out_dir / "a_bar_verdict.json", {
        "hard_buckets": [{"bucket_id": row["bucket_id"], "pass": row["a_bar_pass"]}
                         for row in summary if row["bucket_id"] in HARD_BUCKETS],
        "overall_a_bar_pass": overall,
    })
    print(f"A-bar overall: {'PASS' if overall else 'FAIL'}", flush=True)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default=None)
    parser.add_argument("--api-key", default="EMPTY")
    parser.add_argument("--model", default=None)
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--reuse-pool-manifest", action="store_true",
                        help="resume using the already-frozen admitted pool")
    parser.add_argument("--only-bucket", default=None,
                        help="run only this bucket (for parallel execution)")
    parser.add_argument("--summarize-only", action="store_true",
                        help="skip evaluation; merge per-bucket group files and produce summary/figure/verdict")
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[main] starting (only_bucket={args.only_bucket}, "
          f"summarize_only={args.summarize_only})", flush=True)

    if args.summarize_only:
        tasks = load_and_validate_manifest(args.out_dir / "pool_manifest.json")
        rows = load_all_groups(args.out_dir)
        return finalize(rows, tasks, args.out_dir)

    if not args.base_url or not args.model:
        parser.error("--base-url and --model are required for evaluation")

    history_replacements = prepare_catalog_history(args.out_dir)
    print("[main] catalog history ready", flush=True)
    write_run_metadata(args, args.out_dir, history_replacements)
    print("[main] metadata validated", flush=True)

    manifest_path = args.out_dir / "pool_manifest.json"
    if args.reuse_pool_manifest:
        tasks = load_and_validate_manifest(manifest_path)
    else:
        if manifest_path.exists():
            raise FileExistsError("pool manifest exists; pass --reuse-pool-manifest to resume")
        tasks = build_pool_manifest(args.out_dir, args.concurrency)

    plan = BUCKET_PLAN
    if args.only_bucket:
        if args.only_bucket not in BUCKET_PLAN:
            parser.error(f"unknown bucket: {args.only_bucket}")
        plan = {args.only_bucket: BUCKET_PLAN[args.only_bucket]}

    rows = run_groups(args, tasks, args.out_dir, plan)
    if not args.only_bucket:
        return finalize(rows, tasks, args.out_dir)
    print(f"bucket {args.only_bucket} complete ({len(rows)} groups)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
