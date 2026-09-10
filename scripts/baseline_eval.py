"""Baseline evaluation for the OctoRL environment package (P1.23/P1.24).

Two modes:

  --base-url/--model   evaluate a served model via an OpenAI-compatible endpoint
                       (pass@1 per bucket, G rollouts per instance)
  (default)            no model: report the admitted census only

Admission runs first in both modes, so a baseline is never computed over
instances whose defect no test detects (see artifacts/p1/findings.md, F1).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import struct
import subprocess
import sys
import tempfile
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.environment.harness import admission_report, admit, package, prepare_grading
from src.environment.sandbox import create_pool
from src.environment.tools import ToolSession
from src.environment.verifier import verify
from src.injector.buckets import BUCKETS
from src.injector.catalog import inject_from_catalog
from src.injector.core import BUG_TYPES, Difficulty

ROOT = Path(__file__).resolve().parents[1]
_image_file = ROOT / "artifacts/p1/docker-image.id"
IMAGE = _image_file.read_text().strip() if _image_file.exists() else None
DEFAULT_SOURCE = "mutation"

SYSTEM_PROMPT = (ROOT / "harness/prompt.md").read_text().strip()

TOOL_SCHEMAS = [
    {"type": "function", "function": {
        "name": "list_files",
        "description": "List repository files under a directory.",
        "parameters": {"type": "object", "properties": {
            "path": {"type": "string", "description": "Directory path (default: '.')"}
        }}}},
    {"type": "function", "function": {
        "name": "search_code",
        "description": "Search source files by regular expression pattern.",
        "parameters": {"type": "object", "properties": {
            "pattern": {"type": "string", "description": "Regular expression to search for"}
        }, "required": ["pattern"]}}},
    {"type": "function", "function": {
        "name": "read_file",
        "description": "Read a file, optionally a specific line range.",
        "parameters": {"type": "object", "properties": {
            "path": {"type": "string", "description": "File path relative to repository root"},
            "range": {"type": "array", "items": {"type": "integer"},
                      "description": "Inclusive 1-based [start, end] line range"}
        }, "required": ["path"]}}},
    {"type": "function", "function": {
        "name": "apply_patch",
        "description": "Apply a unified diff to source files. Tests are not writable.",
        "parameters": {"type": "object", "properties": {
            "diff": {"type": "string", "description": "Unified diff content"}
        }, "required": ["diff"]}}},
    {"type": "function", "function": {
        "name": "run_tests",
        "description": "Run visible tests. Defaults to tests affected by your edits.",
        "parameters": {"type": "object", "properties": {
            "subset": {"type": "array", "items": {"type": "string"},
                       "description": "Specific test files or ids to run"}
        }}}},
]


def satisfiable(repo: Path, bucket, seed: int, source: str) -> Difficulty | None:
    for bug_type in BUG_TYPES:
        difficulty = Difficulty(source, bug_type, bucket.count, bucket.hint, bucket.span)
        try:
            inject_from_catalog(repo, difficulty, seed)
            return difficulty
        except Exception:
            continue
    return None


def assistant_turn_format(msg) -> dict[str, int]:
    """Classify one model response for the SFT tool-format gate."""
    tool_calls = msg.tool_calls or []
    valid_json = bool(tool_calls)
    for tc in tool_calls:
        try:
            json.loads(tc.function.arguments)
        except (json.JSONDecodeError, TypeError):
            valid_json = False
    return {
        "total_turns": 1,
        "tool_call_turns": int(bool(tool_calls)),
        "valid_json_turns": int(valid_json),
    }


def derive_seed(master: int, task_key: str, rollout_index: int,
                turn_index: int) -> int:
    """Derive a stable per-request seed without Python's salted hash()."""
    payload = f"{master}|{task_key}|{rollout_index}|{turn_index}".encode()
    h = hashlib.sha256(payload)
    return struct.unpack(">I", h.digest()[:4])[0] % (2**31)


def request_params(top_k: int, top_p: float) -> dict:
    return {"temperature": 1.0, "top_k": top_k, "top_p": top_p}


def build_request_args(model: str, messages: list[dict], *, top_k: int,
                       top_p: float, sampling_master_seed: int | None,
                       task_key: str, rollout_index: int,
                       turn_index: int) -> tuple[dict, int | None]:
    """Build an explicit OpenAI request, omitting seed for legacy unseeded runs."""
    api_args = {
        "model": model,
        "messages": messages,
        "tools": TOOL_SCHEMAS,
        "temperature": 1.0,
        "top_p": top_p,
        "extra_body": {"top_k": top_k},
    }
    turn_seed = None
    if sampling_master_seed is not None:
        turn_seed = derive_seed(sampling_master_seed, task_key,
                                rollout_index, turn_index)
        api_args["seed"] = turn_seed
    return api_args, turn_seed


def build_run_metadata(args) -> dict:
    git_commit = subprocess.check_output(
        ["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, text=True
    ).strip()
    injector_seed: int | list[int]
    injector_seed = args.seeds[0] if len(args.seeds) == 1 else args.seeds
    return {
        "git_commit": git_commit,
        "model_path": args.model,
        "base_url": args.base_url,
        "request_params": request_params(args.top_k, args.top_p),
        "sampling_master_seed": args.sampling_seed,
        "injector_seed": injector_seed,
        "rollouts_per_instance": args.rollouts,
        "enforce_eager": args.enforce_eager,
    }


def load_checkpoint(path: Path, metadata: dict) -> dict[str, dict]:
    """Load only checkpoint rows proven compatible with this run."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        path.write_text(json.dumps({"run_metadata": metadata}) + "\n")
        return {}

    records = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    if not records or "run_metadata" not in records[0]:
        print("WARNING: deprecated checkpoint without run_metadata ignored", file=sys.stderr)
        path.write_text(json.dumps({"run_metadata": metadata}) + "\n")
        return {}

    stored = records[0]["run_metadata"]
    compared = ("model_path", "request_params", "enforce_eager")
    mismatches = [name for name in compared if stored.get(name) != metadata.get(name)]
    if mismatches:
        raise ValueError("checkpoint run_metadata mismatch: " + ", ".join(mismatches))

    done = {}
    for record in records[1:]:
        if "key" not in record or "rollout_details" not in record:
            print("WARNING: deprecated checkpoint record ignored", file=sys.stderr)
            continue
        done[record["key"]] = record
    return done


def run_rollout(client, model: str, sandbox, instance,
                session: ToolSession, *, task_key: str, rollout_index: int,
                sampling_master_seed: int | None, top_k: int,
                top_p: float) -> dict:
    """Run one agent rollout and return its auditable result details."""
    issue = instance.prompt_instance()
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": json.dumps(issue, sort_keys=True)},
    ]
    format_counts = {
        "total_turns": 0,
        "tool_call_turns": 0,
        "valid_json_turns": 0,
    }
    sampling_seeds: list[int] = []
    termination = "max_steps"

    for turn_index in range(ToolSession.MAX_STEPS):
        api_args, turn_seed = build_request_args(
            model, messages, top_k=top_k, top_p=top_p,
            sampling_master_seed=sampling_master_seed, task_key=task_key,
            rollout_index=rollout_index, turn_index=turn_index)
        if turn_seed is not None:
            sampling_seeds.append(turn_seed)
        try:
            response = client.chat.completions.create(**api_args)
        except Exception as exc:
            print(f"    API error: {exc}", flush=True)
            termination = "api_error"
            break

        choice = response.choices[0]
        msg = choice.message
        turn_counts = assistant_turn_format(msg)
        for name, count in turn_counts.items():
            format_counts[name] += count

        if not msg.tool_calls:
            messages.append({"role": "assistant", "content": msg.content or ""})
            termination = "stop"
            break

        assistant_msg = {"role": "assistant", "content": msg.content or None, "tool_calls": [
            {"id": tc.id, "type": "function",
             "function": {"name": tc.function.name, "arguments": tc.function.arguments}}
            for tc in msg.tool_calls
        ]}
        messages.append(assistant_msg)

        for tc in msg.tool_calls:
            try:
                args = json.loads(tc.function.arguments)
            except (json.JSONDecodeError, TypeError):
                args = {}
            result = session.call(tc.function.name, args)
            output = result.stdout if result.exit_code == 0 else f"[error] {result.stderr}"
            messages.append({"role": "tool", "tool_call_id": tc.id, "content": output[:8000]})

        if session.steps >= ToolSession.MAX_STEPS:
            termination = "max_steps"
            break

        if choice.finish_reason == "stop":
            termination = "stop"
            break

    patched = {name: (sandbox.root / name).read_text(errors="replace")
               for name in sorted(session.changed) if (sandbox.root / name).is_file()}
    outcome = verify(sandbox, hidden_tests_dir=instance.hidden_tests_dir,
                     originals=instance.originals, patched=patched,
                     tool_cheat_flags=session.cheat_flags,
                     seed=instance.verification_seed,
                     expected_tests=instance.expected_tests)
    return {
        "reward": float(outcome.reward),
        "format_validity": format_counts,
        "turn_count": format_counts["total_turns"],
        "termination": termination,
        "changed_files": sorted(session.changed),
        "sampling_seeds": sampling_seeds,
    }


def _run_single_rollout(client, model: str, spec: dict, pool, workspace: Path,
                        g: int, sampling_master_seed: int | None,
                        top_k: int, top_p: float) -> tuple[int, dict]:
    """Run one rollout and return its index and auditable details."""
    dest = workspace / f"{spec['repo_name']}-{spec['bucket']}-{spec['injector_seed']}-r{g}"
    instance = package(Path(spec["repo"]), Difficulty(**spec["difficulty"]),
                       spec["injector_seed"], dest)
    lease = pool.lease(instance.root)
    sandbox = lease.__enter__()
    try:
        prepare_grading(instance, sandbox)
        session = ToolSession(sandbox, seed=instance.verification_seed,
                              affected_tests=instance.affected_tests)
        task_key = f"{spec['repo_name']}|{spec['bucket']}|{spec['injector_seed']}"
        details = run_rollout(
            client, model, sandbox, instance, session, task_key=task_key,
            rollout_index=g, sampling_master_seed=sampling_master_seed,
            top_k=top_k, top_p=top_p)
        return g, details
    except Exception as exc:
        print(f"    rollout {g} error: {exc}", flush=True)
        return g, {
            "reward": 0.0,
            "format_validity": {"total_turns": 0, "tool_call_turns": 0,
                                "valid_json_turns": 0},
            "turn_count": 0,
            "termination": "api_error",
            "changed_files": [],
            "sampling_seeds": [],
        }
    finally:
        lease.__exit__(None, None, None)


def evaluate_instance(client, model: str, spec: dict, pool, workspace: Path,
                      rollouts: int, concurrency: int = 1,
                      sampling_master_seed: int | None = None,
                      top_k: int = 20, top_p: float = 0.95) -> list[dict]:
    """Run G rollouts for one admitted instance and return indexed details."""
    details = [dict() for _ in range(rollouts)]
    workers = min(concurrency, rollouts)
    if workers <= 1:
        for g in range(rollouts):
            _, result = _run_single_rollout(
                client, model, spec, pool, workspace, g,
                sampling_master_seed, top_k, top_p)
            details[g] = result
        return details
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(
            _run_single_rollout, client, model, spec, pool, workspace, g,
            sampling_master_seed, top_k, top_p): g
                   for g in range(rollouts)}
        for future in as_completed(futures):
            g, result = future.result()
            details[g] = result
    return details


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repos", nargs="+", default=None,
                        help="repo names (default: all train + held-out repos)")
    parser.add_argument("--buckets", nargs="+", default=["c1-L0-single-function", "c2-L0-single-file"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[11])
    parser.add_argument("--source", default=DEFAULT_SOURCE,
                        choices=["mutation", "commit-rollback", "feat-add"])
    parser.add_argument("--base-url", default=None, help="OpenAI-compatible endpoint, e.g. vLLM")
    parser.add_argument("--api-key", default="EMPTY")
    parser.add_argument("--model", default=None)
    parser.add_argument("--rollouts", type=int, default=8, help="G rollouts per instance")
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--sampling-seed", type=int, default=None,
                        help="master seed for stable per-request sampling")
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--enforce-eager", action="store_true",
                        help="record that the external server uses enforce_eager")
    parser.add_argument("--out", default=str(ROOT / "artifacts/p1/baseline_eval.json"))
    args = parser.parse_args()

    repo_dirs = sorted(p for p in (ROOT / "repos/train").iterdir() if p.is_dir())
    held_out = ROOT / "repos/held_out"
    if held_out.is_dir():
        repo_dirs += sorted(p for p in held_out.iterdir() if p.is_dir())
    if args.repos:
        repo_dirs = [p for p in repo_dirs if p.name in set(args.repos)]
    wanted = {b.id: b for b in BUCKETS}
    buckets = [wanted[name] for name in args.buckets]

    workspace = Path(tempfile.mkdtemp(prefix="octorl-baseline-"))
    specs, rows = [], []
    with create_pool(IMAGE, size=args.concurrency) as pool:
        print("=== Admission ===", flush=True)
        for repo in repo_dirs:
            for bucket in buckets:
                for seed in args.seeds:
                    difficulty = satisfiable(repo, bucket, seed, args.source)
                    if difficulty is None:
                        print(f"  {repo.name:20} {bucket.id:24} no satisfiable task", flush=True)
                        continue
                    instance = package(repo, difficulty, seed, workspace / f"{repo.name}-{bucket.id}-{seed}")
                    row = admit(instance, pool)
                    rows.append(row)
                    status = "admitted" if row["admitted"] else "REJECTED " + ";".join(row["reasons"])[:40]
                    print(f"  {repo.name:20} {bucket.id:24} {difficulty.type:28} {status}", flush=True)
                    if row["admitted"]:
                        specs.append({"repo": str(repo),
                                      "difficulty": {"source": difficulty.source, "type": difficulty.type,
                                                     "count": difficulty.count, "hint": difficulty.hint,
                                                     "span": difficulty.span},
                                      "injector_seed": seed, "bucket": bucket.id, "repo_name": repo.name})

        report = {"admission": admission_report(rows), "specs": specs, "source": args.source}

        if args.base_url and args.model:
            from openai import OpenAI
            client = OpenAI(base_url=args.base_url, api_key=args.api_key)

            run_metadata = build_run_metadata(args)
            report["run_metadata"] = run_metadata
            ckpt_path = Path(args.out + ".ckpt")
            done = load_checkpoint(ckpt_path, run_metadata)
            if done:
                print(f"\n  Resuming from checkpoint: {len(done)}/{len(specs)} instances done", flush=True)

            print(f"\n=== Evaluation ({len(specs)} instances × {args.rollouts} rollouts) ===", flush=True)
            per_bucket: dict[tuple[str, str], list[float]] = defaultdict(list)
            all_rewards: list[float] = []
            instance_format_validity: dict[str, list[dict[str, int]]] = {}
            rollout_details: dict[str, list[dict]] = {}
            all_format_counts: list[dict[str, int]] = []
            eval_start = time.monotonic()

            for i, spec in enumerate(specs):
                key = f"{spec['repo_name']}|{spec['bucket']}|{spec['injector_seed']}"
                if key in done:
                    details = done[key]["rollout_details"]
                    rewards = [row["reward"] for row in details]
                    format_counts = [row["format_validity"] for row in details]
                    per_bucket[(spec["repo_name"], spec["bucket"])].extend(rewards)
                    all_rewards.extend(rewards)
                    instance_format_validity[key] = format_counts
                    rollout_details[key] = details
                    all_format_counts.extend(format_counts)
                    mean = statistics.mean(rewards) if rewards else 0.0
                    print(f"  [{i+1}/{len(specs)}] {spec['repo_name']:20} {spec['bucket']:24} "
                          f"mean={mean:.3f} (checkpoint)", flush=True)
                    continue
                print(f"  [{i+1}/{len(specs)}] {spec['repo_name']:20} {spec['bucket']:24} ", end="", flush=True)
                details = evaluate_instance(
                    client, args.model, spec, pool, workspace, args.rollouts,
                    concurrency=args.concurrency,
                    sampling_master_seed=args.sampling_seed,
                    top_k=args.top_k, top_p=args.top_p)
                rewards = [row["reward"] for row in details]
                format_counts = [row["format_validity"] for row in details]
                per_bucket[(spec["repo_name"], spec["bucket"])].extend(rewards)
                all_rewards.extend(rewards)
                instance_format_validity[key] = format_counts
                rollout_details[key] = details
                all_format_counts.extend(format_counts)
                mean = statistics.mean(rewards) if rewards else 0.0
                print(f"mean={mean:.3f} ({sum(r > 0 for r in rewards)}/{len(rewards)} pass)", flush=True)
                with open(ckpt_path, "a") as f:
                    f.write(json.dumps({"key": key, "rollout_details": details}) + "\n")

            elapsed = time.monotonic() - eval_start
            report["model"] = {"base_url": args.base_url, "model": args.model,
                               "rollouts_per_instance": args.rollouts}
            report["pass_at_1"] = {f"{repo}|{bucket}": {
                "rollouts": len(values), "mean_reward": statistics.mean(values),
                "mixed_group": 0 < sum(values) < len(values)}
                for (repo, bucket), values in sorted(per_bucket.items())}
            report["overall_mean_reward"] = statistics.mean(all_rewards) if all_rewards else 0.0
            valid_turns = sum(row["valid_json_turns"] for row in all_format_counts)
            total_turns = sum(row["total_turns"] for row in all_format_counts)
            report["instance_format_validity"] = instance_format_validity
            report["rollout_details"] = rollout_details
            report["format_validity"] = {
                "valid_turns": valid_turns,
                "total_turns": total_turns,
                "rate": valid_turns / total_turns if total_turns else 0.0,
            }
            report["eval_wall_time_s"] = round(elapsed, 1)
            report["total_rollouts"] = len(all_rewards)
            if ckpt_path.exists():
                ckpt_path.unlink()
        else:
            print("\nNo --base-url/--model: reporting the admitted census only. "
                  "A pass@1 baseline needs a served model.")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    census = report["admission"]
    print(f"\nadmitted {census['admitted']}/{census['requested']}  "
          f"(rejected {census['rejected']})")
    print("rejections:", dict(Counter(r for row in rows for r in row["reasons"])) or "none")
    if "overall_mean_reward" in report:
        print(f"overall mean reward: {report['overall_mean_reward']:.3f}")
        print(f"eval wall time: {report['eval_wall_time_s']:.0f}s  "
              f"({report['total_rollouts']} rollouts)")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
