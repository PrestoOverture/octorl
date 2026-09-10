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
import json
import statistics
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


def run_rollout(client, model: str, sandbox, instance, session: ToolSession) -> float:
    """Run one agent rollout. Returns reward (0.0 or 1.0)."""
    issue = instance.prompt_instance()
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": json.dumps(issue, sort_keys=True)},
    ]

    for _step in range(ToolSession.MAX_STEPS):
        try:
            response = client.chat.completions.create(
                model=model, messages=messages, tools=TOOL_SCHEMAS,
                temperature=1.0,
            )
        except Exception as exc:
            print(f"    API error: {exc}", flush=True)
            break

        choice = response.choices[0]
        msg = choice.message

        if not msg.tool_calls:
            messages.append({"role": "assistant", "content": msg.content or ""})
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
            break

        if choice.finish_reason == "stop":
            break

    patched = {name: (sandbox.root / name).read_text(errors="replace")
               for name in sorted(session.changed) if (sandbox.root / name).is_file()}
    outcome = verify(sandbox, hidden_tests_dir=instance.hidden_tests_dir,
                     originals=instance.originals, patched=patched,
                     tool_cheat_flags=session.cheat_flags,
                     seed=instance.verification_seed,
                     expected_tests=instance.expected_tests)
    return float(outcome.reward)


def _run_single_rollout(client, model: str, spec: dict, pool, workspace: Path,
                        g: int) -> tuple[int, float]:
    """Run one rollout, return (index, reward)."""
    dest = workspace / f"{spec['repo_name']}-{spec['bucket']}-{spec['injector_seed']}-r{g}"
    instance = package(Path(spec["repo"]), Difficulty(**spec["difficulty"]),
                       spec["injector_seed"], dest)
    lease = pool.lease(instance.root)
    sandbox = lease.__enter__()
    try:
        prepare_grading(instance, sandbox)
        session = ToolSession(sandbox, seed=instance.verification_seed,
                              affected_tests=instance.affected_tests)
        reward = run_rollout(client, model, sandbox, instance, session)
        return g, reward
    except Exception as exc:
        print(f"    rollout {g} error: {exc}", flush=True)
        return g, 0.0
    finally:
        lease.__exit__(None, None, None)


def evaluate_instance(client, model: str, spec: dict, pool, workspace: Path,
                      rollouts: int, concurrency: int = 1) -> list[float]:
    """Run G rollouts for one admitted instance. Returns list of rewards."""
    rewards = [0.0] * rollouts
    workers = min(concurrency, rollouts)
    if workers <= 1:
        for g in range(rollouts):
            _, reward = _run_single_rollout(client, model, spec, pool, workspace, g)
            rewards[g] = reward
        return rewards
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(_run_single_rollout, client, model, spec, pool, workspace, g): g
                   for g in range(rollouts)}
        for future in as_completed(futures):
            g, reward = future.result()
            rewards[g] = reward
    return rewards


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

            ckpt_path = Path(args.out + ".ckpt")
            done: dict[str, list[float]] = {}
            if ckpt_path.exists():
                for line in ckpt_path.read_text().splitlines():
                    rec = json.loads(line)
                    done[rec["key"]] = rec["rewards"]
                print(f"\n  Resuming from checkpoint: {len(done)}/{len(specs)} instances done", flush=True)

            print(f"\n=== Evaluation ({len(specs)} instances × {args.rollouts} rollouts) ===", flush=True)
            per_bucket: dict[tuple[str, str], list[float]] = defaultdict(list)
            all_rewards: list[float] = []
            eval_start = time.monotonic()

            for i, spec in enumerate(specs):
                key = f"{spec['repo_name']}|{spec['bucket']}|{spec['injector_seed']}"
                if key in done:
                    rewards = done[key]
                    per_bucket[(spec["repo_name"], spec["bucket"])].extend(rewards)
                    all_rewards.extend(rewards)
                    mean = statistics.mean(rewards) if rewards else 0.0
                    print(f"  [{i+1}/{len(specs)}] {spec['repo_name']:20} {spec['bucket']:24} "
                          f"mean={mean:.3f} (checkpoint)", flush=True)
                    continue
                print(f"  [{i+1}/{len(specs)}] {spec['repo_name']:20} {spec['bucket']:24} ", end="", flush=True)
                rewards = evaluate_instance(client, args.model, spec, pool, workspace, args.rollouts,
                                            concurrency=args.concurrency)
                per_bucket[(spec["repo_name"], spec["bucket"])].extend(rewards)
                all_rewards.extend(rewards)
                mean = statistics.mean(rewards) if rewards else 0.0
                print(f"mean={mean:.3f} ({sum(r > 0 for r in rewards)}/{len(rewards)} pass)", flush=True)
                with open(ckpt_path, "a") as f:
                    f.write(json.dumps({"key": key, "rewards": rewards}) + "\n")

            elapsed = time.monotonic() - eval_start
            report["model"] = {"base_url": args.base_url, "model": args.model,
                               "rollouts_per_instance": args.rollouts}
            report["pass_at_1"] = {f"{repo}|{bucket}": {
                "rollouts": len(values), "mean_reward": statistics.mean(values),
                "mixed_group": 0 < sum(values) < len(values)}
                for (repo, bucket), values in sorted(per_bucket.items())}
            report["overall_mean_reward"] = statistics.mean(all_rewards) if all_rewards else 0.0
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
