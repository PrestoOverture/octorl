"""Baseline evaluation for the OctoRL environment package (P1.23).

Two modes:

  --base-url/--model   evaluate a served model through the Verifiers interface
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
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.environment.harness import admission_report, admit, package
from src.environment.sandbox import create_pool
from src.injector.buckets import BUCKETS
from src.injector.catalog import inject_from_catalog
from src.injector.core import BUG_TYPES, Difficulty

ROOT = Path(__file__).resolve().parents[1]
IMAGE = (ROOT / "artifacts/p1/docker-image.id").read_text().strip()
# All three sources generate at count=1 since the F2 repair. `commit-rollback`
# has no multi-edit entries yet, so a baseline over count>=2 must pass --source.
DEFAULT_SOURCE = "mutation"


def satisfiable(repo: Path, bucket, seed: int, source: str) -> Difficulty | None:
    for bug_type in BUG_TYPES:
        difficulty = Difficulty(source, bug_type, bucket.count, bucket.hint, bucket.span)
        try:
            inject_from_catalog(repo, difficulty, seed)
            return difficulty
        except Exception:
            continue
    return None


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repos", nargs="+", default=None, help="repo names (default: all train repos)")
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
    if args.repos:
        repo_dirs = [p for p in repo_dirs if p.name in set(args.repos)]
    wanted = {b.id: b for b in BUCKETS}
    buckets = [wanted[name] for name in args.buckets]

    workspace = Path(tempfile.mkdtemp(prefix="octorl-baseline-"))
    specs, rows = [], []
    with create_pool(IMAGE, size=args.concurrency) as pool:
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
            from src.adapters.verifiers_env import load_environment
            env = load_environment(pool, [{k: v for k, v in s.items() if k in
                                           ("repo", "difficulty", "injector_seed")} for s in specs],
                                   workspace=workspace / "eval")
            client = OpenAI(base_url=args.base_url, api_key=args.api_key)
            results = env.evaluate(client, args.model, rollouts_per_example=args.rollouts,
                                   max_concurrent=args.concurrency)
            per_bucket = defaultdict(list)
            for spec, reward in zip([s for s in specs for _ in range(args.rollouts)], results.reward):
                per_bucket[(spec["repo_name"], spec["bucket"])].append(reward)
            report["model"] = {"base_url": args.base_url, "model": args.model,
                               "rollouts_per_instance": args.rollouts}
            report["pass_at_1"] = {f"{repo}|{bucket}": {
                "rollouts": len(values), "mean_reward": statistics.mean(values),
                "mixed_group": 0 < sum(values) < len(values)}
                for (repo, bucket), values in sorted(per_bucket.items())}
            report["overall_mean_reward"] = statistics.mean(results.reward) if results.reward else 0.0
        else:
            print("\nNo --base-url/--model: reporting the admitted census only. "
                  "A pass@1 baseline needs a served model.")

    Path(args.out).write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    census = report["admission"]
    print(f"\nadmitted {census['admitted']}/{census['requested']}  "
          f"(rejected {census['rejected']})")
    print("rejections:", dict(Counter(r for row in rows for r in row["reasons"])) or "none")
    if "overall_mean_reward" in report:
        print(f"overall mean reward: {report['overall_mean_reward']:.3f}")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
