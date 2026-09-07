"""Re-audit of F1 after the grading/admission repair.

Runs the admission check over the same (repo x bug type) cells the original
reward-signal audit covered, and re-tests the question that mattered: can a
no-op policy still earn reward 1 on anything that survives admission?
"""
from __future__ import annotations

import json
import shutil
import sys
import tempfile
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.environment.harness import (
    ADMISSION_RULE_VERSION, ScriptedPolicy, admission_report, admit, package, run_trajectory,
)
from src.environment.sandbox import create_pool
from src.injector.catalog import inject_from_catalog
from src.injector.core import BUG_TYPES, SOURCES, Difficulty

ROOT = Path(__file__).resolve().parents[1]
IMAGE = (ROOT / "artifacts/p1/docker-image.id").read_text().strip()


def main() -> int:
    before = json.loads((ROOT / "artifacts/p1/reward_signal_audit.json").read_text())
    prior = {(row["repo"], row["bug_type"]): row["status"] for row in before["cells"]}

    rows, noop = [], []
    with create_pool(IMAGE, size=1) as pool:
        for repo in sorted(p for p in (ROOT / "repos").glob("*/*") if p.is_dir()):
          for source in SOURCES:
            for bug_type in BUG_TYPES:
                difficulty = Difficulty(source, bug_type, 1, "L0", "single-function")
                try:
                    inject_from_catalog(repo, difficulty, 11)
                except Exception:
                    continue
                work = Path(tempfile.mkdtemp(prefix="octorl-audit-"))
                try:
                    instance = package(repo, difficulty, 11, work)
                    row = admit(instance, pool)
                    row["prior_status"] = (prior.get((repo.name, bug_type), "unknown")
                                           if source == "mutation" else "not-previously-audited")
                    rows.append(row)
                    if row["admitted"]:
                        # The original failure mode: does doing nothing still pay?
                        with pool.lease(instance.root) as box:
                            record = run_trajectory(
                                instance, box, ScriptedPolicy([]), sampling_seed=11,
                                trajectory_id=f"noop-{repo.name}-{source}-{bug_type}",
                                policy_version="noop", base_model_revision="none",
                                manifest_ref="admission-audit")
                        noop.append({"repo": repo.name, "source": source, "bug_type": bug_type,
                                     "noop_reward": record["reward"]})
                    print(f"  {repo.name:20} {source:16} {bug_type:28} "
                          f"admitted={row['admitted']} {';'.join(row['reasons'])[:50]}", flush=True)
                finally:
                    shutil.rmtree(work, ignore_errors=True)

    report = admission_report(rows)
    transitions = Counter((r["prior_status"], "admitted" if r["admitted"] else "rejected") for r in rows)
    report["prior_status_transitions"] = {f"{a} -> {b}": n for (a, b), n in sorted(transitions.items())}
    report["noop_probe"] = {"admitted_instances_probed": len(noop),
                            "admitted_instances_where_noop_earns_reward_1":
                                sum(r["noop_reward"] == 1 for r in noop), "rows": noop}
    per_source = {}
    for row in rows:
        source = row["difficulty"]["source"]
        bucket = per_source.setdefault(source, {"admitted": 0, "rejected": 0, "reasons": Counter()})
        bucket["admitted" if row["admitted"] else "rejected"] += 1
        bucket["reasons"].update(row["reasons"])
    report["by_source"] = {s: {"admitted": v["admitted"], "rejected": v["rejected"],
                               "reasons": dict(v["reasons"])} for s, v in per_source.items()}
    OUT = ROOT / "artifacts/p1/admission_report.json"
    OUT.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(f"\nadmission rule: {ADMISSION_RULE_VERSION}")
    print(f"requested {report['requested']}  admitted {report['admitted']}  rejected {report['rejected']}")
    print("\nadmitted / rejected per source:")
    for source, value in sorted(report["by_source"].items()):
        print(f"  {source:16} admitted={value['admitted']:3} rejected={value['rejected']:3} "
              f"{dict(value['reasons'])}")
    print("\nprior status -> outcome:")
    for key, count in report["prior_status_transitions"].items():
        print(f"  {key:44} {count}")
    print(f"\nno-op earns reward 1 on "
          f"{report['noop_probe']['admitted_instances_where_noop_earns_reward_1']}"
          f"/{report['noop_probe']['admitted_instances_probed']} admitted instances")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
