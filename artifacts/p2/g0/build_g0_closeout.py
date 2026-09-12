"""Build the proposed G0 closeout from frozen, existing evidence only."""
from __future__ import annotations

import hashlib
import json
import subprocess
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
LOW, HIGH = 0.05, 0.25
REQUIRED_CELLS = 2

INPUTS = [
    "docs/progress.md",
    "docs/roadmap.md",
    "misc/v2_7_delta_2026-09-02.md",
    "artifacts/p2/bucket_table.json",
    "artifacts/p1/baseline_eval.json",
    "artifacts/p2/replication.json",
    "artifacts/p2/replication_summary.md",
    "artifacts/p2/gate_review_2026-09-10.md",
    "artifacts/p2/d1/d1_summary.json",
    "artifacts/p2/d1/run_metadata.json",
    "artifacts/p2/d1/prequential_groups_c1-L0-single-function.jsonl",
    "artifacts/p2/d1/prequential_groups_c2-L0-single-function.jsonl",
    "artifacts/p2/d1/prequential_groups_c3-L0-single-file.jsonl",
    "artifacts/p2/d1/prequential_groups_c3-L0-single-function.jsonl",
    "scripts/null_gate_power.py",
    "artifacts/research/g0_decision_20260912/analyze.py",
    "artifacts/research/g0_decision_20260912/analysis.json",
]


def load(path: str) -> dict:
    return json.loads((ROOT / path).read_text())


def sha256(path: str) -> str:
    return hashlib.sha256((ROOT / path).read_bytes()).hexdigest()


def d1_rows(bucket: str) -> list[dict]:
    path = ROOT / "artifacts/p2/d1" / f"prequential_groups_{bucket}.jsonl"
    return [json.loads(line) for line in path.read_text().splitlines()]


def describe_groups(rows: list[dict]) -> dict:
    groups = len(rows)
    successes = sum(int(row["K"]) for row in rows)
    disagreement = sum(2 * row["K"] * (8 - row["K"]) / (8 * 7) for row in rows) / groups
    return {
        "groups": groups,
        "rollouts": groups * 8,
        "successes": successes,
        "mean_reward": successes / (groups * 8),
        "all_pairs_disagreement": disagreement,
        "unique_tasks": len({json.dumps(row["task"], sort_keys=True) for row in rows}),
        "infrastructure_failure_groups": sum(not row["infrastructure_ok"] for row in rows),
    }


def main() -> None:
    bucket_table = load("artifacts/p2/bucket_table.json")["buckets"]
    baseline = load("artifacts/p1/baseline_eval.json")

    by_bucket: dict[str, list[dict]] = defaultdict(list)
    for key, result in baseline["pass_at_1"].items():
        repo, bucket = key.split("|", 1)
        by_bucket[bucket].append({"repo": repo, **result})

    selection_rows = []
    for bucket in sorted(bucket_table):
        source_rows = by_bucket[bucket]
        successes = sum(round(row["mean_reward"] * row["rollouts"]) for row in source_rows)
        rollouts = sum(row["rollouts"] for row in source_rows)
        measured = successes / rollouts
        stated = bucket_table[bucket]["pass_at_1"]
        assert abs(measured - stated) < 1e-15
        eligible = LOW <= measured <= HIGH
        selection_rows.append({
            "bucket_id": bucket,
            "successes": successes,
            "rollouts": rollouts,
            "pass_at_1": measured,
            "instances": bucket_table[bucket]["n_instances"],
            "train_instances": bucket_table[bucket]["n_train"],
            "held_out_instances": bucket_table[bucket]["n_held_out"],
            "source_population": {
                "source": baseline["source"],
                "difficulty_type": "condition-inversion",
                "injector_seed": 11,
                "repositories": sorted(row["repo"] for row in source_rows),
            },
            "eligible": eligible,
            "eligibility_reason": "within_inclusive_0.05_0.25" if eligible else (
                "below_0.05" if measured < LOW else "above_0.25"
            ),
        })

    eligible = [row["bucket_id"] for row in selection_rows if row["eligible"]]
    assert len(selection_rows) == 15
    assert eligible == ["c1-L0-single-function"]

    selection = {
        "record_type": "g0_selection_table",
        "input": "artifacts/p2/bucket_table.json",
        "underlying_rewards": "artifacts/p1/baseline_eval.json",
        "eligibility_interval": {"lower": LOW, "upper": HIGH, "inclusive": True},
        "required_distinct_buckets": REQUIRED_CELLS,
        "rows": selection_rows,
        "eligible_cells": eligible,
    }
    (OUT / "selection_table.json").write_text(json.dumps(selection, indent=2) + "\n")

    diagnostics = {}
    for bucket in [
        "c1-L0-single-function", "c3-L0-single-function",
        "c2-L0-single-function", "c3-L0-single-file",
    ]:
        diagnostics[bucket] = describe_groups(d1_rows(bucket))
    c1_mutation = [r for r in d1_rows("c1-L0-single-function") if r["task"]["source"] == "mutation"]
    diagnostics["c1-L0-single-function_mutation_only"] = describe_groups(c1_mutation)
    diagnostics["equal_cell_pool_c1_c3_single_function"] = {
        "all_pairs_disagreement": (
            diagnostics["c1-L0-single-function"]["all_pairs_disagreement"]
            + diagnostics["c3-L0-single-function"]["all_pairs_disagreement"]
        ) / 2
    }
    diagnostics["equal_cell_pool_c1_mutation_only_c3_single_function"] = {
        "all_pairs_disagreement": (
            diagnostics["c1-L0-single-function_mutation_only"]["all_pairs_disagreement"]
            + diagnostics["c3-L0-single-function"]["all_pairs_disagreement"]
        ) / 2
    }
    expected = {
        "c1-L0-single-function": (19, 600, 0.06047619047619049),
        "c3-L0-single-function": (13, 600, 0.039523809523809524),
        "c2-L0-single-function": (34, 1600, None),
        "c3-L0-single-file": (34, 1600, None),
    }
    for bucket, (successes, rollouts, disagreement) in expected.items():
        assert diagnostics[bucket]["successes"] == successes
        assert diagnostics[bucket]["rollouts"] == rollouts
        if disagreement is not None:
            assert abs(diagnostics[bucket]["all_pairs_disagreement"] - disagreement) <= 1e-12
    assert diagnostics["c3-L0-single-file"]["infrastructure_failure_groups"] == 2
    assert diagnostics["c1-L0-single-function_mutation_only"]["successes"] == 7
    assert diagnostics["c1-L0-single-function_mutation_only"]["rollouts"] == 136
    assert abs(diagnostics["equal_cell_pool_c1_c3_single_function"]["all_pairs_disagreement"] - 0.05) <= 1e-12
    assert abs(diagnostics["equal_cell_pool_c1_mutation_only_c3_single_function"]["all_pairs_disagreement"] - 0.06913165266106444) <= 1e-12

    sensitivity = load("artifacts/research/g0_decision_20260912/analysis.json")
    assert sensitivity["master_seed"] == 20260912
    assert sensitivity["N_SIM"] == 20000
    assert sensitivity["checks"]["within_one_SE"] is True
    assert sensitivity["checks"]["greedy_exact_check"] is True
    assert sensitivity["checks"]["deterministic_check"] is True
    # The saved analysis authenticates the code and every overlapping input it used.
    for path, digest in sensitivity["input_sha256"].items():
        assert sha256(path) == digest, f"supplied sensitivity input changed: {path}"

    hashes = {path: sha256(path) for path in INPUTS}
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    decision = {
        "record_type": "proposed_fork_decision",
        "generated_at_date": "2026-09-12",
        "implementation_commit": commit,
        "g0_status": "unmet",
        "g0_reason": "fewer_than_two_eligible_buckets",
        "required_cells": REQUIRED_CELLS,
        "eligible_cells": eligible,
        "g0_input_policy": "designated_p2_4_table_unchanged",
        "g1_status": "not_run",
        "g2_status": "not_run",
        "t2_status": "parked_under_current_design",
        "a_gate_status": "fail_as_recorded",
        "recommended_fork": "infra_eval_exit",
        "new_model_rollouts": 0,
        "new_api_calls": 0,
        "new_gpu_hours": 0,
        "registered_constants_preserved": {
            "g0_required_cells": 2,
            "g0_pass_at_1_interval": [0.05, 0.25],
            "g2_a_min_discordance": 0.10,
            "g2_b_min_greedy_fresh_commits": 30,
            "g2_b_min_greedy_stored_commits": 3,
            "g2_seeds_per_cell": 2,
            "g2_greedy_rounds": 30,
        },
        "diagnostic_evidence": {
            "designated_original": {"successes": 7, "rollouts": 56, "pass_at_1": 0.125},
            "replication": {"successes": 2, "rollouts": 56, "pass_at_1": 2 / 56},
            "three_cohort_descriptive_total": {"successes": 10, "rollouts": 168, "pass_at_1": 10 / 168},
            "d1": diagnostics,
        },
        "population_notes": {
            "original_screen": "mutation / condition-inversion / injector seed 11",
            "d1": "broader injector-seed pool; c1 also includes commit-rollback and feat-add",
            "dependence": "tasks repeat across groups and the 28 within-group rollout pairs are not independent task draws",
        },
        "sensitivity_analysis": {
            "path": "artifacts/research/g0_decision_20260912/analysis.json",
            "status": sensitivity["status"],
            "master_seed": sensitivity["master_seed"],
            "simulation_samples": sensitivity["N_SIM"],
            "assumptions": {"n_dev": sensitivity["N_DEV"], "rounds": sensitivity["ROUNDS"]},
            "synthetic_checks": sensitivity["checks"],
            "interpretation": "descriptive operating characteristics only; not an observed G2 or an e-process validity/power simulation",
        },
        "input_sha256": hashes,
    }
    (OUT / "g0_decision.json").write_text(json.dumps(decision, indent=2) + "\n")

    review = f"""# Proposed G0 evidence review — 2026-09-12

## Decision

**G0 is unmet under the unchanged registered rule.** The designated P2.4 table contains one, not two, distinct `(count, hint, span)` buckets with pass@1 in the inclusive interval [5%, 25%]. `c1-L0-single-function` is eligible at 7/56 = 0.125. The next closest named cell, `c3-L0-single-function`, is ineligible at 1/40 = 0.025. All 15 rows are retained in `selection_table.json`; no source, repository, seed, or repeated cohort was promoted into a substitute bucket.

This is a proposed fork decision for the docs owner. It does not modify the live decision record. G1 and G2 were not run, T2 is parked under the current design, and no G2 commit or gate outcome is inferred. Given the separately recorded A-bar failure, the proposed route is the infra/eval exit.

## Population-aware evidence

The designated screen is a narrow mutation / condition-inversion / injector-seed-11 population. Its easy cell was 7/56. The controlled replication was separately 2/56, and the descriptive three-cohort total was 10/168; neither replaces the designated G0 input.

D1 sampled a broader injector-seed pool. Its c1 cell also included commit-rollback and feat-add sources. Recomputed directly from the group JSONL:

| D1 bucket or subset | Task groups | Rollouts | Successes | All-pairs disagreement |
|---|---:|---:|---:|---:|
| c1-L0-single-function, overall | 75 | 600 | 19 | {diagnostics['c1-L0-single-function']['all_pairs_disagreement']:.17g} |
| c1-L0-single-function, mutation only | 17 | 136 | 7 | {diagnostics['c1-L0-single-function_mutation_only']['all_pairs_disagreement']:.17g} |
| c3-L0-single-function | 75 | 600 | 13 | {diagnostics['c3-L0-single-function']['all_pairs_disagreement']:.17g} |
| c2-L0-single-function | 200 | 1600 | 34 | {diagnostics['c2-L0-single-function']['all_pairs_disagreement']:.17g} |
| c3-L0-single-file | 200 | 1600 | 34 | {diagnostics['c3-L0-single-file']['all_pairs_disagreement']:.17g} |

Both c3-single-file infrastructure-failure groups remain included as recorded. Equal-cell pooling of the two overall single-function cells gives disagreement 0.05. Substituting only c1's mutation-matched subset gives 0.06913165266106444; this sensitivity is shown, not used as a new screening panel.

The disagreement statistic is the group-level mean of `2*K*(8-K)/(8*7)`. Its 28 within-group rollout pairings are dependent and are not 28 independent observations. D1 also repeats some task identities across groups, so groups and rollouts must not be interpreted as independent task draws.

## Simulation interpretation

The supplied descriptive sensitivity is linked at `artifacts/research/g0_decision_20260912/analysis.json` and authenticated by the hashes in `g0_decision.json`. It used master seed 20260912, stable SHA-256 seed derivation, 20,000 simulations per scenario, n_dev=40, and 30 rounds. Its 200,000-group iid-null control at mu=0.025 measured disagreement {sensitivity['checks']['measured_discordance']:.17g} against 0.04875, within one Monte Carlo SE ({sensitivity['checks']['SE']:.17g}); its exact-symmetry and deterministic controls also pass.

These simulations describe possible G2-a/G2-b operating characteristics. Zero simulated passes do not prove that a real experiment could not pass, and the sensitivity does not simulate e-process validity or power. D1 is diagnostic evidence, not a G2 experiment.

## Proposed paragraph for the docs owner

> G0 closed as unmet: the designated P2.4 screening table supplies one eligible bucket against the required two at pass@1 in [5%,25%]. The eligibility floor and G2 design are unchanged. Replication and D1 are retained as diagnostic context with their population differences stated; neither substitutes for an executed G2 test. Given the recorded A-bar failure, the fork proceeds to the infra/eval exit. T2 is parked under the current design; G1 and G2 were not run. This decision makes no claim that harness evolution or greedy-acceptor pathologies are absent outside this setup.

## Provenance

Implementation commit: `{commit}`. Exact SHA-256 hashes for every input used are recorded in `g0_decision.json`. No model rollouts, API calls, GPU work, threshold changes, or modifications to protected documents or prior result artifacts were made.
"""
    (OUT / "g0_evidence_review.md").write_text(review)

    handoff = """# Infra/eval handoff

This is a bounded checklist for future contracts, not a claim that the package/report track is complete. The fork recommendation assumes the docs owner adopts the proposed G0 decision.

| PRD §7.1 requirement | Current evidence | Missing deliverable / finite next contract |
|---|---|---|
| 1. D1 quantifies B1/B_emp/B2 across the (mu, sd) grid | D1 headline and group logs exist; A-bar is a valid small-gap result. | Complete D2 noise-corrected heterogeneity and grid, D2b posterior predictive check, and D3 monotonicity; produce the final diagnostic figure and text without reopening A. |
| 2. Two negative results, including D5 | Probe-domination rationale is documented; D5 has not been completed. | Write the formal probe-domination result with Pilot-Commit relationship; run CPU-only estimator study at fitted parameters plus the already-scoped real prequential sanity checks, then write D5 either way. |
| 3. Tool-call format validity >95% | P2 format probe reports 99.05%; replication reports 98.52%. | Consolidate provenance and denominator in the final report. |
| 4. Reward-hacking gate | 27 seeded attacks are reported at reward 0; grading-integrity tests exist. | Manually audit at least 50 high-reward trajectories and report seeded-attack recall. Preserve the residual in-process-forgery limitation. |
| 5. Open-source package complete and independently valid | Verifiers adapter, README, baseline evaluation, heatmap, wheel/sdist, and GitHub v0.1.0 release exist. | Test installation and quickstart from a clean environment; resolve or explicitly defer PyPI publishing; perform independent reproduction/package audit. |
| 6. Algorithm defense self-test | No completed evidence identified in the live progress record. | Complete all 28 questions without notes and retain an auditable record. |

Cross-cutting report work: retain F3's unsupported cross-file limitation; report the narrow original baseline population; reconcile GPU cost (including S5); include the D1 small-gap result, negative results, package limitations, and reproducibility hashes. No A-side P3/P4 training matrix, T2 evolver, acceptor, or archive work belongs in this exit route.
"""
    (OUT / "infra_eval_handoff.md").write_text(handoff)
    print(json.dumps({"rows": len(selection_rows), "eligible": eligible, "d1": diagnostics}, indent=2))


if __name__ == "__main__":
    main()
