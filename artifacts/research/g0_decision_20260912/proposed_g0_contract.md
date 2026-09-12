# Proposed G0 Contract

**Decision proposed for adoption: C — G0 closeout and infra/eval handoff.** This contract records a failed selection requirement; it does not manufacture a two-cell G2 experiment. It is a proposal for the spec owner to issue, not an already registered amendment or an executed fork decision.

## Goal

Produce an auditable G0 selection decision using the designated P2.4 baseline table and the existing rule of two distinct difficulty buckets with pass@1 in the inclusive interval [0.05, 0.25]. Preserve the original gate and distinguish its outcome from the later diagnostic evidence. With the current inputs, return G0 unmet, T2 parked under the current design, G1/G2 not run, and the infra/eval fork recommendation given the recorded A-bar failure.

## Authoritative inputs

- `docs/progress.md`: current A-gate and P2.3 decisions; all pre-registered constants.
- `docs/roadmap.md`: Fork Gate's G0 definition and exit route.
- `misc/v2_7_delta_2026-09-02.md`: T2's design and claim boundaries.
- `artifacts/p2/bucket_table.json`, derived from `artifacts/p1/baseline_eval.json`: designated G0 eligibility evidence.
- `artifacts/p2/replication_summary.md`, `artifacts/p2/gate_review_2026-09-10.md`, and the D1 summary, group logs and run metadata: diagnostic context only.
- `scripts/null_gate_power.py`: original simulation assumptions, not a license to change the registered thresholds.
- `artifacts/research/g0_decision_20260912/`: decision memo and reproducible descriptive sensitivity results.

Resolve these paths against `/Users/kwang/projects/octorl`. Hash each input actually used and record the implementation commit. If the designated input has changed since this proposal, report the mismatch and recompute the existing rule; do not carry forward stale counts.

## Constraints

1. **Cost and execution:** zero new GPU hours, zero API calls, zero model rollouts, no training, no remote job launch. Reuse existing records; CPU analysis only. No evolver, acceptor or archive implementation in this contract.
2. **Governance:** do not modify `docs/`, `CLAUDE.md`, `AGENTS.md`, the original simulation, or existing experimental artifacts. The implementer writes new result artifacts and a proposed decision paragraph for the docs owner. Do not automatically mark the fork decided in the live record.
3. **Eligibility:** a cell remains a distinct `(count, hint, span)` bucket. Seeds, repositories, successful instances, source subsets and repeated measurements of one bucket are not substitute second cells. Use the full designated 15-bucket table; no outcome-selected exclusions.
4. **No rule changes:** keep G0 at two cells and 5–25%; preserve G2-a ≥0.10, G2-b fresh≥30 and stored≥3, two seeds per cell and 30 greedy rounds, and all other registered constants. Do not reinterpret the source-matched D1 subset as a new authorized screening panel.
5. **Evidence labels:** original selection, replication, expanded-pool diagnostics and simulation must remain separately identified. D1 is not a G2 experiment. A simulated gate failure is not an observed G2 failure. Historical screening outcomes do not become untouched held-out evidence by renaming their manifest role.
6. **Statistical units:** report task/group and rollout counts. Keep model sampling seeds separate from injector seeds. Do not treat 28 within-group rollout pairings as independent observations. Do not pool repeated-task cohorts as if all rollouts were independent task draws.
7. **Determinism:** any new random calculation uses explicit integer master seed **20260912**, with a stable SHA-256 derivation containing the analysis/scenario identifier. Never use Python `hash()` for a seed. Keep the existing study's original seeds visible; do not retroactively assign missing historical seeds.
8. **Scope:** no environment simplification, temperature tuning, model replacement, task-generator change, new candidate search or G0 rerun. Such work would require a separate future contract with an explicit population and finite stopping rule.

## Required work

### 1. Reconstruct the eligibility table

Read every bucket in `bucket_table.json` and reconcile its pass@1 with the underlying baseline reward counts. For each row record bucket ID, successes, rollouts, instances, train/held-out counts, source population, eligibility and reason. Apply `0.05 <= pass_at_1 <= 0.25` directly, without rounded-percentage comparisons.

The expected eligible list is exactly `["c1-L0-single-function"]`, with 7/56 = 0.125. Record c3-L0-single-function as ineligible at 1/40 = 0.025. Preserve all other rows, including zero-success buckets. A workflow is successfully completed when it correctly records unmet G0; passing G0 is not an implementation success condition.

### 2. Attach the existing diagnostic evidence

Report the easy-cell replication's 2/56 and the three-cohort descriptive total of 10/168 separately from the designated input. Recompute D1 reward totals from group records: c1 19/600, c3-single-function 13/600, c2-single-function 34/1600 and c3-single-file 34/1600. Retain both c3-single-file infrastructure-failure groups as recorded; no new exclusion is authorized.

State that the original screen is mutation/condition-inversion/seed 11, while D1 samples a broader seed pool and includes additional sources for c1. Show c1's mutation-only sensitivity as 7/136 over 17 groups; do not silently replace the 75-group overall estimate with that subset.

Compute the descriptive all-pairs disagreement as the mean of `2*K*(8-K)/(8*7)`. Expected values are c1 0.06047619047619049 and c3-single-function 0.039523809523809524; equal-cell pooling is 0.05. The mutation-only c1 sensitivity produces pooled disagreement 0.06913165266106444, within numerical rounding. These values are evidence about the likely regime, not scores in an executed G2-a test.

### 3. Preserve the simulation interpretation

Link the supplied descriptive analysis and its seed, sample count, assumptions, tests and source hashes. Re-running it is optional if hashes and code match. If new simulation code is introduced, include a known-answer synthetic control with an explicit tolerance; the supplied 200,000-group iid null at μ=0.025 must estimate disagreement 0.04875 within ±1 Monte Carlo SE using the fixed seed. Exact symmetry and deterministic-outcome controls must also pass.

Do not select a new floor or threshold by maximizing the simulated pass rate. Do not call zero passes in 20,000 simulations proof that the real experiment cannot pass. Distinguish simulated G2-a/G2-b operating characteristics from e-process validity and power, which this sensitivity does not simulate.

### 4. Return a decision record and finite handoff

Write proposed artifacts under `artifacts/p2/g0/`:

- `selection_table.json`: all bucket rows and exact eligibility inputs.
- `g0_decision.json`: machine-readable status, scope, original input hashes, qualified cells, unmet requirement, A-gate evidence reference, diagnostic evidence references and proposed fork route.
- `g0_evidence_review.md`: concise population-aware explanation and proposed paragraph for the docs owner.
- `infra_eval_handoff.md`: remaining PRD §7.1 work mapped to current evidence and missing deliverables. This is a checklist for later contracts; completing all package/report work is not part of G0 closeout.

The decision JSON must distinguish these fields:

```json
{
  "record_type": "proposed_fork_decision",
  "g0_status": "unmet",
  "g0_reason": "fewer_than_two_eligible_buckets",
  "required_cells": 2,
  "eligible_cells": ["c1-L0-single-function"],
  "g0_input_policy": "designated_p2_4_table_unchanged",
  "g1_status": "not_run",
  "g2_status": "not_run",
  "t2_status": "parked_under_current_design",
  "a_gate_status": "fail_as_recorded",
  "recommended_fork": "infra_eval_exit",
  "new_model_rollouts": 0,
  "new_api_calls": 0
}
```

Proposed owner-recorded paragraph:

> G0 closed as unmet: the designated P2.4 screening table supplies one eligible bucket against the required two at pass@1 in [5%,25%]. The eligibility floor and G2 design are unchanged. Replication and D1 are retained as diagnostic context with their population differences stated; neither substitutes for an executed G2 test. Given the recorded A-bar failure, the fork proceeds to the infra/eval exit. T2 is parked under the current design; G1 and G2 were not run. This decision makes no claim that harness evolution or greedy-acceptor pathologies are absent outside this setup.

## Success conditions

1. All 15 designated buckets are accounted for, reward fractions reconcile with source data, and exactly one eligible bucket is returned for the current inputs.
2. G0 is correctly reported as unmet; G1 and G2 are explicitly not run, without fabricated commit counts, disagreement scores or gate outcomes.
3. Original, replication and D1 populations are differentiated; both overall and source-matched c1 diagnostics are visible; the repeated-task and within-group dependence limitations are stated.
4. Recomputed D1 totals match the existing summaries and disagreement formulas agree to absolute tolerance **1e-12**. Any new statistical implementation passes its fixed-seed known-answer control.
5. The proposal preserves the registered thresholds and includes a source-hashed decision record and a bounded infra/eval handoff.
6. No protected document, existing result, model configuration or generator changed; no GPU/API work occurred. Report modified/new paths and verification results.

An evidence mismatch is an implementation/data issue to resolve or explicitly return. An unmet scientific gate is a valid completed result, not a reason to repeat screening until a pass appears.
