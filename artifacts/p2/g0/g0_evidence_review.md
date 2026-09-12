# Proposed G0 evidence review — 2026-09-12

## Decision

**G0 is unmet under the unchanged registered rule.** The designated P2.4 table contains one, not two, distinct `(count, hint, span)` buckets with pass@1 in the inclusive interval [5%, 25%]. `c1-L0-single-function` is eligible at 7/56 = 0.125. The next closest named cell, `c3-L0-single-function`, is ineligible at 1/40 = 0.025. All 15 rows are retained in `selection_table.json`; no source, repository, seed, or repeated cohort was promoted into a substitute bucket.

This is a proposed fork decision for the docs owner. It does not modify the live decision record. G1 and G2 were not run, T2 is parked under the current design, and no G2 commit or gate outcome is inferred. Given the separately recorded A-bar failure, the proposed route is the infra/eval exit.

## Population-aware evidence

The designated screen is a narrow mutation / condition-inversion / injector-seed-11 population. Its easy cell was 7/56. The controlled replication was separately 2/56, and the descriptive three-cohort total was 10/168; neither replaces the designated G0 input.

D1 sampled a broader injector-seed pool. Its c1 cell also included commit-rollback and feat-add sources. Recomputed directly from the group JSONL:

| D1 bucket or subset | Task groups | Rollouts | Successes | All-pairs disagreement |
|---|---:|---:|---:|---:|
| c1-L0-single-function, overall | 75 | 600 | 19 | 0.060476190476190475 |
| c1-L0-single-function, mutation only | 17 | 136 | 7 | 0.098739495798319324 |
| c3-L0-single-function | 75 | 600 | 13 | 0.039523809523809524 |
| c2-L0-single-function | 200 | 1600 | 34 | 0.038571428571428569 |
| c3-L0-single-file | 200 | 1600 | 34 | 0.040000000000000001 |

Both c3-single-file infrastructure-failure groups remain included as recorded. Equal-cell pooling of the two overall single-function cells gives disagreement 0.05. Substituting only c1's mutation-matched subset gives 0.06913165266106444; this sensitivity is shown, not used as a new screening panel.

The disagreement statistic is the group-level mean of `2*K*(8-K)/(8*7)`. Its 28 within-group rollout pairings are dependent and are not 28 independent observations. D1 also repeats some task identities across groups, so groups and rollouts must not be interpreted as independent task draws.

## Simulation interpretation

The supplied descriptive sensitivity is linked at `artifacts/research/g0_decision_20260912/analysis.json` and authenticated by the hashes in `g0_decision.json`. It used master seed 20260912, stable SHA-256 seed derivation, 20,000 simulations per scenario, n_dev=40, and 30 rounds. Its 200,000-group iid-null control at mu=0.025 measured disagreement 0.048763392857142852 against 0.04875, within one Monte Carlo SE (0.00023541593075641277); its exact-symmetry and deterministic controls also pass.

These simulations describe possible G2-a/G2-b operating characteristics. Zero simulated passes do not prove that a real experiment could not pass, and the sensitivity does not simulate e-process validity or power. D1 is diagnostic evidence, not a G2 experiment.

## Proposed paragraph for the docs owner

> G0 closed as unmet: the designated P2.4 screening table supplies one eligible bucket against the required two at pass@1 in [5%,25%]. The eligibility floor and G2 design are unchanged. Replication and D1 are retained as diagnostic context with their population differences stated; neither substitutes for an executed G2 test. Given the recorded A-bar failure, the fork proceeds to the infra/eval exit. T2 is parked under the current design; G1 and G2 were not run. This decision makes no claim that harness evolution or greedy-acceptor pathologies are absent outside this setup.

## Provenance

Implementation commit: `77d65cd680a3f51b3b496a6dbbdb68f2b61e0e76`. Exact SHA-256 hashes for every input used are recorded in `g0_decision.json`. No model rollouts, API calls, GPU work, threshold changes, or modifications to protected documents or prior result artifacts were made.
