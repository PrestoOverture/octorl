# OctoRL G0 Decision

**Recommendation: choose C for the current registered design. Close G0 as unmet, park T2, and take the infra/eval exit. Preserve G2 as not run.** Neither lowering the G0 eligibility floor nor halving the number of cell–seed runs is justified by the full evidence. A future T2 study remains possible, but it would require a new, explicitly amended measurement design rather than an administrative workaround for a missing cell.[1][2]

This is a decision under a finite research budget, not a finding that harness evolution cannot work. There is insufficient eligible-cell coverage in the designated screening table, the apparent strongest cell did not replicate at its original rate, and the newer data predict disagreement well below the existing G2-a floor. The proposed next contract is a CPU-only G0 closeout and evidence handoff; it does not launch evolution or spend GPU/API budget.

## Evidence and its scope

The five project documents consistently make the environment package an independent deliverable, keep Direction A's gate separate from T2, and constrain the fork to a predeclared decision. The live record reports A's gate as failed and P2.3's model gate as failed by owner decision. It also records F0-RSI as passed, but that assessment used extrapolated throughput rather than a complete real-repair inference benchmark.[1][2][3]

The original G0 table contains 15 supported buckets. Exactly one satisfies the inclusive 5–25% rule: c1-L0-single-function. Its 12.5% is seven successful rollouts out of 56, across seven task instances. The c3-L0-single-function runner-up is one successful rollout out of 40, across five instances. These are small screening measurements, not known population parameters.[4]

| Evidence set | c1-L0-single-function | c3-L0-single-function | Interpretation |
|---|---:|---:|---|
| Original P2.4 table, derived from P1.24 | 7/56 = 12.50% | 1/40 = 2.50% | Designated G0 screening evidence |
| Controlled easy-cell replication | 2/56 = 3.57% | Not measured | Same seven admitted easy tasks, fresh recorded sampling seeds |
| P1.24 + probe + replication, easy cell | 10/168 = 5.95% | Not measured | Descriptive pooling of repeated tasks/configurations; not 168 independent tasks |
| D1, all sampled sources | 19/600 = 3.17% | 13/600 = 2.17% | Broader admitted-pool measurement; not a direct replacement for the original screen |
| D1, mutation subset | 7/136 = 5.15% | 13/600 = 2.17% | Closer source match; only 17 easy-cell groups |

Sources: original baseline and bucket table; replication summary; D1 group records and run metadata.[4][5][6]

Population differences are material. The original screen uses mutation, condition-inversion, injector seed 11. D1 samples the admitted pool over seeds 0–49; its easy cell contains 28 commit-rollback groups, 30 feat-add groups, and 17 mutation groups. Its three other measured buckets happen to contain mutation groups only. Both experiments include held-out repositories, so neither set of already inspected outcomes should subsequently be represented as a pristine final evaluation set.[5][6]

D1 also samples with replacement from its frozen pool: the two 75-group cells contain 73 and 62 distinct task identities. Repeated groups and the eight rollouts within a group must not be treated as independent repository draws. The older replication's pooled Wilson interval is a descriptive binomial approximation, not a generalization interval over repositories and generator choices. These limitations weaken claims about exact population rates; they do not create a second eligible cell in the designated G0 table.

## Why the disagreement floor matters more than the 2% eligibility proposal

G2-a requires pooled candidate/incumbent disagreement of at least 0.10. Under the existing simulator's reseeded-null model, conditional on a task's success probability p, the two outcomes are independent Bernoulli(p). Therefore

\[
d=P(Y_1\ne Y_2)=2E[p(1-p)]=2\{\mu(1-\mu)-\sigma_p^2\}.
\]

Heterogeneity reduces disagreement at fixed mean. The homogeneous case provides an upper envelope **for the expectation at a specified true mean**, not a bound on every finite observed realization. Even without conditional independence, two binary outcomes with equal marginal mean μ satisfy d ≤ 2μ in this low-success regime. Fresh sampling seeds do not by themselves prove the conditional-independence assumption.[7][8]

With the old assumed means 0.125 and 0.025, equally weighted cells yield homogeneous expected pooled disagreement of 0.13375. Thus a 2.5% second cell is not intrinsically fatal when paired with a genuinely 12.5% first cell. Rejecting A solely because its second mean lies outside the original grid would be an incomplete argument.

At the D1 all-source means, however, the same calculation gives only 0.051861. The more general equal-marginal bound at those means is 0.053333. Using the mutation-only easy-cell mean instead gives a homogeneous expected pooled disagreement of about 0.070. None reaches the existing 0.10 floor. These are conditional sensitivity calculations at estimated inputs, not proof that the unknown real population cannot pass.

The original lower G0 endpoint is itself imperfectly aligned with G2-a: at μ=0.05, even homogeneous expected disagreement is 0.095. A single homogeneous cell needs μ ≥ (1−√0.8)/2 ≈ 0.05279 to reach 0.10. Two cells can compensate for each other under pooling, so this is not an additional per-cell rule. It does show why a pass@1 eligibility interval cannot replace a direct disagreement assessment.

Existing eight-rollout groups provide a descriptive estimate without fitting a Beta distribution. If a group contains K successes out of G=8, the fraction of its unordered rollout pairs that disagree is

\[
\widehat d_i=\frac{K_i(8-K_i)}{\binom82}=\frac{2K_i(8-K_i)}{8\cdot7}.
\]

The 28 pairs share outcomes and are not 28 independent observations. Averaging over groups produces 6.0476% for c1 and 3.9524% for c3, or **exactly 5.00% pooled**. Restricting c1 to mutation produces 9.8739%, or **6.9131% pooled** with c3. This agrees with the broad sensitivity argument while preserving the population distinction.[6][8]

## Sensitivity simulations

A separate descriptive simulation preserves the registered greedy mechanics: a 40-item dev set, 30 rounds, strict score improvements, and two run seeds per cell. Each cell's frozen dev set is shared across its two seeds; each simulated experiment draws a new dev set. The original calibration script instead pools independently drawn runs, which does not explicitly represent that shared-manifest dependence. All new scenarios use master seed **20260912**, stable SHA-256-derived streams, and **20,000 experiments per scenario**.[7][8]

The simulation varies assumed mean success rates and task heterogeneity, leaving G2-a and G2-b thresholds unchanged. It does not simulate the e-process or claim a full G2 pass, real-edit power, or an empirical T2 result.

| Scenario | Expected/mean pooled disagreement | Mean fresh commits | Mean stored commits | P(G2-b) | P(G2-a and G2-b) |
|---|---:|---:|---:|---:|---:|
| Reference: two μ=0.08 homogeneous cells | 14.71% | 49.98 | 8.73 | 99.86% | 99.86% |
| A: old μ=(0.125,0.025), homogeneous | 13.38% | 46.71 | 8.19 | 99.77% | 99.77% |
| A: old means, sd=0.07 in each cell | 12.40% | 45.22 | 7.78 | 99.31% | 98.08% |
| A: D1 means, homogeneous | 5.18% | 41.85 | 7.27 | 98.92% | 0/20,000 |
| A: D1 means and moment-fitted sd | 4.99% | 41.27 | 7.12 | 97.94% | 0/20,000 |
| A: D1 mutation-only means and moment-fitted sd | 6.86% | 43.19 | 7.46 | 99.04% | 0/20,000 |
| B: one μ=0.125 homogeneous cell, two seeds | 21.87% | 25.90 | 4.59 | 15.97% | 15.97% |
| B: one D1 easy cell, homogeneous | 6.14% | 21.97 | 3.80 | 2.03% | 0/20,000 |

Full inputs, source-matched sensitivity, quantiles, Monte Carlo standard errors, and outcomes are in `analysis.json`.[8] For a zero-success scenario, the one-sided 95% upper bound on that **simulation-model** pass probability is approximately 0.015%. This is Monte Carlo precision conditional on the assumed model; it does not incorporate uncertainty in μ, distributional misspecification, repository composition, or engine behavior.

The most informative result is that the D1-rate A simulations still produce roughly 41–42 fresh greedy commits and about seven stored commits. **Low disagreement does not imply absence of greedy false commits.** The existing G2-a rule excludes this regime by design, even though G2-b may pass. Accordingly, C closes the registered design; it does not empirically falsify the broader acceptor-pathology thesis.

For B, an exact calculation gives additional perspective. At μ=0.125 and n_dev=40, the probability that one iid candidate score strictly exceeds a fresh incumbent score is 0.432208. Over only 60 rounds, the probability of at least 30 fresh commits is **17.61%**, before imposing the stored-commit requirement. That requirement lowers the joint rate to roughly 16% in simulation. Reusing thresholds calibrated for 120 rounds would make failure mostly an artifact of the reduced design.

## Choice among A, B, and C

**A is a possible new study, but not the recommended continuation.** It would need an explicit eligibility amendment and recalibration at a declared target population. The latest evidence undermines its favorable first-cell assumption, and lowering only G0 does not repair G2-a. Lowering G2-a as well would be a substantive redesign motivated by observed outcomes. That may be defensible for a future exploratory study, but should not be presented as fulfillment of the original fork contract.

**B should be rejected.** It violates both G0's two-cell selection and the explicitly protected 2 cells × 2 seeds × 30 greedy rounds in the descope rule.[1][2] Rescaling the thresholds would be another amendment and would still leave only one task regime. Four seeds within one cell could restore the number of rounds but would not restore between-cell coverage. More rounds at the same population mean improve precision; they do not raise expected disagreement to 0.10.

**C is the defensible present decision.** It preserves the original gate, avoids spending a full fork window on a predictably excluded regime, and follows an exit path the project explicitly values. The recommended decision language is: “G0 is unmet on the designated P2.4 screening panel. A has failed its registered bar. T2 is parked under the current design, and the project proceeds on the infra/eval track. G1 and G2 were not run.”[1][2][3]

The freeze policy allows measured contradictions and errors that waste budget to motivate amendments. It does not oblige an amendment whenever a gate is inconvenient. Open-science guidance likewise permits transparent departures while distinguishing analyses informed by observed outcomes from prospective tests. Preserve the original decision and label any successor design accordingly.[9][10]

## What the literature does and does not settle

PACE supports treating the acceptor as a statistical decision and emphasizes that its guarantee is per candidate, not familywise across the run. Its experiments concern prompt evolution on arithmetic and multiple-choice tasks; they do not establish the power of OctoRL's 40-item code-repair experiment. It also explicitly separates type-I control from power. Its nominal α therefore cannot rescue G0 or determine OctoRL's empirical commit-count thresholds.[11]

Waudby-Smith and Ramdas provide the underlying bounded-mean betting framework. Such validity depends on the appropriate conditional null; it does not require baseline success to be above 5%, nor promise useful power or affordable stopping times below that level.[12]

The harness-evaluation study by Wang and colleagues motivates matched inference/feedback budgets and held-out task evaluation. It supports keeping those controls if T2 is reopened, rather than financing an apparently viable result by quietly removing them. Its frontier-model Terminal-Bench setting cannot answer whether OctoRL's 4B cells will yield measurable compounding.[13]

## Additional issues to resolve only if T2 is reopened

The existing simulation is not a sufficient implementation contract or budget ledger. Its e-process routine returns **paired instances used**. The budget expression multiplies that count once, although evaluating both candidate and incumbent normally requires two trajectories per paired instance. The same accounting issue affects fresh paired audits. The main-cost expression includes one generic greedy arm, one e-process arm and best-of-N; it does not explicitly account for both registered greedy arms and each generation's held-out evaluation.[7]

There is also a round-count inconsistency: the gate and null simulations use 120 e-process decisions, while the budget uses 15 rounds per cell–seed, or 60 decisions. This changes cost and the interpretation of the pooled control band. The record's “more than ten means implementation broken” is an operational investigation rule, not a mathematical certainty: even Binomial(120,0.05) exceeds ten with probability about 3.85%. A valid implementation can cross that diagnostic threshold by chance. Preserve the rule operationally until amended; do not infer a proven defect from that count alone.[7][11]

D1's summed group execution times imply approximately 57–79 rollouts/hour for its measured configurations. These do not replace a controlled N=8/N=16 throughput sweep, and the source populations differ. They do make the earlier raw-throughput extrapolation an insufficient basis for committing new budget without a real-task benchmark. No GPU failure or F0-RSI reclassification is declared by this memo.[6]

These issues strengthen the case against launching G2 immediately. They are follow-up requirements for a successor design, not requests to broaden the current implementation task.

## Infra/eval handoff

The exit ramp is a scope decision, not a claim that the release is already complete. PRD §7.1 still requires the diagnostic and negative-result writeups, D2/D2b/D3 work where owed, a defensible estimator comparison, reward-hacking evidence including the manual high-reward audit, and package/reproducibility completion. The progress file leaves several of these open.[1][3]

Prioritize completing those requirements on existing data and tooling. Preserve the A-gate failure as an observed gate outcome; do not reinterpret its wide intervals as proof that the true aggregation gap is zero. Preserve T2 as a possible successor with a clearly described population, fresh role-separated evaluation manifests, and an affordable calibrated design. There is no evidence-based reason to keep rerunning the current screening panel until two cells happen to cross 5%.

## Sources

All project records below are local workspace sources; source hashes for the five documents and core analysis inputs are recorded in the accompanying `analysis.json`. Project observations are distinguished above from newly derived quantities and recommendations.

1. OctoRL. [Progress Tracker](/Users/kwang/projects/octorl/docs/progress.md), live state updated September 11, 2026; P2, Fork Gate, Pre-registered Decisions.
2. OctoRL. [Project Roadmap](/Users/kwang/projects/octorl/docs/roadmap.md:160), v2.7; Fork Gate, T2 factorial, descope and exit rules. See also [Architecture](/Users/kwang/projects/octorl/docs/architecture.md), §§7b–9, and [Tech Stack](/Users/kwang/projects/octorl/docs/tech_stack.md), §§4 and 7.
3. OctoRL. [Project Requirements Document](/Users/kwang/projects/octorl/docs/prd.md:632), v2.7; §§2.5–2.6, 4.5.5, 5.3, 7–9.
4. OctoRL. [P2 bucket table](/Users/kwang/projects/octorl/artifacts/p2/bucket_table.json), derived from [P1 baseline evaluation](/Users/kwang/projects/octorl/artifacts/p1/baseline_eval.json), September 10, 2026 record.
5. OctoRL. [Replication Summary](/Users/kwang/projects/octorl/artifacts/p2/replication_summary.md) and [P2 Gate Evidence Review](/Users/kwang/projects/octorl/artifacts/p2/gate_review_2026-09-10.md), September 10, 2026. Owner's subsequent decision is in source 1.
6. OctoRL. [D1 summary](/Users/kwang/projects/octorl/artifacts/p2/d1/d1_summary.json), [run metadata](/Users/kwang/projects/octorl/artifacts/p2/d1/run_metadata.json), and four `prequential_groups_*.jsonl` files in the same directory, September 11, 2026; selection implementation [p2_d1_gap.py](/Users/kwang/projects/octorl/scripts/p2_d1_gap.py:268).
7. OctoRL. [null_gate_power.py](/Users/kwang/projects/octorl/scripts/null_gate_power.py), seed 20260902, and [v2.7 delta](/Users/kwang/projects/octorl/misc/v2_7_delta_2026-09-02.md), §§3 and 5–6, September 2, 2026.
8. [G0 descriptive analysis](/Users/kwang/projects/octorl/artifacts/research/g0_decision_20260912/analysis.json), generated by [analyze.py](/Users/kwang/projects/octorl/artifacts/research/g0_decision_20260912/analyze.py), September 12, 2026; master seed 20260912. Known-answer discordance test falls within one standard error; exact greedy symmetry and deterministic-null checks pass.
9. OctoRL. [Career / direction governance](/Users/kwang/projects/octorl/misc/career.md:89), §5 termination rule and preregistration.
10. Center for Open Science. [Preregistration](https://www.cos.io/initiatives/prereg), “I need to change my preregistration” and existing-data guidance; accessed September 12, 2026.
11. ZayxShawn. [PACE: Anytime-Valid Acceptance Tests for Self-Evolving Agents](https://arxiv.org/html/2606.08106v1), June 6, 2026; §§4 and 6. Preprint.
12. Ian Waudby-Smith and Aaditya Ramdas. [Estimating means of bounded random variables by betting](https://arxiv.org/abs/2010.09686), arXiv v7 August 25, 2022; published in JRSSB 86(1), 2024.
13. Yike Wang and colleagues. [Rethinking the Evaluation of Harness Evolution for Agents](https://arxiv.org/html/2607.12227v1), July 14, 2026; abstract and evaluation protocol. Preprint.
