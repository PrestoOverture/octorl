# P2 gate evidence review — 2026-09-10

This is an advisory review for the owner. It does not record a P2.3 decision, change a threshold, or adjudicate a stop-loss. No additional rollouts or training were run. The GPU host was accessed read-only to inspect the two saved vLLM logs.

**Recommendation.** Retain P1.24 as the designated P2.3 input, as stated in the supplied context and `easy_bucket_summary.md`; its 7/56 score numerically passes the registered 10% point-estimate rule. Keep the probe discrepancy visible. Before expanding into P2.5, fix sampling provenance and take one bounded replication under P1.24's inference configuration. Treat that replication as diagnostic evidence with its use declared before execution. Keep T2 pending at G0 under the existing rules. Difficulty changes, a replacement gate measurement, and stop-loss decisions remain with the owner.

**Verified evidence.**

| Item | P1.24 | P2 format probe |
|---|---:|---:|
| Easy successful rollouts | 7/56 | 1/56 |
| Easy pass@1 | 12.5% | 1.7857% |
| Easy instances with any success / mixed groups | 4/7 | 1/7 |
| Injector seed | 11 | 11 |
| Request temperature | 1.0 | 1.0 |
| Explicit request sampling seed | Absent | Absent |
| vLLM version in retained server log | 0.10.2 | 0.10.2 |
| Engine seed in retained server log | 0 | 0 |
| `enforce_eager` in retained server log | True | False |
| Maximum sequence length | 16,384 | 16,384 |
| Model dtype | bfloat16 | bfloat16 |
| Model sampling defaults logged by server | top_k=20, top_p=0.95 | top_k=20, top_p=0.95 |

The requests override the model's default temperature of 0.6 with 1.0. A matching engine seed does not supply a recorded, independently addressable sampling seed for each rollout and turn. In particular, seed 11 is the task injection seed. See `scripts/baseline_eval.py:121` and `:221`.

All eight requested easy-task identities match between the JSON artifacts, including the seven admitted tasks: task ID, repo content reference, difficulty, injector seed, verification seed, admission rule version, and admission outcome. No differences were found in the compared admission reward/cheat fields. The version diff between the two evaluations adds format counting and checkpoint bookkeeping; it does not change the inference request parameters, prompt, tools, or grader.

The eager-mode difference is a confirmed configuration mismatch and a candidate explanation, not a demonstrated cause of the score change. The retained P1 evaluation log also resumes 17 instance checkpoints, including two easy instances. Its retained server log contains one initialization; it cannot establish the complete settings history of the earlier checkpointed segment. The saved reports contain aggregate reward counts, not the full request/response and final-patch traces needed to replay the discrepancy. A causal attribution remains unresolved.

**Statistical interpretation.**

The stated probability `P(X <= 1 | n=56, p=0.125) = 0.0050897` is arithmetically correct. It conditions on 0.125 being a known success probability, whereas 0.125 was estimated from another 56 rollouts. It therefore overstates the evidence when used as a comparison between these two estimated rates.

- My two-sided Fisher exact calculation for `[[7,49],[1,55]]` gives **p=0.0606021**. [Method reference](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.fisher_exact.html).
- A task-stratified exact calculation, conditioning on each task's combined success count and exchanging its two sets of eight rollout labels, gives **p=0.0600000** for an absolute total-success difference at least six. This allows different success probabilities across tasks, assuming exchangeable independent draws within a task under the null.
- A one-sided decline comparison would be approximately 0.0303 (0.0300 stratified), if that direction had been specified beforehand. These are descriptive checks, not new gate rules. The discrepancy warrants investigation; the two-sided result does not establish that the runs are equivalent.
- The descriptive pooled count is **8/112 = 7.1429%**. Under a common-binomial model, its exact 95% interval is **[3.1341%, 13.5897%]**; Wilson gives [3.6636%, 13.4645%]. Both include 10%. Pooling also averages two recorded runtime configurations and repeated observations of only seven task instances; it does not by itself establish performance over the broader generator population. [Interval methods](https://www.itl.nist.gov/div898/handbook/prc/section2/prc241.htm).
- With another 56 rollouts, **nine** new successes suffice for a pooled point estimate above 10%: `(8+9)/168 = 10.119%`. Twelve are unnecessary for that arithmetic, and even twelve would leave the Wilson interval at approximately [7.84%, 17.67%]. A small extra batch is a reproducibility check, not a reliable way to certify which side of 10% the underlying rate occupies.

Calculations used exact combinatorial enumeration or deterministic numerical inversion; no Monte Carlo seeds were used. Controls checked Fisher's published `[[6,2],[1,4]]` example, p=1 on a balanced identical table, exact null rejection probability <=0.05, normalization/symmetry of the task-stratified null, and the closed-form zero-success binomial confidence endpoint.

**Population and G0.**

Every admitted P1.24 task uses **source=mutation, type=condition-inversion, injector_seed=11**. `satisfiable()` chooses the first feasible bug type. Thus the 99 tasks form a narrow screening panel. The progress record separately establishes admitted instances from commit-rollback and feat-add, including parcel_ledger; parcel_ledger's rejection in this evaluation does not establish its exclusion from every source.

Under the bucket-level interpretation used by the supplied G0 table, exactly one bucket currently meets [5%,25%]. This establishes incomplete G0 selection on the observed panel. It does not establish a G2-a or G2-b failure, nor impossibility of a second candidate across the supported generator.

Additional rollouts restricted to `c1-L0-single-function` can update only that bucket. They cannot bring another bucket into range. Splitting the same bucket into two seeds or selecting two observed successful instances would change the selection unit. Widening the eligibility range or using a trained checkpoint would also require an explicit amendment and review of the downstream design.

To assess further G0 candidates, a separate, fixed screening plan can specify supported buckets, source/type proportions, new injector seeds, admission/deduplication handling, and a finite sample budget before collecting more outcomes. Preserve the existing selection unit and thresholds. If that screen still yields fewer than two candidates, report T2 as unavailable under the current G0 design, without labelling an unrun G2 as failed. A's route remains governed by its own registered bar.

The easy bucket already has one defect, single-function span, and the most informative current hint: L0 names the defect type, file, line, and function. Restricting to single-function or choosing L0 again does not make this particular gate panel easier. Any further reduction needs an explicit new intervention and a newly described population. Preserve the supported hard-bucket contrasts needed for D1/D2/D3.

**Proposed bounded next contract, for the owner to issue.**

1. Add and record explicit model sampling seeds, independently from injector/verifier seeds, with a stable derivation that never uses Python `hash()`. Freeze model/tokenizer identity, request sampling parameters, server execution settings, task identities, and evaluator/harness/grader versions. Retain per-rollout outcomes, termination/error reasons, and enough traces to audit repairs. Make checkpoint reuse reject incompatible run metadata.
2. Run one fixed **7 easy tasks x 8 fresh rollouts = 56** replication under the recorded P1.24 configuration, including `enforce_eager=True`, T=1.0, top_k=20, top_p=0.95, and the same model/context/tool settings. A proposed fresh sampling master seed is **20260910**, with task/rollout/turn identifiers incorporated into the stable derivation. Keep injector seed 11. This is a diagnostic replication of the existing panel; fresh-task coverage belongs in a separately specified screen.
3. Freeze the sample count and diagnostic-only use before launching, run inside tmux, finish the entire batch, and report all outcomes. Do not repeat until a favorable rate appears or silently replace the designated P1.24 gate input. The observed format probe took 1,991.3 seconds for 56 rollouts; the user's roughly 30-minute / CNY 1 estimate is consistent with that run, without guaranteeing eager-mode timing.
4. Return the baseline, probe, replication, configuration differences, and population limitations together for the owner's P2.3 judgment. Any decision to reduce difficulty or replace the authoritative measurement should be recorded by the docs owner before dependent work.

P2.5-P2.8 produce A's own fork evidence. The roadmap's fork window begins after P2.4 and consumes those diagnostics; T2's G0 is not a prerequisite for measuring A's registered gap. A pass keeps A primary. An unavailable fallback is not a reason to change A's bar.

**Evidence provenance.**

Local and GPU-host copies of both evaluation JSON files have matching SHA-256 hashes:

| File | SHA-256 |
|---|---|
| `artifacts/p1/baseline_eval.json` | `ba1fae61857806ea8b510ac414a65a3695cbf5abc25430fa1be96b981b8e962a` |
| `artifacts/p2/format_probe.json` | `a4712a78361a5c493dfcfaf0f16a3342c6946f99ad7dc79c2d046c8803054c67` |
| `artifacts/p2/bucket_table.json` (local) | `a7e77f002003d43939e07aecbfbddc2f28f06bfed46ade37774d9eedd5637b7d` |
| `/root/octorl/artifacts/p1/vllm.log` (GPU host) | `bc73fe2aa0d55c4d9d721f3e282662fa73c4fe5b2bbc18c096bc1bb25810b35d` |
| `/root/octorl/artifacts/p2/vllm.log` (GPU host) | `06a711746aeb357f4b4772d92a2e99c566acfc47de5320c186caf4b808f980aa` |

P1 server settings were read at log lines 3/11 and sampling defaults at 47/50/51. P2 settings were read at 3/10 and defaults at 76/79/80. Each retained server log contains one engine initialization. Reference definitions: `docs/roadmap.md:143-168`, `docs/progress.md:113-125`, `docs/progress.md:303-325`, `docs/prd.md:510-529`, and `misc/v2_7_delta_2026-09-02.md:49-75`. Documentation, implementation files, and existing result artifacts were not changed by this review.
