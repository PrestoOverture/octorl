# R5 handoff

Completed CPU/offline only. No GPU job, new training, new evaluation, full base-model load, or commit. No remote files changed.

## Files

Modified: `README.md` (one appended study section), `.gitignore` (R5 checkpoints directory).

New scripts:
- `scripts/self_improve/r5_pull.py`
- `scripts/self_improve/r5_evidence.py`
- `scripts/self_improve/r5_scheduler_evidence.py`
- `scripts/self_improve/r5_load_adapter.py`
- `scripts/self_improve/r5_diagnostics.py`
- `scripts/self_improve/r5_gap.py`
- `scripts/self_improve/r5_report.py`
- `scripts/self_improve/r5_verify_report.py`

New test: `tests/test_r5_deliverable.py`.

All new deliverables are under `artifacts/self_improve/r5/`: `R5_report.md`, `checkpoint_manifest.json`, `pull_inventory.json`, `verification.json`, this handoff, raw source records, 13 ignored adapter directories, nine diagnostics CSVs, five PNG curves, reconstructed seed-42 JSONL, and audit JSON/Markdown records. `pull_inventory.json` enumerates every downloaded file and hash.

## Inventory and controls

- 133 pulled files: 107 raw records (85 requested plus 22 supplemental) and 26 adapter files across all 13 adapters. Missing: none. Every downloaded file matched its remote SHA-256; the test re-hashed all adapter files and checked evaluation links.
- Seed-137 parser control: 80 matching steps, 10,488 matching shared values, no source JSONL keys absent from the log. Control preceded reconstruction.
- Seed-42 log: 84 occurrences; duplicate steps 41–44, each retained with all occurrences and source line numbers. Last occurrence wins. Reconstructed steps are exactly 1–80. Save timings agree with pruning-step records and latest checkpoint 80.
- Checker passed all 13. Truncated-file and zero-LoRA-B temporary copies exit nonzero; nonfinite tensor controls also pass.
- Nine CSVs have exact step coverage. All 60 numeric report-table values match frozen source fields.
- `pytest tests/test_r5_*.py tests/test_r4_*.py`: 54 passed in 19.33 seconds. See `raw/pytest_final.log` and `verification.json`.
- Protected tracked paths unchanged; no staged paths; all adapter weights ignored. Existing tests unchanged.

## Gap summary

Resolved-config differences: PRESENT (13 keys per stage). Stage-horizon cosine difference: PRESENT; the source formula reproduces all 60 R4 fixed-137 LRs within 1e-15. Fresh five-step warmup at each stage: ABSENT. Optimizer discard between stages: ABSENT based on restore logs and resume links; exact historical moment-by-moment continuity is NOT DETERMINABLE OFFLINE after intermediate pruning. Row-distribution/ID/order differences: PRESENT (233/240 instance overlap). U21 metrics differ: PRESENT. Seed-noise contribution: NOT DETERMINABLE OFFLINE. No causal conclusion is claimed; seed noise cannot be ruled out offline. R3c test evidence is exploratory, not pre-registered.

Q1: `no_evidence_of_improvement`. Q2: `no_evidence_of_difference`.

## Wall time, risks and deviations

Acquisition wall time was 1,491.633517 seconds (24 minutes 51.6 seconds), including transfers and hash checks. The compact evidence query took 2.272237 seconds within that acquisition interval; do not add overlapping wall intervals. No R5 GPU expense; CPU/network billing rate was not supplied.

Full load/generation is implemented but not run, as required. Adapter configs consistently have a null revision and a character-list target regex; recorded evaluation provenance supplies the base revision, and the checker normalizes the regex only in memory. Token masks are absent from attributed records, so token-level mask correctness is not established. Original R3c within-batch dataloader order is not independently established; numbered dump order is labeled accordingly. Full optimizer state was pruned from superseded R4 checkpoints; final U80 and some R3c states still exist remotely. The local archive supports adapter warm starts, not exact optimizer resumes. No task-scope deviations; extra read-only source/dump evidence was gathered to substantiate the gap investigation.

## Finalisation (R4r)

Completed CPU/offline. Q1/Q2 now use R4r; R4 remains a labelled protocol-deviation case study. No GPU, remote access, training, evaluation, experimental bootstrap rerun, full base-model load, staging or commit. `uv.lock` is untouched.

Changed/new files:

- Modified scripts: [r5_report.py](/Users/kwang/projects/octorl/scripts/self_improve/r5_report.py), [r5_verify_report.py](/Users/kwang/projects/octorl/scripts/self_improve/r5_verify_report.py), [r5_diagnostics.py](/Users/kwang/projects/octorl/scripts/self_improve/r5_diagnostics.py).
- New helpers: [r5_r4r_evidence.py](/Users/kwang/projects/octorl/scripts/self_improve/r5_r4r_evidence.py), [r5_r4r_report.py](/Users/kwang/projects/octorl/scripts/self_improve/r5_r4r_report.py), [r5_r4r_validate.py](/Users/kwang/projects/octorl/scripts/self_improve/r5_r4r_validate.py).
- Tests: [test_r5_deliverable.py](/Users/kwang/projects/octorl/tests/test_r5_deliverable.py), [test_r5_r4r_finalisation.py](/Users/kwang/projects/octorl/tests/test_r5_r4r_finalisation.py).
- Documentation and evidence: [README.md](/Users/kwang/projects/octorl/README.md), [R5_report.md](/Users/kwang/projects/octorl/artifacts/self_improve/r5/R5_report.md), [checkpoint_manifest.json](/Users/kwang/projects/octorl/artifacts/self_improve/r5/checkpoint_manifest.json), [verification.json](/Users/kwang/projects/octorl/artifacts/self_improve/r5/verification.json), [HANDOFF.md](/Users/kwang/projects/octorl/artifacts/self_improve/r5/HANDOFF.md), [pytest_final.log](/Users/kwang/projects/octorl/artifacts/self_improve/r5/raw/pytest_final.log).
- [Diagnostics](/Users/kwang/projects/octorl/artifacts/self_improve/r5/diagnostics): updated `adapter_checks.json`, `attribution_summary.json`, `run_summary.json`; added seven `r4r_*.csv` files, five `r4r_*.png` curves, and `r4r_{duplicate_steps,fault_type_breakdown,lr_check,protected_before,resume_evidence,selector_exposure}.json`. Historical R4 CSVs and plots are unchanged.

Validation commands and results (run from the repository root):

- `.venv/bin/python scripts/self_improve/r5_r4r_evidence.py`: passed; 20 unique manifest rows, local file hashes, exact paired endpoint links and recorded lineage.
- `.venv/bin/python scripts/self_improve/r5_diagnostics.py --r4r-only`: passed; exact U21–U80 coverage for six branches and U1–U20 for warm-up; each branch has 960 rollouts / 240 groups; attribution agrees at every step. All 380 LR values match the reference, max absolute difference 0.0; R4 fixed-137 U22 fails that equality control.
- `.venv/bin/python scripts/self_improve/r5_load_adapter.py --check-only > artifacts/self_improve/r5/diagnostics/adapter_checks.json`: all 20 pass, 504 finite tensors each, every LoRA-B non-zero. Tests retain truncated-file, zero-LoRA-B and nonfinite controls.
- `.venv/bin/python scripts/self_improve/r5_report.py` twice: byte-identical output.
- `.venv/bin/python scripts/self_improve/r5_verify_report.py`: passed, 226 exact numeric entries; recomputed descriptive cells agree within 1e-12. All five required corruption controls fail verification, as does a changed prose number.
- `.venv/bin/python -m pytest tests/test_r5_*.py tests/test_r4_*.py tests/test_r4r_*.py -q > artifacts/self_improve/r5/raw/pytest_final.log 2>&1`: final run 110 passed, 0 skipped in 47.40s. Initial run before the added prose control: 109 passed in 53.74s.
- `.venv/bin/python scripts/self_improve/r5_r4r_validate.py`: passed; regeneration, source verification and all 3,016 protected hashes unchanged, including every adapter file; staged paths empty.
- `git status --porcelain`, `git diff --stat`, `git diff --check`: only contract outputs plus untouched pre-existing `uv.lock`; no whitespace errors. README changes are confined to the study section.

Exact pooled differences and CI95 bounds (proportions):

| Endpoint | Question | Verdict | Difference | CI95 |
|---|---|---|---:|---|
| test2 | Q1 | `no_evidence_of_improvement` | 0.013541666666666667 | [0.0010416666666666667, 0.028125] |
| test2 | Q2 | `Q2_failure_driven_better` | 0.03072916666666667 | [0.0171875, 0.04531249999999999] |
| test_r3 (secondary, no decision authority) | Q1 | `no_evidence_of_improvement` | 0.006249999999999999 | [-0.004166666666666667, 0.018749999999999996] |
| test_r3 (secondary, no decision authority) | Q2 | `Q2_failure_driven_better` | 0.03541666666666667 | [0.014583333333333332, 0.06041666666666666] |

Fault-type table (`descriptive_only`; equal seed weighting; no cell CIs or tests):

| Endpoint | Fault type | n | Q1 pooled | Q2 pooled |
|---|---|---:|---:|---:|
| test2 | constraint_violation | 60 | 0.018055555555555557 | -0.001388888888888889 |
| test2 | missing_dependency | 52 | 0.019230769230769232 | 0.09455128205128205 |
| test2 | stale_version | 48 | 0.001736111111111111 | 0.0017361111111111112 |
| test_r3 | constraint_violation | 15 | 0.005555555555555556 | 0.011111111111111112 |
| test_r3 | missing_dependency | 13 | 0.019230769230769232 | 0.08974358974358976 |
| test_r3 | stale_version | 12 | -0.006944444444444444 | 0.006944444444444444 |

Warm-up provenance: `r4r_warmup_2718_u20.remote_path = null`. Neither permitted local log contains its literal adapter path; checkpoint-parent evidence was not used to construct one. Its lineage parent is null (base), and all local files are hashed. All R4r remote hash checks remain explicitly unverified.

Risks: only three training seeds; intervals exclude training-seed variance; the modest effect concentrates in missing_dependency; fixed-GRPO dev gains do not transfer; the exploratory R3c-vs-pipeline gap remains despite corrected LR; only two held-out families; token masks are unverified. Optimizer state was not pulled; local adapters support warm starts only. Full loading was not tested. This is not evidence of general or recursive self-improvement.

Contract reconciliations: wrote `raw/pytest_final.log` because the Tests success condition explicitly requires it, although the general output allowlist omits that path. Retained the mandated R4 “about 10%” comparison sentence in §3 despite the general §4-only wording; R4 source paths and historical verdicts remain restricted to §4 or labelled numeric rows. No other deviations.

## Finalisation polish — Delta Contract

Diff for this delta only:

- `scripts/self_improve/r5_r4r_report.py`: added the JSON-derived Q1 decision-rule clauses directly after the §2 verdict, including all seed differences and the explanation that every clause is required. This helper is the current §2 generator called by `r5_report.py`.
- `scripts/self_improve/r5_report.py`: replaced the stale §4 paragraph with the requested R4r rerun / unchanged as-run verdict wording.
- `scripts/self_improve/r5_verify_report.py`: independently recomputes the expected Q1 clause line from test2 JSON and requires exact equality.
- `tests/test_r5_r4r_finalisation.py`: added a temporary-report negative control flipping `= False` to `= True`; checks non-zero CLI exit and the specific clause-validation error.
- `artifacts/self_improve/r5/R5_report.md`: regenerated with only those two wording changes.
- `artifacts/self_improve/r5/raw/pytest_final.log` and `verification.json`: refreshed validation evidence.
- `artifacts/self_improve/r5/HANDOFF.md`: appended this delta record.

Validation:

- `.venv/bin/python scripts/self_improve/r5_report.py` and `.venv/bin/python scripts/self_improve/r5_verify_report.py`: passed; all 226 numeric entries still match.
- `.venv/bin/python -m pytest tests/test_r5_*.py tests/test_r4_*.py tests/test_r4r_*.py -q`: 111 passed in 53.82s, including the new clause-flip control. An initial invocation stopped at collection due to an f-string syntax error; that error was corrected before this passing run.
- `.venv/bin/python scripts/self_improve/r5_r4r_validate.py`: passed; two regenerations are byte-identical, verifier passes, and all 3,016 protected files remain unchanged.
- Delta baseline SHA-256 comparison: only the eight files listed above changed; no files added or removed. `git diff --check` is clean. No staging or commit.

No other changes or deviations.
