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
