# Exploratory (NOT pre-registered): R3c checkpoints on the R4 sealed test

Run 2026-09-28 after the pre-registered R4 test analysis was complete; does not alter Q1/Q2 verdicts.
Same protocol as R4 test: test_manifest 072e6fe876c3, G=4, eval seed 410000 (CRN with base).

| Checkpoint | dev fault FCR (Δ vs base 66.25%) | test fault FCR (Δ vs base 63.75%) | test up/down vs base | test normal |
|---|---:|---:|---:|---:|
| R3c seed 42 U50 | 70.63% (+4.38pp) | 64.38% (+0.63pp) | 1/0 | 72.5% |
| R3c seed 42 U80 | 71.25% (+5.00pp) | 64.38% (+0.63pp) | 1/0 | 75.0% |
| R3c seed 137 U50 | 78.75% (+12.50pp) | 66.88% (+3.13pp) | 5/0 | 77.5% |
| R3c seed 137 U80 | 76.88% (+10.63pp) | 67.50% (+3.75pp) | 6/0 | 77.5% |

Reading: R3c's dev gains shrink by ~70-90% on the held-out test families; seed 137 keeps a small,
one-directional gain (6 up / 0 down, descriptive, not clustered). R4 fixed_137 U80 (same seed, forked at
U20, R4 stage pipeline) shows 0/0 on test and lower dev (68.75%) than R3c U80 — an unexplained
R3c-vs-R4-fixed gap (sampler stratification, seen-last ordering, per-stage restarts, or seed noise).
