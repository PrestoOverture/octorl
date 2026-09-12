# Infra/eval handoff

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
