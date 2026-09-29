# R4r results: failure-driven vs fixed GRPO, LR schedule as preregistered

Prereg `artifacts/self_improve/contracts/r4r_lr_fixed_rerun_prereg.yaml` (sha256 `46d844d8…`), with pre-launch
decisions in `prelaunch_decisions.md`. Training was 2026-09-28 22:54 → 2026-09-29 20:38; evaluation was
20:39 → 21:02. Numbers below are read from `r4r_analysis_test2.json` / `r4r_analysis_test_r3.json` (frozen analysis
code, regression-checked against R4) and were recomputed independently by Claude Code: point estimates identical,
bootstrap CIs equal within Monte Carlo error, and CRN sampling seeds identical across all models.

## Verdicts (primary endpoint: test2, 160 fault instances × 4 rollouts, eval seed 510000, bootstrap seed 20260928)

| Question | Decision rule outcome |
|---|---|
| **Q2** failure-driven vs fixed | **`Q2_failure_driven_better`**: all three Q2_s > 0, pooled 95% CI lower bound > 0, normal guardrail passes |
| **Q1** fixed vs base | **`no_evidence_of_improvement`**: Q1_s ≤ 0 for seeds 42 and 137 |

| Estimand | Seed 42 | Seed 137 | Seed 2718 | Pooled | 95% instance CI |
|---|---:|---:|---:|---:|---|
| Q2 = FD − fixed | +2.50 pp | +3.75 pp | +2.97 pp | **+3.07 pp** | [+1.72, +4.53] pp |
| Q1 = fixed − base | −0.16 pp | −0.63 pp | +4.84 pp | +1.35 pp | [+0.10, +2.81] pp |
| FD − base (descriptive) | +2.34 pp | +3.12 pp | +7.81 pp | — | — |

Test2 fault success rate: base 61.25%; fixed 61.09 / 60.62 / 66.09%; FD 63.59 / 64.38 / 69.06%.
Normal guardrail (≥ base − 5 pp; base normal 33.8%): every model passes (FD 38.8 / 40.6 / 37.5%).

**Secondary endpoint** (R3 test, 40 fault instances, already opened in R4, no decision authority): the same
verdicts. Q2 pooled +3.54 pp, CI [+1.46, +6.04] pp, with all three seeds positive (6/0, 5/0, 6/0 up/down).
Q1 pooled +0.63 pp, CI [−0.42, +1.88] pp. Base reproduces R4's 63.75% exactly.

## Where the Q2 effect lives (descriptive)

Pooled Q2 by fault type on test2: **missing_dependency +9.46 pp (52 instances)**, constraint_violation −0.14 pp
(60), stale_version +0.17 pp (48). The R3 test shows the same pattern (MD +8.97 pp). Missing_dependency is the
cell the failure-driven selector up-weighted during training (17–22 of 40 rows per stage, 42.5–55%, vs π0 = 31%).
In R4, where the LR was about 10% of the plan, the same up-weighting produced no test gain.

## Required caveats

- The CI covers test-instance and evaluation-sampling uncertainty only. **Training-seed variance is not
  estimated** (n = 3 seeds). All three seeds are positive on Q2 on both test sets, but three seeds cannot bound
  seed-to-seed variability.
- The effect is modest (+3 pp pooled) and concentrated in one fault type. It is evidence that *this* adaptive
  selection rule beats fixed-distribution GRPO on held-out families under this budget. It is not evidence of
  general self-improvement.
- Q1: fixed-distribution GRPO shows no preregistered improvement over base on held-out families (two of three
  seeds ≈ 0).
- R4 (LR-deviating run) verdicts are superseded and are not pooled with R4r.

## Exploratory (not preregistered for any decision): R3c U80 on test2

R3c 42 U80 61.88% (+0.63 pp vs base), R3c 137 U80 64.69% (+3.44 pp). R4r fixed 137 (same seed, same LR curve,
R4r stage pipeline) is 60.62%, so the R3c-vs-pipeline gap seen on dev also appears on test. The candidates are
row selection/order and per-stage restarts (LR is now ruled out). This is an R5 limitation item.

## Cost

Training ¥46.86 (37 processes, 21.49 GPU-h, all attempt_1) + dev health ¥0.52 + evaluation ≈ ¥0.8
(20:39–21:02) → about ¥48.2 for R4r. Preflight about ¥0.7.
