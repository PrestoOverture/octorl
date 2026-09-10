# P2 Easy-Bucket Summary

Bucket: `c1-L0-single-function`, Qwen3-4B, G=8.

| Evidence | Successful rollouts | Total rollouts | pass@1 | Mixed groups |
|---|---:|---:|---:|---:|
| Frozen P1.24 baseline | 7 | 56 | 12.5% | 4/7 |
| P2 format-validity probe | 1 | 56 | 1.79% | 1/7 |

The registered P2.3 model-gate input is the frozen P1.24 baseline: **pass@1 =
12.5%, which meets the ≥10% threshold**. The model-gate decision belongs to the
owner; this artifact surfaces the result but does not take that decision.

The probe differs from P1.24 by **−10.71 percentage points**, outside the allowed
±8 pp comparison band. This discrepancy is flagged for handoff and is not
auto-resolved. Both evaluations nevertheless have non-zero success and mixed
groups, so with the probe's 99.05% format-validity rate the PRD §4.9 SFT gate
selects **straight to RL, no SFT**.
