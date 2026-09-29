# R6 pre-launch power

160 fault instances; 4 rollouts; 1000 replications; 10000 instance bootstrap resamples; PCG64(20260930).

The single-seed simulation is the unchanged R4r procedure with the test3 n=160 design. The pooled simulation requires all three seed-level effects to be positive and the pooled CI lower bound to exceed zero.

| Condition | q | Independent-procedure null coverage | +0 detection | +.03 | +.05 | +.10 |
|---|---:|---:|---:|---:|---:|---:|
| independent | 1.000000000 | 0.950 | 0.026 | 0.149 | 0.358 | 0.899 |
| coupled_q_hi | 0.260869565 | 0.950 | 0.034 | 0.403 | 0.758 | 0.999 |
| coupled_q_lo | 0.043478261 | 0.962 | 0.021 | 0.805 | 0.981 | 1.000 |

## Pooled over three seeds

| Condition | q | +0 detection | +.03 | +.05 |
|---|---:|---:|---:|---:|
| independent | 1.000000000 | 0.024 | 0.351 | 0.740 |
| coupled_q_hi | 0.260869565 | 0.019 | 0.781 | 0.980 |
| coupled_q_lo | 0.043478261 | 0.016 | 0.990 | 1.000 |

Independent null coverage pass: `True`.
A/A CI: `[0, 0]`.
R4r regression: all fields deep-equal.

CI covers test-instance and evaluation-sampling uncertainty only; training-seed variance is not estimated (n=3)
