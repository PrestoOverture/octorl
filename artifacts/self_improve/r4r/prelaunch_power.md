# R4r pre-launch power

160 instances; 4 rollouts; 1000 replications; 10000 instance resamples; seed 20260928.
MDE is the first tested shift reaching 80% detection; shifts are clipped at probability 1.

| Condition | q | Null coverage | +0 detection | +.03 | +.05 | +.10 | MDE |
|---|---:|---:|---:|---:|---:|---:|---|
| independent | 1.000000000 | 0.951 | 0.027 | 0.151 | 0.336 | 0.839 | 0.1 |
| coupled_q_hi | 0.260869565 | 0.943 | 0.034 | 0.380 | 0.730 | 0.998 | 0.1 |
| coupled_q_lo | 0.043478261 | 0.972 | 0.014 | 0.739 | 0.970 | 1.000 | 0.05 |

A/A CI: [0, 0]. R4 n=40 regression: all fields deep-equal.

Null acceptance (independent: coverage of 0 in [0.93, 0.97]; coupled: coverage reported, Type I (shift 0 detection) <= 0.05): {"independent": true, "coupled_q_hi": true, "coupled_q_lo": true}
See prelaunch_decisions.md D1.
CI covers test-instance and evaluation-sampling uncertainty only; training-seed variance is not estimated (n=3)
