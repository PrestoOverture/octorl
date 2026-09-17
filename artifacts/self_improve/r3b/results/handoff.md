# R3b fault-only binary Sign pilot handoff

## Outcome

R3b completed all 20 seed-42 updates from the frozen base Qwen3-4B model. The
training distribution contained only the 320 frozen fault tasks, reward was the
binary verifier score, and the patched Agent-R1 estimator used the fixed
reference advantage `A = 2r - 1` without group centering or normalization.

After U=20 was fully logged and saved, one DataLoader worker emitted a
killed-at-shutdown traceback. The trainer exited 0, and the U=20 LoRA adapter
loaded successfully for the dev evaluation; this is recorded as a shutdown
diagnostic rather than a failed update or numerical crash.

The optimizer-signal pilot passes. All 20 updates had nonzero gradients, every
logged trajectory advantage was exactly -1 or +1, and `advantage_abs_mean` was
1.0 on every update. Mean gradient norm was 0.17284, 1.64x R3's five-update mean
of 0.10510; peak norms were similar (0.19597 versus 0.19422). R3b therefore
improved signal consistency rather than peak magnitude.

## Training signal

| Measure | R3b result | R3 reference |
|---|---:|---:|
| Completed updates | 20 | 5 |
| Completed rollouts | 320 | 80 |
| Nonzero-gradient updates | 20/20 (100%) | 4/5 (80%) |
| Grad norm, min / mean / max | 0.14959 / 0.17284 / 0.19597 | 0 / 0.10510 / 0.19422 |
| Mean mixed-group fraction | 62.5% | 35.0% |
| Mean binary training reward | 71.5625% | n/a (continuous target) |
| Mean Sign advantage | +0.43125 | n/a |
| Mean advantage absolute value | 1.0 | n/a |

The pre-run expectation of approximately 6% fault-task pass rate was not borne
out by the frozen data. The reused frozen base dev evaluation has 63.125%
fault-only FCR, and R3b training batches averaged 71.5625%. The resulting mean
Sign advantage was positive, not negative. This discrepancy is recorded rather
than retroactively changing the experiment.

## Frozen dev evaluation

Both checkpoints use the full 50-instance dev manifest, G=4, and eval seed
310000. The base row is the identical R3 base evaluation reused at zero new
rollout cost; U=20 was evaluated after checkpoint selection.

| Measure | Base | U=20 | Delta |
|---|---:|---:|---:|
| Full-dev continuous reward | 0.943038 | 0.947563 | +0.004525 |
| Full-dev binary pass rate | 52.0% | 55.5% | +3.5 pp |
| Fault-only binary FCR | 63.125% | 68.125% | +5.0 pp |
| Normal-only binary pass rate | 7.5% | 5.0% | -2.5 pp |
| Binary mixed-group fraction | 36.0% | 34.0% | -2.0 pp |

This is positive single-seed pilot evidence, not a two-seed capability claim.
The final test split remains sealed.

## Cost and checkpoints

- Experiment window: 2026-09-17 15:20:48–16:42:45 CST (4,917 seconds).
- Registered rate: CNY 2.18/GPU-hour.
- Training plus U=20 evaluation cost: approximately CNY 2.98, below the CNY 8 ceiling.
- Required checkpoints exist at U=5, U=10, U=15, and U=20.
- Final checkpoint: `/root/autodl-tmp/octorl_r3b/seed_42/checkpoints/global_step_20`.

## Delivered evidence

- `smoke_summary.json`: machine-readable outcome, signal comparison, dev results, and cost.
- `update_metrics.csv`: 20 per-update rows including Sign diagnostics and group composition.
- `dev_evaluations.csv`: base/U=20 rows for full-dev, fault-only, and normal-only scopes.
- `../raw/seed_42/metrics_target_20.jsonl`: original per-update framework metrics.
- `../raw/seed_42/training.log`: complete training log.
- `../raw/seed_42/trajectories.jsonl`: 320 training trajectories.
- `../raw/evals/`: reused base and new U=20 evaluation JSON, logs, and trajectories.
- `../agent_r1_sign_advantage.patch`: exact Agent-R1 framework patch.
