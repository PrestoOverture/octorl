# R3 fixed-distribution GRPO handoff

## Outcome

R3 stopped at the pre-registered five-update smoke gate. The measured training rate projects to **25.06 GPU-hours / 54.63 CNY for training alone**, exceeding the 50 CNY total ceiling before required dev evaluations and the 3 CNY final-test reserve. Neither 100-update seed run was started.

Provisional recommendation: **no improvement**. This is smoke-only evidence: seed 42 U=5 improved continuous dev reward by 0.00262, well below the +0.05 threshold, and seed 137 was not run because the budget stop fired.

## Smoke gate

| Measure | Result |
|---|---:|
| Successful updates | 5 |
| Successful rollouts | 80 |
| Measured update time | 2,255.33 s |
| Updates/hour | 7.98 |
| Rollouts/hour | 127.70 |
| Peak observed VRAM | 21,476 MiB |
| Maximum grad norm | 0.1942 |
| NaN observed | No |
| Numerical gate | Pass |
| Cost gate | **Fail / stop** |

Update 2 contained no mixed groups and therefore had zero policy gradient. The other four updates had non-zero gradients. The pre-registered 20-consecutive-zero-mixed-groups stop was not reached.

## Frozen dev evaluations

All evaluations used the same 50-instance dev manifest (`b865c8f79e4498fd342946da7273c8722deca24536a50673b802f8bf5a913ca9`) with G=4 and eval seed 310000.

| Checkpoint | Continuous mean | Binary FCR | Delta continuous vs. base |
|---|---:|---:|---:|
| Base | 0.943038 | 0.63125 | — |
| Seed 42, U=1 | 0.944104 | 0.64375 | +0.001066 |
| Seed 42, U=5 | 0.945660 | 0.65625 | +0.002622 |

The test split remains sealed and was never evaluated.

## Cost

| Component | Projection |
|---|---:|
| Two seeds × 100 training updates | 25.06 h / 54.63 CNY |
| 25 required dev evaluations, from observed mean | 0.68 h / 1.49 CNY |
| Final-test reserve | 3.00 CNY |
| Projected total | **59.11 CNY** |
| Ceiling | 50.00 CNY |

The observed remote R3 window was 17:57:48–19:18:51 CST (1.351 h, approximately 2.94 CNY at 2.18 CNY/hour). This window includes preparation, base/U1/U5 evaluation, successful smoke work, and two diagnosed retries.

## Checkpoints

Remote checkpoint root: `/root/autodl-tmp/octorl_r3/seed_42/checkpoints`

- `global_step_1` through `global_step_4`: LoRA adapter, optimizer, scheduler/RNG extra state, and dataloader state; full model shard pruned.
- `global_step_5`: the same artifacts plus the full model shard.
- U=1 and U=5 LoRA adapters were successfully loaded for dev evaluation.
- U=50 reload validation is inapplicable because the smoke cost stop prevented main training.

## Diagnostics and deviations

1. The first U=3 attempt failed in FSDP2 backpropagation because policy offload produced a CPU gradient for a CUDA-sharded parameter. Checkpoint 2 remained valid.
2. Disabling policy offload while retaining the actor on GPU left insufficient memory for the vLLM allocation.
3. The successful U=3–5 retry used the framework's manual parameter and optimizer offload path with policy offload disabled. It completed all checkpoints without the device mismatch.
4. A dataloader worker was killed during shutdown after checkpoint 5 was fully saved; the trainer exited 0 and the checkpoint loaded successfully for U=5 evaluation.
5. The framework logs one pre-clip global gradient norm. `grad_norm_clipped` is reported as `min(grad_norm_unclipped, 1.0)`; all observed norms were already below the clipping threshold.
6. The smoke checkpoints were saved every update to satisfy U=1/U=5 evaluation and resume validation. Full-model shards were retained only for the latest checkpoint to stay within disk limits.

## Delivered artifacts

- `update_metrics.csv`: five per-update metric rows with loss, clipped/unclipped gradient norm, reward mean/std, KL, response length, mixed-group fraction, wall time, throughput, and LR.
- `dev_evaluations.csv`: base, U=1, and U=5 frozen-dev results.
- `seed_42_smoke_curves.png`: loss, KL, reward, and mixed-group curves.
- `smoke_summary.json`: machine-readable gate, cost, dev, checkpoint, and diagnostic summary.
- `raw/`: source metrics, eval JSON, logs, VRAM samples, trajectories, and failed-attempt diagnostics.
- `../data/`: frozen train/dev/test manifests and Parquet files. The test data was generated and sealed but not evaluated.

## Verification

- `45 passed` for `tests/test_tool_recovery.py` and `tests/test_tool_recovery_adapter.py`.
- Training/evaluation/report scripts compile; the shell launcher passes `bash -n`.
- Frozen split fingerprints have zero overlap and reproduce the remote hashes.
- All five reported metric rows are finite.

## What next

Close R3 as a budget-stop result. Any continuation should be a new contract that first lowers the long-tail rollout cost (for example, a framework-level total-trajectory token cap or a cheaper model/hardware plan) and then repeats the five-update gate. Do not resume these checkpoints under the current 50 CNY ceiling.
