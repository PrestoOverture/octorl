# P2 Replication Summary

Controlled replication of the P1.24 easy-bucket evaluation, run under P1.24's
exact inference configuration (`enforce_eager=True`) with deterministic
sampling seeds (master seed 20260910). **P1.24 remains the designated P2.3
model-gate input.**

## Configuration match

| Setting | P1.24 | Probe | Replication |
|---------|-------|-------|-------------|
| enforce_eager | True | **False** | True |
| temperature | 1.0 | 1.0 | 1.0 |
| top_k | 20 (server default) | 20 (server default) | 20 (explicit) |
| top_p | 0.95 (server default) | 0.95 (server default) | 0.95 (explicit) |
| sampling seed | absent | absent | 20260910 |
| vLLM version | 0.10.2 | 0.10.2 | 0.10.2 |
| model | Qwen3-4B | Qwen3-4B | Qwen3-4B |
| dtype | bfloat16 | bfloat16 | bfloat16 |
| max_model_len | 16384 | 16384 | 16384 |
| git commit | 88bde8f | 1b49cae | 882d134 |

The replication matches P1.24 on enforce_eager (True) and sampling parameters.
The probe ran with enforce_eager=False, which was the configuration mismatch
identified in the gate review.

## Per-instance comparison

| Instance | P1.24 | Probe | Replication |
|---|---:|---:|---:|
| config_parser | 0/8 | 0/8 | 0/8 |
| metric_aggregator | 1/8 | 0/8 | 0/8 |
| record_index | 0/8 | 1/8 | **2/8** |
| route_graph | 1/8 | 0/8 | 0/8 |
| slot_planner | 2/8 | 0/8 | 0/8 |
| stock_reservations | 0/8 | 0/8 | 0/8 |
| task_scheduler | 3/8 | 0/8 | 0/8 |
| **Total** | **7/56** | **1/56** | **2/56** |
| **pass@1** | **12.5%** | **1.79%** | **3.57%** |

No instance succeeded in all three runs. Only config_parser and
stock_reservations were consistently zero. Each run's successes concentrated in
a different instance: P1.24 in task_scheduler (3/8), the probe in record_index
(1/8), the replication in record_index (2/8).

## Aggregate metrics

| Metric | P1.24 | Probe | Replication |
|---|---:|---:|---:|
| pass@1 | 12.50% | 1.79% | 3.57% |
| Format validity | not tracked | 99.05% (625/631) | 98.52% (598/607) |
| Mixed groups | 4/7 | 1/7 | 1/7 |
| Wall time | 28,129s (full) | 1,991s | 2,174s |
| GPU cost (est.) | — | ~¥1.11 | ~¥1.21 |

Pooled across all 168 rollouts: **10/168 = 5.95%** (Wilson 95% CI: [3.2%, 10.7%]).

## Observations

1. The replication (enforce_eager=True, 2/56) is closer to the probe
   (enforce_eager=False, 1/56) than to P1.24 (enforce_eager=True, 7/56).
   The enforce_eager mismatch identified in the gate review does not explain
   the P1.24–probe discrepancy.

2. The dominant source of variance is rollout sampling at T=1.0 across
   instances with very low true success rates. With only 7 instances and
   8 rollouts each, which instance "gets lucky" varies substantially between
   runs.

3. The pooled estimate (5.95%) is below the 10% model-gate threshold.
   Its 95% CI [3.2%, 10.7%] barely touches 10% at the upper bound.

## P2.3 model-gate status

P1.24's 12.5% remains the designated gate input per the pre-implementation
recommendation (gate_review_2026-09-10.md). It numerically passes the
registered point-estimate rule (≥10%). The replication is diagnostic evidence;
the final P2.3 decision is the owner's.

## Run provenance

| Field | Value |
|---|---|
| Git commit | 882d134 |
| Sampling master seed | 20260910 |
| Seed derivation | SHA-256(`"{master}\|{task}\|{rollout}\|{turn}"`) mod 2³¹ |
| Injector seed | 11 |
| Termination breakdown | 47 max_steps, 9 stop |
| vLLM log | `artifacts/p2/replication_vllm.log` (enforce_eager=True confirmed) |
| Eval log | `artifacts/p2/replication.log` |
| Full results | `artifacts/p2/replication.json` |
