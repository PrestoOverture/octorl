# R4r pre-launch decisions

Recorded by Claude Code on 2026-09-28, after the R4r-A implementation (Codex, stopped by usage limit, finished by
Claude Code) and **before any R4r training output or any model evaluation on test2**. The prereg
(`artifacts/self_improve/contracts/r4r_lr_fixed_rerun_prereg.yaml`, sha256 `46d844d8…`) is hash-gated by the queue
and is not edited; decisions that interpret or complete it are recorded here.

## D1 — Null-coverage acceptance for the pre-launch power simulation

- **Observed** (frozen seeds, not re-run with other seeds): null coverage of 0 at n=160 is independent 0.951,
  coupled_q_hi 0.943, coupled_q_lo 0.972. The R4r-A contract required [0.93, 0.97] "for each condition", so
  q_lo failed it.
- **Why that requirement was wrong:** the inherited R4 synthetic control is the null simulation with *both arms
  Bernoulli(p_i)*, i.e. the independent condition. The coupled conditions come from amendment A2-power-model and
  only bound power. With coupling, both arms share a uniform on most rollouts, so most d_i are exactly 0 and the
  percentile CI contains 0 by construction; R4 itself had q_lo null coverage 0.996 at n=40. Over-coverage is
  conservative. The risk that matters is false positives, and q_lo's shift-0 detection rate is 0.014 (below a
  nominal one-sided 0.025).
- **Decision:** acceptance is the independent null coverage in [0.93, 0.97] (0.951, pass). For the coupled
  conditions, coverage is reported and acceptance is shift-0 detection ≤ 0.05 (q_hi 0.034, q_lo 0.014, pass).
  The rule is encoded in `scripts/self_improve/r4r_analysis.py`. Regenerating `prelaunch_power.json` left every
  simulated number byte-identical; only the acceptance fields changed.
- **Power as registered** (80% MDE, first tested shift): independent +0.10 (0.839), q_hi +0.10 (0.998),
  q_lo +0.05 (0.970; +0.03 gives 0.739). This is about twice R4's precision at n=40.

## D2 — Infra retry restored

The R4r-A queue stopped on the first failed trainer process. R4's prereg `stops.infra` (inherited) is "a stage
fails to complete its 10 updates twice → stop branch", and R4 ran `attempt_1, attempt_2`. `r4r_run_branch.execute`
now retries **only** a process that exits non-zero or leaves no checkpoint, exactly once, as `attempt_2`. The
retry resumes from the same previous checkpoint, and the next stage resumes from whichever attempt succeeded. Every
gate (LR, selection, prereg hash, numerical, checkpoint contents, attribution, budget) still stops the queue
without retry. The seed-2718 warm-up keeps fail-stop, and Claude Code decides on any restart. Covered by
`tests/test_r4r_queue.py::test_infra_retry_once_then_next_stage_resumes_from_attempt_2` and
`::test_gate_failure_after_successful_process_is_not_retried`.

## D3 — Scope of the GPU preflight

The preflight proves the fix in real veRL processes at two points: U21–U22, resumed from R3c 137 U20 (U22 is
decisive, because a stage-target horizon would give a very different LR there), and U1–U2 from base. It does not
run a resume from a checkpoint written by an R4r process (the stage-2+ boundary). That case is covered by:
(a) the CPU simulation, which uses veRL's verbatim scheduler with optimizer and scheduler `state_dict` restore at
every boundary and reproduces R4's 60 wrong LRs exactly under the bug model; and (b) the LR gate after every stage,
which means a boundary failure can cost at most one extra stage (≈ ¥1.6) before the queue stops. Accepted.

## D4 — Review findings on the R4r-A implementation (all verified by Claude Code)

- The Agent-R1 patch re-asserts the horizon after `super().__init__` (veRL `ray_trainer.py:277→356`
  `_create_dataloader`), before `init_workers` (`main_agent_ppo.py:340→355`). The files on the remote equal
  `source_after/` (sha256), and veRL's cosine function equals the tested copy.
- The rendered R4r scripts differ from the R4 scripts only in the two opt-in flags, `r4r` naming, and the
  warm-up root. No path refers to `octorl_r4/`.
- test2 was independently recomputed: 200 instances, counts 60/52/48/40, families exactly
  {deployment_service, notification_service}, and zero fingerprint and task_id overlap with the train, dev and R3
  test manifests. Both seal hashes match, and the parquet schema equals the R3 test's.
- The analysis regression with power (`regression()`) deep-equals `r4/r4_analysis.json`.
