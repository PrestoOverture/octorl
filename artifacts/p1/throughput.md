# P1.21 throughput gate

Measured 2026-09-06T23:41:58+0800 on macOS-26.6.2-arm64-arm-64bit (arm64, 8 CPUs), Docker linux/arm64 29.7.2.

- Image: `sha256:886d2583b75ae4556a59725b5a3c63d24da56d3e5e12b830aa07bb7eca6577ca`
- Policy: **scripted-12step (no model). Per-trajectory latency here is a LOWER bound and the rate an UPPER bound on any model-in-the-loop figure: adding generation latency lengthens trajectories and lowers traj/hour.**
- Design: 3 repos x 2 buckets x 8 trajectories/cell, 12 steps each; one warm-up trajectory per pool excluded from every figure.
- Packaging (once per task instance, amortized over G=8 rollouts): 0.08s mean, excluded from trajectory wall-clock.

## Rates

| N | trajectories | wall (s) | traj/hour | p50 (s) | p95 (s) | max (s) | <90s gate |
|---|---|---|---|---|---|---|---|
| 1 | 48 | 153.98 | **1122.2** | 3.14 | 3.76 | 4.29 | PASS |
| 4 | 48 | 61.21 | **2822.9** | 5.26 | 6.29 | 6.42 | PASS |
| 8 | 48 | 59.01 | **2928.4** | 10.85 | 11.25 | 11.46 | PASS |

## Phase breakdown (mean seconds per trajectory)

| N | lease+reset | tools (non-test) | run_tests | verify | sum |
|---|---|---|---|---|---|
| 1 | 0.003 | 0.006 | 1.49 | 1.609 | 3.108 |
| 4 | 0.006 | 0.01 | 2.38 | 2.615 | 5.011 |
| 8 | 0.011 | 0.015 | 4.056 | 5.581 | 9.663 |

## Validity of this measurement

- **Host caveat.** The gate number that matters is the one from the machine that runs
  rollouts during training. The AutoDL GPU host cannot run any container runtime
  (no Docker, no rootless Podman -- `unshare` is blocked, `/dev/fuse` absent), so this
  run is from macOS Docker Desktop, i.e. a Linux VM on arm64. Treat it as a
  functional demonstration of the pipeline, not as the host gate.
- **Linux-side bound.** A container-free pytest probe on the GPU host
  (`linux_pytest_probe.json`, 128 CPUs) puts the dominant term -- four `run_tests`
  plus verify's two suites -- at 5.62 s mean / 8.82 s max per 12-step trajectory.
  Docker `exec` overhead is *not* separately measured: the lease+reset column below is
  0.005-0.012 s, but per-command container overhead sits inside the run_tests and
  verify segments. The probe is therefore a partial cost estimate, not an upper bound
  on a model-driven trajectory, and does not by itself close the gate.
- **Scripted policy.** No model is in the loop; per-step generation latency is absent
  from every number above, so these rates are ceilings, not floors.
- **Concurrency.** N=8 on an 8-CPU laptop is CPU-oversubscribed (each container is
  capped at 2 CPUs), which is why N=8 is slower than N=4 here. The GPU host reports
  128 logical CPUs but its cgroup quota is `cpu.max = 1600000 100000` = 16 cores,
  shared with training -- not 128.


## Trajectory validity

- Steps per trajectory: [12]
- `run_tests` calls per trajectory: [4]
- Tools exercised: apply_patch, list_files, read_file, run_tests, search_code
- Rewards observed: [1] (the scripted trajectory repairs the bug, so 1 is expected)
