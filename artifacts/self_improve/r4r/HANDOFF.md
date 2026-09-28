# R4r CPU preparation handoff

Contract: `/Users/kwang/.codex/attachments/d0483f0b-31a9-4711-813f-a5d525362ea2/goal-objective.md`.
Frozen prereg SHA-256: `46d844d862eb55048c3be346ecf4ebc56f8f1133596e54bd4221891a256f78a4`.

## Acceptance status

**Power coverage:** coupled q_lo null coverage is **0.972**, outside the contract's [0.93, 0.97]. Claude Code resolved this in `prelaunch_decisions.md` D1: the inherited null control is the independent condition (0.951, pass), and the coupled conditions are accepted on Type I ≤ 0.05. No simulation number changed.
GPU preflight is written and dry-run only. No GPU work, training, model evaluation, remote deletion, or commit was performed. The real queue requires a passing GPU-preflight record before it starts.

## Implementation and controls

- Agent-R1 reasserts the opt-in optimizer horizon after the inherited trainer initialization. Only the actor optimizer horizon is restored; the trainer stop target remains 20 or 30–80. Worker logging occurs at the `_build_model_optimizer` boundary after actual scheduler creation.
- CPU tests execute the patched trainer initializer and worker method with a simulated veRL overwrite and the verbatim veRL cosine scheduler. Fixed U1–U80 matches R3c within 1e-12. The negative model reproduces all 60 fixed-137 R4 LRs and all 20 seed-2718 warm-up LRs within 1e-15, with optimizer and scheduler state restored at every boundary.
- LR gate: R3c 137 and reconstructed R3c 42 pass with supplied construction evidence. All 36 R4 stage metrics and the old 2718 warm-up fail the LR comparison even when supplied a valid marker. Missing steps, missing marker, duplicate steps, and incomplete reference fail closed.
- Selection gate: offline replay passes all 18 fixed stages and FD stage 1 for seeds 42/137. Selection-byte changes and reordered parquet rows fail. The warm-up gate compares per-step task multisets over all 320 rollouts.
- Prereg gate: one-byte changes, launch-record changes, and changes between processes stop execution.
- CPU trainer-stub queue completes 37 processes, skips done jobs, refuses half-finished branches, and stops without retry on LR, selection, prereg, warm-up, numerical, infra, budget, and missing preflight evidence.
- Future R4r training retains R4 fork/resume behavior (adapter + optimizer + extra state; no data.pt), selector seeds, stage boundaries, dev monitoring, budget limits, and pruning of superseded stage optimizer state inside the new root. No such training or pruning ran remotely in this contract.
- Analysis is generic in fault-instance count. R4r defaults to bootstrap seed 20260928; `--secondary` uses 20260927. R4 regression reproduces every field of the original analysis JSON, including power. A/A at n=160 has CI [0, 0].

## Sealed test2

- Generator: r2.0; start seed 300000; 200 unique fingerprints.
- Families: deployment_service, notification_service. Counts: CV 60 / MD 52 / SV 48 / normal 40.
- Independently recomputed overlap with train/dev/R3-test manifests: zero for each.
- Manifest SHA-256: `1b729b717c3dfffde8579cf19bb2eb1fae9b446aba6890d3fd75e96f47f38bb3`.
- Parquet SHA-256: `4451801e13cb1fc1ea23626ef0d8f1b1f4a48c02f0e1b4831f73c587bccb9f4c`.
- Files: `data/test2_manifest.json`, `data/test2.parquet`, `data/test2_seal.json`. No model has been evaluated on test2.

## Frozen pre-launch power

160 instance rates resampled with replacement from base dev using PCG64(20260928); 1000 replications, 10000 instance bootstrap resamples; four rollouts per instance. Couplings and simulation stream offsets retain R4 semantics. MDE is the first tested shift reaching 80% detection, not an interpolated threshold.

| Condition | q | Null coverage | Shift 0 detection | +.03 | +.05 | +.10 | 80% MDE |
|---|---:|---:|---:|---:|---:|---:|---:|
| coupled_q_hi | 0.260869565 | 0.943 | 0.034 | 0.380 | 0.730 | 0.998 | 0.1 |
| coupled_q_lo | 0.043478261 | 0.972 | 0.014 | 0.739 | 0.970 | 1.000 | 0.05 |
| independent | 1.000000000 | 0.951 | 0.027 | 0.151 | 0.336 | 0.839 | 0.1 |

`prelaunch_power.json` records the failed q_lo coverage control explicitly. CI covers test-instance and evaluation-sampling uncertainty only; training-seed variance is not estimated (n=3).

## Remote patch and deployment

Patch SHA-256: `44709cd4c219526241c95fe05f754aed5e4f5ac8141bc81a86d76472339adb7a`.
Incremental patch: `agent_r1_r4r.patch`, against the remote working tree as found, preserving its pre-existing modifications. No site-packages file was edited.

| Agent-R1 file | Before SHA-256 | After SHA-256 |
|---|---|---|
| /root/Agent-R1/agent_r1/trainer/ppo/ray_trainer.py | `808eff533b131ef8de0678d96a76fffadce5f04e29b83cf0b535145d6435e1f7` | `c892431308edebf94789176ce2051d303271722da74b4e76d2b8b253587a0682` |
| /root/Agent-R1/agent_r1/workers/fsdp_workers.py | `e710ad4a756a7b47ba9d3d2adb3f0f651c682621d563b56d09e5cbfc37c3c9d2` | `e2548c3be5a3983e113096a3a50a314d0f19b929c1dc9042a37319d5d919764d` |

veRL scheduler source SHA-256: `811aadb6c480a3ebd02f2242981a513d056ceb309347aebd6eff9ba5e52590ec`.
`dependency_hashes.json` verifies unchanged inherited code, selector, training parquet, warm-up records, and veRL source against the remote.
Runtime evidence absent from the remote checkout is deployed as byte-identical copies under `scripts/self_improve/r4r_inputs/`; protected original artifact paths are untouched. The bundled prereg is subject to the same frozen hash gate.
`remote_preflight_dry_run.txt` records the remote argument-validated commands for U21–U22 resume and U1–U2 fresh warm-up. It ran no trainer and made no scratch outputs.

| Deployed file (under /root/octorl_r3) | Local = remote SHA-256 |
|---|---|
| scripts/self_improve/r4r_analysis.py | `3206aa28368a504f901a1b40121c3cd4b973275cbbc6d685e8086ae8ecc3c1d4` |
| scripts/self_improve/r4r_archive.py | `519786d1a7553fa9ac06b053b8bbc1ea0178b6a44224178d8b4525d2a9927379` |
| scripts/self_improve/r4r_deploy.py | `a02502f33acb8b66855703e6fb637d490740017a8b3156d16b0a1038e436eb9a` |
| scripts/self_improve/r4r_gates.py | `391710337ae3a61ff9a672c42cbc4993c66f0f4503d10e253dae4719c3dfcf03` |
| scripts/self_improve/r4r_launch.py | `87e217034c44b9603f2549136f0f9d4a70461b3f0087960bc0e59c38cb1728f7` |
| scripts/self_improve/r4r_lr_gate.py | `9c8492e2730898dfec402497cf8d96b80001cd43e8ba503cc638a1f64afb0184` |
| scripts/self_improve/r4r_preflight.py | `a89878669fc4b0a4086c3cd7e8a43d8399215e3a057da4c069f3fe5f41cad648` |
| scripts/self_improve/r4r_preflight.sh | `68a6f47a9747e96cba672b3f652bf957c5e0aa55b5e471d092620b54f3e2eff4` |
| scripts/self_improve/r4r_prepare_test2.py | `432f64f750b20bbf2b9db4332bf8bd0e8f56155c8f31fdf1fc346338b0edbd1a` |
| scripts/self_improve/r4r_queue.sh | `fecab838e5b6b4d112b05bd94b6a275a0890c8c1478b3fd96f8fb368144e10ed` |
| scripts/self_improve/r4r_run_branch.py | `5754da82ce52953d37804cbc7bf43b846cb75ecd404113e1856a13600978049e` |
| scripts/self_improve/r4r_train_stage.sh | `7f41973b9b6c62e414f52266fa68ddb604745a90f073b99dd562e1c2e8be904c` |
| scripts/self_improve/r4r_warmup.sh | `95b0f47d999ac09617bc07ed67aa5ea462ad8dc36fa2fe6ba6f8414fccfdee0c` |
| scripts/self_improve/r4r_inputs/artifacts/self_improve/contracts/r4r_lr_fixed_rerun_prereg.yaml | `46d844d862eb55048c3be346ecf4ebc56f8f1133596e54bd4221891a256f78a4` |
| scripts/self_improve/r4r_inputs/artifacts/self_improve/r4/remote_records/seed_137/failure_driven/stage_1/selection.json | `cb32d05f4a61cb2953ee664ce7ccce77e964b657e68e583699d5a843073f27c8` |
| scripts/self_improve/r4r_inputs/artifacts/self_improve/r4/remote_records/seed_137/failure_driven/stage_2/selection.json | `385ceb9daa0b7596680e2938f918a421246383008ccd86cbe87c520646d0200e` |
| scripts/self_improve/r4r_inputs/artifacts/self_improve/r4/remote_records/seed_137/failure_driven/stage_3/selection.json | `ef06a2a8a1a2725419994243acbf024de95297c243167d0f00e4d19db56783b8` |
| scripts/self_improve/r4r_inputs/artifacts/self_improve/r4/remote_records/seed_137/failure_driven/stage_4/selection.json | `01045889be9c9254e6814943d97eb9be8a24598184173c0fc2c91cfdac30ec67` |
| scripts/self_improve/r4r_inputs/artifacts/self_improve/r4/remote_records/seed_137/failure_driven/stage_5/selection.json | `0654a8dba36c175a3e296eb20d5a509308d3f2b1963592edbcab2596dfa98af2` |
| scripts/self_improve/r4r_inputs/artifacts/self_improve/r4/remote_records/seed_137/failure_driven/stage_6/selection.json | `3eb1cb7df726667568e236191a3fc024e4e368a18df85a1526bc0e812250169b` |
| scripts/self_improve/r4r_inputs/artifacts/self_improve/r4/remote_records/seed_137/fixed/stage_1/selection.json | `4834e67ea47830d339fe42ba94a99a1f6ebda9d3dfe19938c75b9fd3ce1bcc18` |
| scripts/self_improve/r4r_inputs/artifacts/self_improve/r4/remote_records/seed_137/fixed/stage_2/selection.json | `6045340b38206a0bf2a870df1cd25c299c779e1d409c13c9befc8cb92e3137df` |
| scripts/self_improve/r4r_inputs/artifacts/self_improve/r4/remote_records/seed_137/fixed/stage_3/selection.json | `21ba99dafd133f07b13405b593ac91135eec18ff962f69eec294f6ef6bed0382` |
| scripts/self_improve/r4r_inputs/artifacts/self_improve/r4/remote_records/seed_137/fixed/stage_4/selection.json | `d8fad95e7225ed68db765a35dc3c6d025b4f4c7cfc37f8c59636ac7184650c21` |
| scripts/self_improve/r4r_inputs/artifacts/self_improve/r4/remote_records/seed_137/fixed/stage_5/selection.json | `6884e171dcfc50cf1c2ec4a971cbd52871942c969066f602c6d6765703bb6adf` |
| scripts/self_improve/r4r_inputs/artifacts/self_improve/r4/remote_records/seed_137/fixed/stage_6/selection.json | `a695876ba25b5509ea34e77413df920677071645bb7d553248dea19a7646699c` |
| scripts/self_improve/r4r_inputs/artifacts/self_improve/r4/remote_records/seed_2718/failure_driven/stage_1/selection.json | `9cc5694b510490dcee517bb12d604a078144c441f8c1c1bf16ca8c8f7e01bf18` |
| scripts/self_improve/r4r_inputs/artifacts/self_improve/r4/remote_records/seed_2718/failure_driven/stage_2/selection.json | `42041de59e24593a9a840aba2312fb7a4d8d5c81358a48c9e25d2edb9a107d84` |
| scripts/self_improve/r4r_inputs/artifacts/self_improve/r4/remote_records/seed_2718/failure_driven/stage_3/selection.json | `e0cb41c56a820fc0db36c9964f32abaa1df235d42a0efd3419aa382fc17c0e0a` |
| scripts/self_improve/r4r_inputs/artifacts/self_improve/r4/remote_records/seed_2718/failure_driven/stage_4/selection.json | `9f42025de10eaee7c7767f418b7c62daf7672730beb58817aeb169c341aa6fab` |
| scripts/self_improve/r4r_inputs/artifacts/self_improve/r4/remote_records/seed_2718/failure_driven/stage_5/selection.json | `e712eb4f2b3b1266264287c27202fbcb99e2246b1b57c583a2a9edd53b6f3b92` |
| scripts/self_improve/r4r_inputs/artifacts/self_improve/r4/remote_records/seed_2718/failure_driven/stage_6/selection.json | `79118b5cc7808bfb36601de345999c485a00ef8e22d86133beecebc4a64c19c1` |
| scripts/self_improve/r4r_inputs/artifacts/self_improve/r4/remote_records/seed_2718/fixed/stage_1/selection.json | `671a7fed53a229e625bbdb3243b22f7cf25f8f87373c30c4d4019d290ece47b8` |
| scripts/self_improve/r4r_inputs/artifacts/self_improve/r4/remote_records/seed_2718/fixed/stage_2/selection.json | `20cb0a0aa328cd60507e5ec59602b00d209250dead830cf6c31539c5391d401b` |
| scripts/self_improve/r4r_inputs/artifacts/self_improve/r4/remote_records/seed_2718/fixed/stage_3/selection.json | `3b9080d0ff0f6014a0e41d431042688bb6c1ee694aed3c110dc48e811e472101` |
| scripts/self_improve/r4r_inputs/artifacts/self_improve/r4/remote_records/seed_2718/fixed/stage_4/selection.json | `199afd565befb6eb3af6cfab6eeb62b91f7a6e2a051290eceb1f8124bd23e2a9` |
| scripts/self_improve/r4r_inputs/artifacts/self_improve/r4/remote_records/seed_2718/fixed/stage_5/selection.json | `cae8b78143719b68a4f171771a422e1ab40135b4f9ca4c962dfe6bbdfe10d929` |
| scripts/self_improve/r4r_inputs/artifacts/self_improve/r4/remote_records/seed_2718/fixed/stage_6/selection.json | `22d9f57c9f736d7774b7823574a9f6228765894453090c1dde1f230508730231` |
| scripts/self_improve/r4r_inputs/artifacts/self_improve/r4/remote_records/seed_2718/warmup/attributed.jsonl | `38b2ee88c4bbd81ab6220a449c9fa7d0595b3dd5d2b5f4cf11427f8e88ced3a2` |
| scripts/self_improve/r4r_inputs/artifacts/self_improve/r4/remote_records/seed_42/failure_driven/stage_1/selection.json | `c0a0ab38c4e69a6b0a30480e346df839af730bbdec4fff3e0ef1165b87c9a4d6` |
| scripts/self_improve/r4r_inputs/artifacts/self_improve/r4/remote_records/seed_42/failure_driven/stage_2/selection.json | `a9ee2518e888447facfe6b0a86052dd4f1d696927e82db73fcd5bae3ec11b8b6` |
| scripts/self_improve/r4r_inputs/artifacts/self_improve/r4/remote_records/seed_42/failure_driven/stage_3/selection.json | `6b5849bf85432f45ac16fdddcc22fbbd642ccca6dd75a8620279ea277d4f8ca6` |
| scripts/self_improve/r4r_inputs/artifacts/self_improve/r4/remote_records/seed_42/failure_driven/stage_4/selection.json | `790e4cda0995ad351f04d8cf064b97d2612bcca677b91302d1d2fd04a60e43b3` |
| scripts/self_improve/r4r_inputs/artifacts/self_improve/r4/remote_records/seed_42/failure_driven/stage_5/selection.json | `8902a358b8ab33706fa71bbd4ca5eab4a26cf7d9ef9b8821c0f3bd4b49b2a500` |
| scripts/self_improve/r4r_inputs/artifacts/self_improve/r4/remote_records/seed_42/failure_driven/stage_6/selection.json | `d358500ffd8b83d675742fa2aba1a56bd57d184f81e72bc8b85716e62447569e` |
| scripts/self_improve/r4r_inputs/artifacts/self_improve/r4/remote_records/seed_42/fixed/stage_1/selection.json | `e08a979b829c4e20820d06a094cb43ba17fbb7c14ec28e075c25462f1a1bf82c` |
| scripts/self_improve/r4r_inputs/artifacts/self_improve/r4/remote_records/seed_42/fixed/stage_2/selection.json | `750af15db548c62840c7858abd1a197b06ec3c3f00022455bc03b04611d6928c` |
| scripts/self_improve/r4r_inputs/artifacts/self_improve/r4/remote_records/seed_42/fixed/stage_3/selection.json | `8abcc31f6de200035d103d9319e7a245a83c4bd5ed8a1196c1337e23068ab709` |
| scripts/self_improve/r4r_inputs/artifacts/self_improve/r4/remote_records/seed_42/fixed/stage_4/selection.json | `c087c8e9c6d5ed9f5912db11ef043eb6359c9a5ac2a7138fd507ce2e225fe480` |
| scripts/self_improve/r4r_inputs/artifacts/self_improve/r4/remote_records/seed_42/fixed/stage_5/selection.json | `d64da3687cf4778e7fd3b8701251deee797d85fcfb0148866b3b82b46ecf3e73` |
| scripts/self_improve/r4r_inputs/artifacts/self_improve/r4/remote_records/seed_42/fixed/stage_6/selection.json | `f9ddea201795ca9aa9839690d8497511502455ac5d04451ea487471a62cd5d9c` |
| scripts/self_improve/r4r_inputs/artifacts/self_improve/r5/raw/octorl_r3c/seed_137/metrics_target_100.jsonl | `677eb7714eff9772a108f8f90c967283ab393d14e76cc980a34923982f648342` |

## Archive

Complete: 1547 regular files, 8,778,814,815 bytes, plus 121 symlinks preserved as links. `r4r_archive.py`
hashed both sides, re-inventoried the remote after the transfer, and wrote
`r4/checkpoint_archive_manifest.json` (`all_matched: true`) and `deletion_list.txt`.

## Completion by Claude Code (2026-09-28; Codex stopped at its usage limit)

- **Independent archive check before deletion:** separate `sha256sum` runs over all 1547 remote and local files
  gave identical path and hash lists, and both sides have 121 symlinks. No remote process had the directories open,
  and no R4r runtime path reads them.
- **Remote deletion** (user-authorised 2026-09-28; executed by Claude Code): exactly the four directories in
  `deletion_list.txt` were removed. `octorl_r4/{test_eval,logs,source_backup,agent_r1_r4.patch}` are kept. The data
  disk now has 13 GB free. The R3c 42/137 U20 fork checkpoints are intact.
- **Acceptance status:** the power-coverage condition is resolved by decision D1, and the infra retry is restored by
  D2 (`prelaunch_decisions.md`). `pytest tests/test_r4_*.py tests/test_r4r_*.py tests/test_r5_*.py`: 94 passed
  (`pytest_results.txt`). Scripts were redeployed, and all 52 deployed-file hashes equal local.
- **Remaining before the queue:** the GPU preflight (`r4r_preflight.sh --execute`), which needs the instance in GPU
  mode. The queue refuses to start without `preflight_pass.json`.

## Validation and limitations

- Test command: `.venv/bin/python -m pytest tests/test_r4_*.py tests/test_r4r_*.py tests/test_r5_*.py -q`.
- Final test output: `pytest_results.txt` (written after the archive completes). Tests preserve and expose the failed statistical acceptance condition rather than changing its threshold.
- Existing tests and protected paths are unchanged; no checkpoint weights or archive files are staged. The pre-existing untracked `uv.lock` is untouched.
- Environment-only tooling: installed PyTorch and OmegaConf in `.venv` for CPU scheduler tests; installed GNU rsync 3.5.1 after macOS openrsync stalled. Archive uses rsync --partial with retries; existing local R5 bytes are reused only after SHA-256 matches, and every final file is hashed again.
- GPU execution remains unverified by design. The separate training operator must resolve the failed power criterion and run the GPU preflight before training.

## Changed/new files

- `.gitignore`: excludes the local old-R4 archive.
- `scripts/self_improve/r4r_*.py`, `r4r_*.sh`: gates, launch wrappers, queue, preflight, data generation, analysis, archive, and deployment.
- `scripts/self_improve/r4r_inputs/**`: frozen read-only input copies for remote execution.
- `tests/test_r4r_{scheduler,gates,queue,artifacts}.py`: new CPU tests.
- `artifacts/self_improve/r4r/**`: incremental patch, source/hash evidence, test2 seal, power, deployment/dry-run/verification records, and this handoff.
- `artifacts/self_improve/r4/checkpoint_archive_manifest.json` and ignored `checkpoint_archive/**`: old-R4 archive and full verification manifest.
