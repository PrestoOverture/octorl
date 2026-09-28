# R4 Training Queue Watch Log

## Preflight Verification (2026-09-27T15:40:00+08:00)

Preflight checks executed against host `autodl-r4` (`connect.bjb1.seetacloud.com:13439`):

- **GPU Status (`nvidia-smi`):**
  - Device: NVIDIA GeForce RTX 4090 (24564 MiB)
  - VRAM used: 1 MiB / 24564 MiB (~0 MiB)
  - Running processes: None (`No running processes found`)
  - Status: **PASS**
- **Disk Space (`df -h /root/autodl-tmp`):**
  - Filesystem: `/dev/md0`
  - Size: 50G, Used: 39G, Available: 12G (78% used)
  - Requirement: $\ge 10\text{ GB}$ free (Available: 12 GB)
  - Status: **PASS**
- **Existing `r4` tmux sessions (`tmux ls`):**
  - Output: `no server running on /tmp/tmux-0/default`
  - Status: **PASS** (no `r4` session exists)
- **Target directory check (`ls /root/autodl-tmp/octorl_r4`):**
  - Contents: `agent_r1_r4.patch`, `source_backup`, `validation`
  - Contains no `seed_*` directories
  - Status: **PASS**
- **Script SHA-256 Checksums (`sha256sum`):**
  - `/root/octorl_r3/scripts/self_improve/r4_queue.sh`:
    - Full: `7616d51a8ca7b715f03ea53b591a0621ad99fab02bf094edf475f10608675b98`
    - Prefix: `7616d51a8ca7` (matches required `7616d51a8ca7`)
    - Status: **PASS**
  - `/root/octorl_r3/scripts/self_improve/r4_run_branch.py`:
    - Full: `ff46f7ec0742066ddf55578f993f98329592f993400c693db0c14bf76b757b50`
    - Prefix: `ff46f7ec0742` (matches required `ff46f7ec0742`)
    - Status: **PASS**

All preflight checks hold. Ready to launch.

---

## Launch Event (2026-09-27T15:40:11+08:00)

- **Launch Command:** `tmux new -d -s r4 'bash /root/octorl_r3/scripts/self_improve/r4_queue.sh'`
- **Tmux Session:** `r4` created.
- **Queue Log Entry:** `2026-09-27T15:40:11+08:00 START seed_42_fixed`
- **Initial Stage:** `seed_42/fixed/stage_1` initialized with `selection.json`, `stage_distribution.json`, and `train.parquet`.
- **First-Stage Target:** Updates 21..30 (`metrics_target_30.jsonl`).

---

## FIRST-STAGE CHECK (2026-09-27T16:22:40+08:00) — PASS

Verification of `seed_42/fixed/stage_1` against the four pre-registered conditions:

1. **`attempt_1/metrics_target_30.jsonl` record count and step list:**
   - Record count: exactly 10 records (`record_count: 10`)
   - Steps: `[21, 22, 23, 24, 25, 26, 27, 28, 29, 30]` (steps 21..30)
   - Status: **PASS**
2. **`attempt_1/exit.txt`:**
   - Output: `TRAIN_EXIT=0`
   - Status: **PASS**
3. **`attributed.jsonl` line count:**
   - Output: `160 /root/autodl-tmp/octorl_r4/seed_42/fixed/stage_1/attributed.jsonl` (exactly 160 lines)
   - Status: **PASS**
4. **`stage_2/` existence:**
   - Path: `/root/autodl-tmp/octorl_r4/seed_42/fixed/stage_2`
   - Attributes: `drwxr-xr-x 3 root root 117 Sep 27 16:15`
   - Status: **PASS**

**Additional Stage 1 facts:**
- Attempts: only `attempt_1/` exists (`/root/autodl-tmp/octorl_r4/seed_42/fixed/stage_1/attempt_1`); no `attempt_2` directory appeared.
- Timing: `elapsed_seconds: 2118.01` (~35.3 min), `cost_cny: 1.28257` (~¥1.28).
- Transition: Stage 2 training commenced automatically at 16:15 CST.

**Conclusion:** **FIRST-STAGE CHECK: PASS**

---

## Hourly Status Reports

| Time | Job | Stage | Last Step | Grad Norm | PG Loss | Reward Mean | Cumulative ¥ | Disk Free | Tmux Session Alive? |
|---|---|---|---|---:|---:|---:|---:|---|---|
| 2026-09-27 16:40 | `seed_42_fixed` | stage_2 | 35 | 0.1794 | -0.7338 | 0.8750 | ¥1.2826 | 11 GB | Yes |
| 2026-09-27 17:40 | `seed_42_fixed` | stage_4 | 51 | 0.1829 | -0.2973 | 0.7500 | ¥3.9961 | 11 GB | Yes |
| 2026-09-27 18:40 | `seed_42_fixed` | stage_5 | 68 | 0.1830 | -0.5684 | 0.7500 | ¥5.3512 | 11 GB | Yes |
| 2026-09-27 19:30 | `seed_42_failure_driven` | stage_1 | 22 | 0.1748 | -0.6181 | 0.8125 | ¥7.8704 | 10 GB | Yes |
| 2026-09-27 20:30 | `seed_42_failure_driven` | stage_3 | 40 | 0.2074 | -0.2673 | 0.6250 | ¥10.4422 | 9.5 GB | Yes |
| 2026-09-27 21:30 | `seed_42_failure_driven` | stage_4 | 57 | 0.1845 | -0.5284 | 0.7500 | ¥11.7139 | 9.4 GB | Yes |
| 2026-09-27 22:30 | `seed_42_failure_driven` | stage_6 | 73 | 0.2077 | -0.4565 | 0.7500 | ¥14.2796 | 9.1 GB | Yes |
| 2026-09-27 23:05 | `seed_137_fixed` | stage_1 | 22 | 0.1887 | -0.6730 | 0.9375 | ¥15.6406 | 8.9 GB | Yes |
| 2026-09-28 00:06 | `seed_137_fixed` | stage_2 | 38 | 0.1881 | -0.6192 | 0.8750 | ¥17.0114 | 8.5 GB | Yes |
| 2026-09-28 01:06 | `seed_137_fixed` | stage_4 | 52 | 0.1958 | -0.4985 | 0.8750 | ¥20.0088 | 8.3 GB | Yes |
| 2026-09-28 02:06 | `seed_137_fixed` | stage_5 | 68 | 0.2135 | -0.1999 | 0.5625 | ¥21.4591 | 8.1 GB | Yes |
| 2026-09-28 02:50 | `seed_137_failure_driven` | stage_1 | 80 | 0.1918 | -0.3079 | 0.6250 | ¥24.1400 | 7.8 GB | Yes |
| 2026-09-28 03:50 | `seed_137_failure_driven` | stage_2 | 37 | 0.2274 | -0.7028 | 0.8750 | ¥25.5344 | 7.4 GB | Yes |
| 2026-09-28 04:50 | `seed_137_failure_driven` | stage_4 | 54 | 0.2195 | -0.4774 | 0.6875 | ¥28.0691 | 7.1 GB | Yes |
| 2026-09-28 05:50 | `seed_137_failure_driven` | stage_5 | 68 | 0.1944 | -0.3124 | 0.6875 | ¥29.4658 | 7.0 GB | Yes |
| 2026-09-28 06:40 | `seed_2718_warmup` | warmup | 1 | 0.1558 | -0.1471 | 0.6250 | ¥32.3793 | 6.7 GB | Yes |
| 2026-09-28 07:34 | `seed_2718_warmup` | warmup | 14 | 0.1688 | -0.3424 | 0.7500 | ¥32.3793 | 6.7 GB | Yes |
| 2026-09-28 08:00 | `seed_2718_fixed` | stage_1 | 20 | 0.1847 | -0.5827 | 0.8125 | ¥35.3507 | 6.3 GB | Yes |
| 2026-09-28 09:00 | `seed_2718_fixed` | stage_2 | 35 | 0.1682 | -0.2495 | 0.6250 | ¥36.8221 | 5.9 GB | Yes |
| 2026-09-28 10:00 | `seed_2718_fixed` | stage_4 | 50 | 0.1661 | 0.0691 | 0.5625 | ¥39.9172 | 5.6 GB | Yes |
| 2026-09-28 11:00 | `seed_2718_fixed` | stage_5 | 64 | 0.1662 | -0.1275 | 0.6875 | ¥41.3958 | 5.5 GB | Yes |
| 2026-09-28 12:09 | `seed_2718_failure_driven` | stage_1 | 80 | 0.1930 | -0.8534 | 0.9375 | ¥44.3605 | 5.2 GB | Yes |
| 2026-09-28 12:42 | `seed_2718_failure_driven` | stage_2 | 30 | 0.1951 | -0.6856 | 0.8125 | ¥45.8216 | 4.8 GB | Yes |
| 2026-09-28 13:18 | `seed_2718_failure_driven` | stage_2 | 37 | 0.1654 | -0.1375 | 0.5625 | ¥45.8216 | 4.8 GB | Yes |
| 2026-09-28 13:44 | `seed_2718_failure_driven` | stage_3 | 43 | 0.1796 | -0.7524 | 0.8750 | ¥47.4194 | 4.7 GB | Yes |
| 2026-09-28 14:20 | `seed_2718_failure_driven` | stage_4 | 52 | 0.1788 | -0.4810 | 0.7500 | ¥48.9586 | 4.5 GB | Yes |
| 2026-09-28 14:51 | `seed_2718_failure_driven` | stage_5 | 60 | 0.1836 | -0.2597 | 0.6250 | ¥50.2673 | 4.4 GB | Yes |
| 2026-09-28 15:22 | `seed_2718_failure_driven` | stage_5 | 68 | 0.1996 | -0.7055 | 0.8125 | ¥50.2673 | 4.4 GB | Yes |
| 2026-09-28 15:47 | `seed_2718_failure_driven` | stage_6 | 74 | 0.1713 | 0.1931 | 0.3750 | ¥51.7751 | 4.3 GB | Yes |
| 2026-09-28 16:03 | `seed_2718_failure_driven` | stage_6 | 80 | 0.1936 | -0.7945 | 0.8750 | ¥53.1675 | 4.1 GB | Yes |

*Note: `seed_2718_fixed` finished all 6 stages at 12:01:13 CST (`DONE seed_2718_fixed`). `seed_2718_failure_driven` started immediately. Stage 1 completed at 12:41:31 CST (TRAIN_EXIT=0, elapsed 2412.8s, cost ¥1.4611). Stage 2 completed at 13:25:30 CST (TRAIN_EXIT=0, elapsed 2638.5s, cost ¥1.5978). Stage 3 completed at 14:06:33 CST (TRAIN_EXIT=0, elapsed 2462.6s, cost ¥1.4912); U50 dev health eval passed (FCR 68.75%, elapsed 79.3s, cost ¥0.0480). Stage 4 completed at 14:43:50 CST (TRAIN_EXIT=0, elapsed 2161.2s, cost ¥1.3087). Stage 5 completed at 15:25:24 CST (TRAIN_EXIT=0, elapsed 2489.9s, cost ¥1.5078). Stage 6 completed at 16:03:38 CST (TRAIN_EXIT=0, elapsed 2217.1s, cost ¥1.3426); U80 dev health eval passed (FCR 68.125%, elapsed 82.2s, cost ¥0.0498). Queue execution completed: ALL_DONE at 16:03:38 CST.*

---

## Branch Summary: `seed_42_fixed` (DONE at 2026-09-27T19:16:49+08:00)

- **Stages Completed:** 6 / 6 (Updates 21..80)
- **U80 Reached:** **YES**
  - Checkpoint: `/root/autodl-tmp/octorl_r4/seed_42/fixed/stage_6/attempt_1/checkpoints/global_step_80/actor/lora_adapter`
- **Wall Time:** 12,996.96s (3h 36m 37s)
- **Branch Cost:** ¥7.8704
- **Per-Stage Cell Counts (`stage_distribution.json`):**
  - `stage_1`: `constraint_violation: 15`, `missing_dependency: 13`, `stale_version: 12`
  - `stage_2`: `constraint_violation: 15`, `missing_dependency: 13`, `stale_version: 12`
  - `stage_3`: `constraint_violation: 15`, `missing_dependency: 13`, `stale_version: 12`
  - `stage_4`: `constraint_violation: 15`, `missing_dependency: 13`, `stale_version: 12`
  - `stage_5`: `constraint_violation: 15`, `missing_dependency: 13`, `stale_version: 12`
  - `stage_6`: `constraint_violation: 15`, `missing_dependency: 13`, `stale_version: 12`
- **Dev Health FCR (`fault_only_binary_fcr`, monitoring only):**
  - U50: **64.38%** (0.64375)
  - U80: **68.12%** (0.68125)

---

## Branch Summary: `seed_42_failure_driven` (DONE at 2026-09-27T22:50:42+08:00)

- **Stages Completed:** 6 / 6 (Updates 21..80)
- **U80 Reached:** **YES**
  - Checkpoint: `/root/autodl-tmp/octorl_r4/seed_42/failure_driven/stage_6/attempt_1/checkpoints/global_step_80/actor/lora_adapter`
- **Wall Time:** 12,831.47s (3h 33m 51s)
- **Branch Cost:** ¥7.7702 (Total R4 Cumulative: ¥15.6406)
- **Per-Stage Cell Counts (`stage_distribution.json`):**
  - `stage_1`: `constraint_violation: 9`, `missing_dependency: 19`, `stale_version: 12`
  - `stage_2`: `constraint_violation: 10`, `missing_dependency: 20`, `stale_version: 10`
  - `stage_3`: `constraint_violation: 9`, `missing_dependency: 22`, `stale_version: 9`
  - `stage_4`: `constraint_violation: 12`, `missing_dependency: 20`, `stale_version: 8`
  - `stage_5`: `constraint_violation: 12`, `missing_dependency: 19`, `stale_version: 9`
  - `stage_6`: `constraint_violation: 8`, `missing_dependency: 19`, `stale_version: 13`
- **Dev Health FCR (`fault_only_binary_fcr`, monitoring only):**
  - U50: **66.88%** (0.66875)
  - U80: **68.12%** (0.68125)

---

## Branch Summary: `seed_137_fixed` (DONE at 2026-09-28T02:44:39+08:00)

- **Stages Completed:** 6 / 6 (Updates 21..80)
- **U80 Reached:** **YES**
  - Checkpoint: `/root/autodl-tmp/octorl_r4/seed_137/fixed/stage_6/attempt_1/checkpoints/global_step_80/actor/lora_adapter`
- **Wall Time:** 14,035.77s (3h 53m 56s)
- **Branch Cost:** ¥8.4994 (Total R4 Cumulative: ¥24.1400)
- **Per-Stage Cell Counts (`stage_distribution.json`):**
  - `stage_1`: `constraint_violation: 15`, `missing_dependency: 13`, `stale_version: 12`
  - `stage_2`: `constraint_violation: 15`, `missing_dependency: 13`, `stale_version: 12`
  - `stage_3`: `constraint_violation: 15`, `missing_dependency: 13`, `stale_version: 12`
  - `stage_4`: `constraint_violation: 15`, `missing_dependency: 13`, `stale_version: 12`
  - `stage_5`: `constraint_violation: 15`, `missing_dependency: 13`, `stale_version: 12`
  - `stage_6`: `constraint_violation: 15`, `missing_dependency: 13`, `stale_version: 12`
- **Dev Health FCR (`fault_only_binary_fcr`, monitoring only):**
  - U50: **68.75%** (0.6875)
  - U80: **68.75%** (0.6875)

---

## Branch Summary: `seed_137_failure_driven` (DONE at 2026-09-28T06:31:26+08:00)

- **Stages Completed:** 6 / 6 (Updates 21..80)
- **U80 Reached:** **YES**
  - Checkpoint: `/root/autodl-tmp/octorl_r4/seed_137/failure_driven/stage_6/attempt_1/checkpoints/global_step_80/actor/lora_adapter`
- **Wall Time:** 13,606.11s (3h 46m 47s)
- **Branch Cost:** ¥8.2393 (Total R4 Cumulative: ¥32.3793)
- **Per-Stage Cell Counts (`stage_distribution.json`):**
  - `stage_1`: `constraint_violation: 12`, `missing_dependency: 19`, `stale_version: 9`
  - `stage_2`: `constraint_violation: 12`, `missing_dependency: 20`, `stale_version: 8`
  - `stage_3`: `constraint_violation: 10`, `missing_dependency: 20`, `stale_version: 10`
  - `stage_4`: `constraint_violation: 11`, `missing_dependency: 19`, `stale_version: 10`
  - `stage_5`: `constraint_violation: 10`, `missing_dependency: 19`, `stale_version: 11`
  - `stage_6`: `constraint_violation: 10`, `missing_dependency: 18`, `stale_version: 12`
- **Dev Health FCR (`fault_only_binary_fcr`, monitoring only):**
  - U50: **68.125%** (0.68125)
  - U80: **66.875%** (0.66875)

---

## Job Summary: `seed_2718_warmup` (DONE at 2026-09-28T07:53:14+08:00)

- **Updates Completed:** 20 / 20 (U1..U20, R3c configuration)
- **Attributed Records:** Exactly 320 records verified (`attributed.jsonl`)
- **Wall Time:** 4,907s (1h 21m 47s)
- **Job Cost:** ¥2.9715 (Total R4 Cumulative: ¥35.3507)
- **Checkpoint Output:** `/root/autodl-tmp/octorl_r4/seed_2718/warmup/checkpoints/global_step_20`

---

## Branch Summary: `seed_2718_fixed` (DONE at 2026-09-28T12:01:13+08:00)

- **Stages Completed:** 6 / 6 (Updates 21..80)
- **U80 Reached:** **YES**
  - Checkpoint: `/root/autodl-tmp/octorl_r4/seed_2718/fixed/stage_6/attempt_1/checkpoints/global_step_80/actor/lora_adapter`
- **Wall Time:** 14,878.57s (4h 07m 59s)
- **Branch Cost:** ¥9.0098 (Total R4 Cumulative: ¥44.3605)
- **Per-Stage Cell Counts (`stage_distribution.json`):**
  - `stage_1`: `constraint_violation: 15`, `missing_dependency: 13`, `stale_version: 12`
  - `stage_2`: `constraint_violation: 15`, `missing_dependency: 13`, `stale_version: 12`
  - `stage_3`: `constraint_violation: 15`, `missing_dependency: 13`, `stale_version: 12`
  - `stage_4`: `constraint_violation: 15`, `missing_dependency: 13`, `stale_version: 12`
  - `stage_5`: `constraint_violation: 15`, `missing_dependency: 13`, `stale_version: 12`
  - `stage_6`: `constraint_violation: 15`, `missing_dependency: 13`, `stale_version: 12`
- **Dev Health FCR (`fault_only_binary_fcr`, monitoring only):**
  - U50: **68.75%** (0.6875)
  - U80: **69.375%** (0.69375)

---

## Branch Summary: `seed_2718_failure_driven` (DONE at 2026-09-28T16:03:38+08:00)

- **Stages Completed:** 6 / 6 (Updates 21..80)
- **U80 Reached:** **YES**
  - Checkpoint: `/root/autodl-tmp/octorl_r4/seed_2718/failure_driven/stage_6/attempt_1/checkpoints/global_step_80/actor/lora_adapter`
- **Wall Time:** 14,543.61s (4h 02m 24s)
- **Branch Cost:** ¥8.8070 (Total R4 Cumulative: ¥53.1675)
- **Per-Stage Cell Counts (`stage_distribution.json`):**
  - `stage_1`: `constraint_violation: 12`, `missing_dependency: 17`, `stale_version: 11`
  - `stage_2`: `constraint_violation: 11`, `missing_dependency: 18`, `stale_version: 11`
  - `stage_3`: `constraint_violation: 11`, `missing_dependency: 18`, `stale_version: 11`
  - `stage_4`: `constraint_violation: 10`, `missing_dependency: 21`, `stale_version: 9`
  - `stage_5`: `constraint_violation: 11`, `missing_dependency: 20`, `stale_version: 9`
  - `stage_6`: `constraint_violation: 10`, `missing_dependency: 20`, `stale_version: 10`
- **Dev Health FCR (`fault_only_binary_fcr`, monitoring only):**
  - U50: **68.75%** (0.6875)
  - U80: **68.125%** (0.68125)

---

## Final Queue Summary: `ALL_DONE` (2026-09-28T16:03:38+08:00)

- **Total Jobs Completed:** 7 / 7 (100% success rate, 0 failed, 0 retries, all `attempt_1`)
- **Total Wall Time:** 87,799.49s (~24h 23m 19s)
- **Total Queue Cost:** ¥53.1675 (Budget Ceiling: ¥80.00, Surplus: ¥26.8325)
- **Host Disk Space Remaining:** 4.1 GB free on `/root/autodl-tmp`
- **Final LoRA Adapters (U80):**
  - `seed_42/fixed`: `/root/autodl-tmp/octorl_r4/seed_42/fixed/stage_6/attempt_1/checkpoints/global_step_80/actor/lora_adapter`
  - `seed_42/failure_driven`: `/root/autodl-tmp/octorl_r4/seed_42/failure_driven/stage_6/attempt_1/checkpoints/global_step_80/actor/lora_adapter`
  - `seed_137/fixed`: `/root/autodl-tmp/octorl_r4/seed_137/fixed/stage_6/attempt_1/checkpoints/global_step_80/actor/lora_adapter`
  - `seed_137/failure_driven`: `/root/autodl-tmp/octorl_r4/seed_137/failure_driven/stage_6/attempt_1/checkpoints/global_step_80/actor/lora_adapter`
  - `seed_2718/fixed`: `/root/autodl-tmp/octorl_r4/seed_2718/fixed/stage_6/attempt_1/checkpoints/global_step_80/actor/lora_adapter`
  - `seed_2718/failure_driven`: `/root/autodl-tmp/octorl_r4/seed_2718/failure_driven/stage_6/attempt_1/checkpoints/global_step_80/actor/lora_adapter`











