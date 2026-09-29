# R4r Training Queue Watch Log

Watcher: Claude Code, directly (user decision 2026-09-28). Host `autodl-r4`, tmux `r4r_queue`, run root
`/root/autodl-tmp/octorl_r4r`. Prereg `r4r_lr_fixed_rerun_prereg.yaml` (sha256 `46d844d8…`).
Budget: ¥80 hard ceiling for training, ¥15 per branch, at ¥2.18/h.

Every completed stage is checked by Claude Code in addition to the code gates. The checks are: logged `actor/lr`
vs the R3c reference (the check R4's watch lacked), grad norm (stop > 100), KL, reward, effective-group rate,
attempt/exit status, and cost.

## Resuming the watch in a new session

The queue runs in tmux `r4r_queue` on the remote and does not depend on any Claude Code session. To resume:

1. Read this log (the last row and Events) and `docs/progress.md` §9.
2. Status:
   `ssh autodl-r4 'R=/root/autodl-tmp/octorl_r4r; tail -3 $R/logs/queue.log; ls $R/logs; find $R -name INFRA_FAILED'`
3. Per-stage check (LR vs R3c, grad, KL, reward, allocation):
   `python3 artifacts/self_improve/r4r/watch_tools/check_stage.py SEED ARM STAGE <scratch-dir>`
   (for the warm-up: `2718 warmup 0`). Append a row to the table below.
4. Wake-up watcher (exits on a new completed stage, queue exit, infra failure, error, tmux gone, or 2 h silence):
   `bash artifacts/self_improve/r4r/watch_tools/watch_r4r.sh` (run in the background).
5. After `ALL_DONE` and all 7 `DONE_*` markers:
   `ssh autodl-r4 'tmux new-session -d -s r4r_eval "export PATH=/root/miniconda3/bin:\$PATH; python3 /root/octorl_r3/scripts/self_improve/r4r_test_eval.py --execute > /root/autodl-tmp/octorl_r4r/test_eval.log 2>&1"'`
   It refuses to run while the test is still sealed. Progress: `/root/autodl-tmp/octorl_r4r/test_eval/progress.log`.
6. Billing: the 1-day package expires 2026-09-29 22:37 and then auto-converts to pay-as-you-go. The balance is about
   ¥5.46, roughly 2.5 h.

## Pre-launch

- **2026-09-28 22:35–22:53 · GPU preflight PASS.** Resumed from R3c 137 U20: U21 LR 1.879473751206489e-05 and U22
  LR 1.863256480957574e-05, both equal to R3c (the bug would give 1.7e-07 at U22). The optimizer and lr_scheduler
  loaded from the fork. The worker marker read `total_steps=100` while the trainer target was 22. Fresh warm-up
  U1/U2 LR was 0 / 4e-06. Evidence: `preflight/` (commit `5db270a`).
- **Environment after restart:** RTX 4090 idle, 13 GB free, 52 deployed files and both Agent-R1 patch files
  matching their hashes.

## Launch

- **2026-09-28 22:54:45 CST:** `tmux new-session -d -s r4r_queue "bash /root/octorl_r3/scripts/self_improve/r4r_queue.sh --execute"`.
  The queue log shows `START seed_42_fixed`. Stage 1's selection identity gate passed, since `attempt_1` is only
  created after it.
- Order: 42 fixed → 42 FD → 137 fixed → 137 FD → 2718 warm-up → 2718 fixed → 2718 FD (36 stages + 1 warm-up).

## Stage checks

| Checked (CST) | Branch | Stage | Updates | Attempt/exit | max \|ΔLR\| vs R3c | grad max | KL mean | reward mean | eff. groups | ¥ stage / cum. | Notes |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 09-28 23:36 | 42 fixed | 1 | U21–U30 | a1 / 0 | **0.0** | 0.230 | 0.00538 | 0.838 | 0.375 | 1.35 / 1.35 | ΣLR 1.80e-4 (R4 same stage 3.98e-5, 4.5×). KL matches R3c U21–30 (0.00551; R4 0.00416). Stage-2 selection gate passed. |
| 09-29 00:07 | 42 fixed | 2 | U31–U40 | a1 / 0 | **0.0** | 0.249 | 0.01157 | 0.869 | 0.375 | 1.14 / 2.48 | **Boundary OK:** U31 LR 1.6773e-05 = R3c (R4 logged 0). ΣLR 1.56e-4 (R4 1.11e-5, 14×). KL ≈ R3c (0.01163; R4 0.00450). Stage-1 optim/extra pruned, adapter kept; 12 GB free. |
| 09-29 00:42 | 42 fixed | 3 | U41–U50 | a1 / 0 | **0.0** | 0.252 | 0.01552 | 0.825 | 0.500 | 1.31 / 3.79 | ΣLR 1.26e-4 (R4 6.8e-6, 19×). KL near R3c (0.01720; R4 0.00467). **U50 dev (monitoring only):** fault FCR 68.1%, normal 15%. Same branch R4 U50: 64.4%; R3c 42 U50: 70.6%; base 66.25%. Dev ran 73 s, ¥0.04. |
| 09-29 01:17 | 42 fixed | 4 | U51–U60 | a1 / 0 | **0.0** | 0.277 | 0.01834 | 0.844 | 0.350 | 1.20 / 5.03 | ΣLR 9.34e-5 (R4 4.6e-6, 20×). KL near R3c (0.01944). 12 GB free. |
| 09-29 01:48 | 42 fixed | 5 | U61–U70 | a1 / 0 | **0.0** | 0.267 | 0.02104 | 0.863 | 0.375 | 1.21 / 6.24 | ΣLR 6.15e-5 (R4 3.3e-6). KL ≈ R3c (0.02168). Pace 34.6 min/stage → queue ETA ≈ 20:55 on 09-29. |
| 09-29 02:23 | 42 fixed | 6 | U71–U80 | a1 / 0 | **0.0** | 0.271 | 0.02143 | 0.863 | 0.375 | 1.18 / 7.42 | **Branch DONE.** Branch ΣLR U21–80 = 6.50e-4 (= R3c; R4 was 10.5% of this). U80 dev (monitoring): fault 68.1%, normal 15% (R4 same branch U80 68.1%, R3c 42 U80 71.25%). U80 checkpoint keeps optim (final stage). 42 FD started. |
| 09-29 02:58 | 42 FD | 1 | U21–U30 | a1 / 0 | **0.0** | 0.220 | 0.00508 | 0.731 | 0.725 | 1.35 / 8.77 | selection.json byte-identical to R4 FD stage 1 (own `cmp`). Effective groups 0.725 vs fixed 0.375, as expected from up-weighting failure cells. From stage 2 on, FD selections legitimately differ from R4. 11 GB free. |
| 09-29 03:34 | 42 FD | 2 | U31–U40 | a1 / 0 | **0.0** | 0.234 | 0.00949 | 0.856 | 0.400 | 1.28 / 10.05 | Selector allocation CV/MD/SV = 10/18/12 (R4 same stage: 10/20/10); it responds to R4r's own failures. |
| 09-29 04:09 | 42 FD | 3 | U41–U50 | a1 / 0 | **0.0** | 0.255 | 0.01378 | 0.863 | 0.375 | 1.14 / 11.19 | Allocation 11/18/11. U50 dev (monitoring): fault 71.25%, normal 10% (R4 FD 42 U50 66.9%; R4r fixed 42 U50 68.1%). |
| 09-29 04:39 | 42 FD | 4 | U51–U60 | a1 / 0 | **0.0** | 0.264 | 0.01783 | 0.856 | 0.425 | 1.15 / 12.34 | Allocation 10/20/10. |
| 09-29 05:10 | 42 FD | 5 | U61–U70 | a1 / 0 | **0.0** | 0.281 | 0.01955 | 0.881 | 0.375 | 1.08 / 13.42 | Allocation 10/19/11. |
| 09-29 05:40 | 42 FD | 6 | U71–U80 | a1 / 0 | **0.0** | 0.275 | 0.02249 | 0.888 | 0.375 | 1.06 / 14.48 | **Branch DONE.** Allocation 10/18/12. U80 dev (monitoring): fault 73.1%, normal 12.5% (R4 FD 42 U80 68.1%; R4r fixed 42 U80 68.1%). Seed 42 pair complete at ¥14.48; pace 3.4 h/branch → queue ETA ≈ 20:25. 137 fixed started. |
| 09-29 06:15 | 137 fixed | 1 | U21–U30 | a1 / 0 | **0.0** | 0.217 | 0.00474 | 0.750 | 0.450 | 1.30 / 15.78 | selection.json byte-identical to R4 (own `cmp`). KL ≈ R3c 137 (0.00502; R4 0.00366). 9.7 GB free; about 4 GB still needed, above the 3 GB stop. |
| 09-29 06:51 | 137 fixed | 2 | U31–U40 | a1 / 0 | **0.0** | 0.233 | 0.00885 | 0.763 | 0.550 | 1.28 / 17.06 | KL ≈ R3c 137 (0.00950). |
| 09-29 07:21 | 137 fixed | 3 | U41–U50 | a1 / 0 | **0.0** | 0.261 | 0.01400 | 0.850 | 0.425 | 1.05 / 18.11 | KL ≈ R3c 137 (0.01270). **U50 dev (monitoring only): fault 66.25%**, the same as base (66.25%). R3c 137 U50 was 78.75%; R4 fixed 137 U50 was 68.75%. See the event entry. |
| 09-29 07:51 | 137 fixed | 4 | U51–U60 | a1 / 0 | **0.0** | 0.284 | 0.01814 | 0.900 | 0.300 | 1.18 / 19.29 | KL a little above R3c 137 (0.01567). |
| 09-29 08:32 | 137 fixed | 5 | U61–U70 | a1 / 0 | **0.0** | 0.290 | 0.01935 | 0.838 | 0.450 | 1.31 / 20.60 | Stage took 40 min (the slowest so far; still within the 90 min timeout). |
| 09-29 09:02 | 137 fixed | 6 | U71–U80 | a1 / 0 | **0.0** | 0.278 | 0.02025 | 0.869 | 0.350 | 1.16 / 21.76 | **Branch DONE (3/7).** U80 dev (monitoring): fault 68.75%, normal 10% (R3c 137 U80 76.9%; R4 fixed 137 U80 ≈ 68.1–69.4%). 137 FD started. ETA ≈ 20:25. |
| 09-29 09:37 | 137 FD | 1 | U21–U30 | a1 / 0 | **0.0** | 0.241 | 0.00466 | 0.688 | 0.725 | 1.35 / 23.11 | selection.json byte-identical to R4 FD stage 1 (own `cmp`). 8.6 GB free. |
| 09-29 10:23 | 137 FD | 2 | U31–U40 | a1 / 0 | **0.0** | 0.260 | 0.00815 | 0.713 | 0.475 | 1.50 / 24.61 | Allocation 12/19/9. **Stage took 45 min**, the longest so far (timeout 90). Average 34.5 min/stage over 20 stages → revised queue ETA ≈ 20:55 (expiry 22:37). The ~20 min of test evaluations after that would not fit comfortably. |
| 09-29 10:58 | 137 FD | 3 | U41–U50 | a1 / 0 | **0.0** | 0.260 | 0.01113 | 0.706 | 0.650 | 1.30 / 25.91 | Allocation 10/19/11. U50 dev (monitoring): fault 71.25%, normal 15% (R4r fixed 137 U50 66.25%). |
| 09-29 11:34 | 137 FD | 4 | U51–U60 | a1 / 0 | **0.0** | 0.249 | 0.01291 | 0.769 | 0.400 | 1.23 / 27.14 | Allocation 10/21/9. |
| 09-29 12:04 | 137 FD | 5 | U61–U70 | a1 / 0 | **0.0** | 0.260 | 0.01451 | 0.769 | 0.425 | 1.12 / 28.26 | Allocation 9/22/9 (MD share 55%, under the 0.6 cap). |
| 09-29 12:36 | 137 FD | 6 | U71–U80 | a1 / 0 | **0.0** | 0.278 | 0.01537 | 0.738 | 0.700 | 1.20 / 29.46 | **Branch DONE (4/7).** Allocation 8/22/10. U80 dev (monitoring): fault 73.1%, normal 10% (R4r fixed 137 U80 68.75%). The 2718 warm-up started at 12:34; its LR gate (U1–U20) and warm-up task-identity gate run when it ends. |
| 09-29 13:50 | 2718 warm-up | — | U1–U20 | a1 / 0 | **0.0** | 0.198 | 0.00116 | 0.653 | 0.750 | 2.71 / 32.17 | **DONE (5/7).** Own check: per-step task multiset over 320 rollouts equals R4's warm-up exactly, so the seen-set is unchanged. ΣLR 3.345e-4 = R3c U1–20 (R4 warm-up 2.0e-4, 20-step horizon). 74.5 min. 2718 fixed started. |
| 09-29 14:30 | 2718 fixed | 1 | U21–U30 | a1 / 0 | **0.0** | 0.213 | 0.00299 | 0.744 | 0.625 | 1.37 / 33.54 | selection.json byte-identical to R4 (own `cmp`). ΣLR 1.80e-4 (R4 same stage 2.1e-5). 7.1 GB free. |
| 09-29 15:06 | 2718 fixed | 2 | U31–U40 | a1 / 0 | **0.0** | 0.221 | 0.00573 | 0.894 | 0.325 | 1.24 / 34.78 | 6.9 GB free. |
| 09-29 15:36 | 2718 fixed | 3 | U41–U50 | a1 / 0 | **0.0** | 0.257 | 0.00966 | 0.869 | 0.350 | 1.22 / 36.00 | U50 dev (monitoring): fault 77.5%, normal 12.5% (base 66.25%; R4 fixed 2718 U50 n/a here). |
| 09-29 16:12 | 2718 fixed | 4 | U51–U60 | a1 / 0 | **0.0** | 0.266 | 0.01384 | 0.925 | **0.100** | 1.14 / 37.14 | Effective groups are low (reward 0.925, most groups all-pass). Under Sign advantage uniform groups still give gradient, and no stop rule applies (mixed > 0). Noted only. |
| 09-29 16:42 | 2718 fixed | 5 | U61–U70 | a1 / 0 | **0.0** | 0.272 | 0.01588 | 0.894 | 0.350 | 1.12 / 38.26 | 6.5 GB free. |
| 09-29 17:12 | 2718 fixed | 6 | U71–U80 | a1 / 0 | **0.0** | 0.288 | 0.01710 | 0.906 | 0.325 | 1.19 / 39.45 | **Branch DONE (6/7).** U80 dev (monitoring): fault 78.1%, normal 15%. 2718 FD, the last branch, started; ETA ≈ 20:35. |
| 09-29 17:52 | 2718 FD | 1 | U21–U30 | a1 / 0 | **0.0** | 0.211 | 0.00294 | 0.713 | 0.675 | 1.42 / 40.87 | Allocation 9/18/13 (R4: 12/17/11). This differs from R4 as the design expects: FD stage 1 reads the U1–U20 failure rates of the **re-run** warm-up (the tasks are identical, the outcomes are not). The prereg gates FD stage-1 identity only for seeds 42/137. 6.0 GB free. |
| 09-29 18:28 | 2718 FD | 2 | U31–U40 | a1 / 0 | **0.0** | 0.222 | 0.00513 | 0.781 | 0.550 | 1.23 / 42.10 | Allocation 10/19/11. 5.8 GB free. |
| 09-29 18:58 | 2718 FD | 3 | U41–U50 | a1 / 0 | **0.0** | 0.233 | 0.00885 | 0.863 | 0.350 | 1.14 / 43.24 | Allocation 10/20/10. U50 dev (monitoring): fault 75.6%, normal 12.5% (R4r fixed 2718 U50 77.5%). |
| 09-29 19:34 | 2718 FD | 4 | U51–U60 | a1 / 0 | **0.0** | 0.250 | 0.01285 | 0.825 | 0.475 | 1.24 / 44.48 | Allocation 11/19/10. Two stages left → ETA ≈ 20:45. |
| 09-29 20:09 | 2718 FD | 5 | U61–U70 | a1 / 0 | **0.0** | 0.273 | 0.01573 | 0.888 | 0.400 | 1.29 / 45.77 | Allocation 10/19/11. Last stage started. |
| 09-29 20:40 | 2718 FD | 6 | U71–U80 | a1 / 0 | **0.0** | 0.274 | 0.01746 | 0.881 | 0.350 | 1.11 / 46.86 | **Branch DONE (7/7).** Allocation 10/17/13. U80 dev (monitoring): fault 81.25%, normal 12.5%. |

## Queue completion

- **2026-09-29 ~20:38:** `ALL_DONE`, `QUEUE_EXIT=0`, all 7 `DONE_*` markers.
- **Totals** (from every `timing.json` on the remote): 37 trainer processes (36 stages + 1 warm-up), **all on
  attempt_1**, with no non-zero exit, no `INFRA_FAILED`, and no gate stop. Training cost **¥46.86** (21.49 GPU-h at
  ¥2.18) plus 12 dev-health evaluations at ¥0.52, total **¥47.38**, under the ¥80 ceiling and the ¥15/branch stop.
- **LR:** every one of the 380 logged steps (20 warm-up + 6 × 60) equals the R3c reference, with max |ΔLR| = 0.0
  in every row above.
- **Selection identity:** the fixed arm matched R4 byte for byte in all 18 stages, as did FD stage 1 for seeds 42/137.
  The 2718 warm-up task multiset equals R4's.
- **U80 dev summary** (monitoring only; not a decision input):

  | seed | fixed | failure-driven | R3c U80 |
  |---|---|---|---|
  | 42 | 68.1% | 73.1% | 71.25% |
  | 137 | 68.75% | 73.1% | 76.9% |
  | 2718 | 78.1% | 81.25% | — |

- **2026-09-29 20:39:** the evaluation queue auto-started in tmux `r4r_eval` after `ALL_DONE`
  (`r4r_test_eval.py --execute`, 16 jobs: test2 ×7, R3 test ×7, exploratory R3c U80 ×2).
- **2026-09-29 21:02:** `EVAL_ALL_DONE`, with every job at exit 0. All 16 results passed the provenance check
  (manifest sha256, instance and rollout counts, LoRA path matches branch/seed, all_finite). CRN sampling seeds are
  identical across models. Base on the R3 test reproduces R4's 63.75% exactly.
- **2026-09-29 21:05:** the frozen analysis gives primary (test2) **Q2 `Q2_failure_driven_better`** (pooled +3.07 pp,
  CI [+1.72, +4.53]) and **Q1 `no_evidence_of_improvement`**. The secondary (R3 test) gives the same verdicts. An
  independent recomputation matches. See `RESULTS.md`.

- **2026-09-29 12:55** · Billing decision (user): no 1-day renewal. The user enabled "到期转按量计费", so at 22:37 the
  instance converts to pay-as-you-go and keeps running. Caveat: the balance is ¥5.46, which is about 2.5 h at
  ¥2.18/h. The ETA is queue done ≈ 20:45 and evaluations ≈ 21:45, so the evaluations should finish before expiry
  and pay-as-you-go is only a buffer. If the queue slips past about 22:00, the user needs a small top-up before
  00:30 to avoid an arrears shutdown.
- **2026-09-29 13:12** · Claude Code session paused for a user-side update. State: 2718 warm-up at U9/20, 4/7
  branches done, ¥29.46. The watch tools were moved to `watch_tools/`, and resume steps are at the top of this log.
  The evaluation queue (`r4r_test_eval.py`, commit `0fd2f97`) is deployed and verified sealed.
- **2026-09-29 13:15** · Watch resumed after the update. Nothing was missed: the 2718 warm-up is still running,
  4/7 DONE, no INFRA_FAILED, tmux up, 7.9 GB free. The watcher was restarted from `watch_tools/`.

## Events

- **2026-09-28 23:04** · Status check: seed_42_fixed stage 1 is in progress (U21 running; GPU 95%, 11.7 GB). No
  step metrics yet. Disk 13 GB free.
- **2026-09-29 00:07** · The D3 residual risk (resuming from a scheduler state written by an R4r process) is closed on
  the real run: stage-2 U31 LR equals R3c.
- **2026-09-29 01:45** · ⚠ AutoDL console (checked through Chrome, read-only): the instance is on a **1-day subscription
  that expires 2026-09-29 22:37:56**. Account balance is ¥5.46, and a 1-day renewal costs ¥50.35, so renewal is not
  possible with the current balance. The queue ETA is about 20:30–21:30 on the 29th, a tight margin. Status at the
  check: seed_42_fixed stage 5 at U67. Asked the user to decide; nothing was purchased.
  **User decision (01:50):** no renewal now; the user will top up and renew after waking if needed. The user also
  authorised Claude Code to operate AutoDL through Chrome if anything goes wrong overnight. Scope: no purchases or
  renewals.
- **2026-09-29 07:21** · Observation for the R5 gap analysis (dev is monitoring only and not a decision input).
  R4r fixed 137 now matches R3c 137's LR curve exactly, and its training KL tracks R3c's, yet U50 dev fault FCR is
  66.25% against R3c's 78.75%. The remaining differences between the two runs are the row distribution and order
  (stratified per-stage selection, seen-last, 233/240 overlap), the per-stage process restarts (RNG and vLLM seed
  per stage), and sampling noise. Dev has 40 fault instances × 4 rollouts, so 12.5 pp is 20 of 160 rollouts.
  The LR deviation therefore probably does not explain the whole R3c-vs-R4-fixed dev gap. Recorded for R5; no
  action taken, and no change to the preregistered plan.
