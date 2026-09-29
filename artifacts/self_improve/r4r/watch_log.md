# R4r Training Queue Watch Log

Watcher: Claude Code, directly (user decision 2026-09-28). Host `autodl-r4`, tmux `r4r_queue`, run root
`/root/autodl-tmp/octorl_r4r`. Prereg `r4r_lr_fixed_rerun_prereg.yaml` (sha256 `46d844d8…`).
Budget: ¥80 hard ceiling for training, ¥15 per branch, at ¥2.18/h.

Every completed stage is checked by Claude Code in addition to the code gates. The checks are: logged `actor/lr`
vs the R3c reference (the check R4's watch lacked), grad norm (stop > 100), KL, reward, effective-group rate,
attempt/exit status, and cost.

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

- **2026-09-29 12:55** · Billing decision (user): no 1-day renewal. The user enabled "到期转按量计费", so at 22:37 the
  instance converts to pay-as-you-go and keeps running. Caveat: the balance is ¥5.46, which is about 2.5 h at
  ¥2.18/h. The ETA is queue done ≈ 20:45 and evaluations ≈ 21:45, so the evaluations should finish before expiry
  and pay-as-you-go is only a buffer. If the queue slips past about 22:00, the user needs a small top-up before
  00:30 to avoid an arrears shutdown.

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
