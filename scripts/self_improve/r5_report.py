#!/usr/bin/env python3
"""Build the R5 null-result report from immutable sources and offline diagnostics."""
import json
from pathlib import Path
from r5_diagnostics import ROOT,OUT,DIAG,RAW
from r5_verify_report import resolve

def main():
    analysis='artifacts/self_improve/r4/r4_analysis.json'
    numeric=[]
    def add(label,source,pointer):
        value=resolve(json.loads((ROOT/source).read_text()),pointer)
        numeric.append(f'| {label} | {json.dumps(value)} | `{source}` | `{pointer}` |')
    for q in ('Q1','Q2'):
        add(q+' pooled difference',analysis,f'/{q}_pooled/difference')
        for i,label in enumerate(('lower','upper')):add(q+' pooled CI '+label,analysis,f'/{q}_pooled/ci95/{i}')
        for seed in (42,137,2718):
            for key in ('difference',):add(f'{q} seed {seed} {key}',analysis,f'/{q}_by_seed/{seed}/{key}')
            for i,label in enumerate(('lower','upper')):add(f'{q} seed {seed} CI {label}',analysis,f'/{q}_by_seed/{seed}/ci95/{i}')
    for p in sorted((ROOT/'artifacts/self_improve/r4/test_eval').glob('*.json')):
        d=json.loads(p.read_text())
        if 'fault_only_binary_fcr' not in d:continue
        for key in ('fault_only_binary_fcr','normal_only_binary_pass_rate'):add(p.stem+' '+key,str(p.relative_to(ROOT)),'/'+key)
    for p in sorted((ROOT/'artifacts/self_improve/r4/test_eval/exploratory_r3c').glob('*.json')):
        d=json.loads(p.read_text())
        if 'fault_only_binary_fcr' not in d:continue
        for key in ('fault_only_binary_fcr','normal_only_binary_pass_rate'):add(p.stem+' '+key+' (exploratory, not pre-registered)',str(p.relative_to(ROOT)),'/'+key)
    for condition in ('independent','coupled_q_lo','coupled_q_hi'):
        for shift in ('0.00','0.05','0.10'):add(f'Power {condition} shift {shift}',analysis,f'/synthetic_power/conditions/{condition}/shifts/{shift}/detection_rate')
    add('80%-power MDE, coupled_q_lo',analysis,'/synthetic_power/mde_at_80pct_power/coupled_q_lo')
    for key in ('bootstrap_seed','resamples'):add(key,analysis,'/'+key)
    add('power simulation seed',analysis,'/synthetic_power/seed')
    add('power simulation repetitions',analysis,'/synthetic_power/reps')
    control=json.loads((DIAG/'parser_control.json').read_text())
    inventory=json.loads((OUT/'pull_inventory.json').read_text())
    summary=json.loads((DIAG/'run_summary.json').read_text())
    lr=json.loads((DIAG/'lr_comparison.json').read_text())
    u21=json.loads((DIAG/'u21_comparison.json').read_text())
    attrib=json.loads((DIAG/'attribution_summary.json').read_text())
    report='''# R5 — Qwen3-4B dev-tool fault-recovery study

## 1. Engineering: completed / not completed

Completed: read-only key-authenticated retrieval, remote/local SHA-256 integrity checks for all 13 LoRA adapters, provenance and exact test-evaluation path links, CPU tensor checks, nine per-run CSVs, raw unsmoothed curves, seed-42 reconstruction after the seed-137 parser control, and an offline gap audit. These are engineering checks, not capability evidence. No GPU job, training, evaluation, or full model loading was performed for R5. Full base+PEFT loading and short generation are implemented but not run.

The provenance manifest is [checkpoint_manifest.json](checkpoint_manifest.json). Weights are ignored by Git. All configs specify rank 16 and alpha 32. Their target-module regex was serialized as a character list; the checker joins it in memory, while full-load mode uses a temporary corrected config. Archived files remain byte-identical. The adapter revision field is consistently null: that field alone cannot establish the base revision. Linked evaluation records identify Qwen3-4B revision `1cfa9a7208912126459214e8b04321603b3df60c`; fork-only adapters rely on recorded lineage and the frozen training contract. R5 did not copy or re-hash base weights.

Parser control: CONTROL shared values across all 80 steps agree within `1e-6 * max(1, abs(source))`; no JSONL keys are absent from the log. It ran before reconstruction. Seed 42 contains 84 occurrences: steps 41, 42, 43, and 44 each occur twice. Last occurrence in file order wins. [All duplicate occurrences](raw/r3c_seed42_duplicate_steps.json) retain source line numbers. Reconstructed steps are exactly 1–80. Logged positive checkpoint-save timing at every pruning-record step and U80 agrees with `latest_checkpointed_iteration.txt = 80`; pruning records do not provide an independent reward/loss oracle. See [parser control](diagnostics/parser_control.json) and [reconstruction check](diagnostics/reconstruction_check.json).

## 2. Capability-improvement evidence

Q1 verdict: **no_evidence_of_improvement**.

The pre-registered comparison is R4 fixed versus base. The pooled test fault-rate difference is about +0.42 percentage points, with a 95% interval from 0 to about +1.04 points. This is a null result under the frozen decision rule. Exact source values and per-seed results are in the numeric table below. Normal guardrails passed for every R4 branch. The test set is narrow and outcomes are highly polarized; this result does not establish general equivalence.

## 3. Adaptive-mechanism evidence

Q2 verdict: **no_evidence_of_difference**.

The pre-registered comparison is failure-driven versus fixed at matched updates. The pooled difference is about −0.42 percentage points, with a 95% interval from about −1.04 to 0 points. Selection changed training fault exposure but supplies no measured additional test benefit here.

The pre-registered power procedure tests shifts of +5 and +10 percentage points. At +10 points, detection is 33.3% under independent sampling, 86.9% under low-discordance coupling, and 64.0% under high-discordance coupling. The 80%-detection MDE is +10 points for low-discordance coupling; it exceeds the tested +10-point range for the other two models. This is not a uniformly powered equivalence test. The already-recorded synthetic run used seed 20260927, 1,000 repetitions, and 10,000 bootstrap resamples; it was not rerun for R5. The observed CI covers only instance and eval-sampling uncertainty. Training-seed variance is not estimated (n=3).

## Cost ledger

Read-only source: `docs/progress.md`, §8. Historical values are approximate: R2 ¥0.21; R3 ¥2.94; R3b ¥2.98; R3c seed 42 ¥15.26 and seed 137 ¥15.0; R4 about ¥54.6 (queue ¥53.17, load-only checks ¥0.89, pre-registered test ¥0.31, exploratory R3c test, not pre-registered, ¥0.18). Recorded total is about ¥91.0, excluding powered-on idle time. The historical GPU rate was ¥2.18/hour. R5 used no GPU; no CPU billing rate was supplied, so no R5 currency amount is inferred.

REMOTE_TIME

## Limitations and offline gap investigation

R3c test results are **exploratory, not pre-registered**. Their small test differences do not reproduce the larger dev differences. Dev→test non-transfer, held-out family scope, the R3c versus R4-fixed gap, and the unestimated training-seed variance limit interpretation. Seed noise cannot be ruled out offline. Candidate differences below are descriptive; none is asserted to cause the test gap. The numeric table labels every R3c test result exploratory, not pre-registered.

| Candidate difference | Label | Evidence and interpretation |
|---|---|---|
| 1. Resolved configuration | PRESENT | [resolved_config_diff.json](diagnostics/resolved_config_diff.json), stage records 1–6, includes exact source line spans. Each stage has 13 differing flattened keys: checkpoint contents, adapter/resume paths, data shuffle/file, output names/paths, total epochs and trainer step horizon. Actor hyperparameters in the top-of-log dump otherwise agree. |
| 2. Per-step LR / cosine horizon | PRESENT | [lr_comparison.json](diagnostics/lr_comparison.json), U21–U80. U21 agrees; U22 is 1.863256480957574e-5 for R3c versus 5.742207084349274e-6 for R4. Later R4 stage starts U31/U41/U51/U61/U71 log zero. Remote veRL `trainer/ppo/ray_trainer.py:422–436` overwrites actor optimizer total steps with the stage target; `workers/fsdp_workers.py:930–932` logs LR before scheduler advancement; `utils/torch_functional.py:725–732` closes over the newly constructed horizon. Numbered, hashed excerpts are in [scheduler_source_evidence.json](raw/scheduler_source_evidence.json). R4 re-creates a stage-horizon cosine function while restoring scheduler progress; this is not the uninterrupted 100-step cosine. [scheduler_formula_check.json](diagnostics/scheduler_formula_check.json) reproduces all 60 observed LR values to absolute tolerance 1e-15 from that formula. |
| 2a. Five-step warmup restarted from zero each stage | ABSENT | Same LR record: no new five-step ramp. The first stage update inherits saved LR, then the stage-horizon cosine takes effect. Restore logs record `Loaded lr_scheduler`; see the per-stage restore evidence below. |
| 3. Optimizer discarded between R4 stages | ABSENT | `scripts/self_improve/r4_train_stage.sh:30–40` requires and links optimizer/extra files; `r4_run_branch.py`, `stage_command` and `execute`, pass the previous stage checkpoint and prune it only after the next stage succeeds. [offline_remote_evidence.json](raw/offline_remote_evidence.json), `resume_views`, preserves actual symlink targets; per-stage training logs record optimizer and scheduler loads. `agent_r1_r4.patch` adds only the load-only validation branch. This supports carry-over, not a fresh optimizer each stage. |
| 3a. Exact historical Adam moments at every boundary | NOT DETERMINABLE OFFLINE | Superseded R4 optimizer files were pruned. Restore logs establish load events but cannot reconstruct every historical moment tensor. Existing R4-A3 gate records check the initial fork. |
| 4. Training-row distribution, IDs and ordering | PRESENT | [row_distribution.json](diagnostics/row_distribution.json), R3c and R4-fixed seed 137 U21–U80: each has 240 unique rows, intersection 233. Fault shares differ; full family shares, R3c dump-observed group order and R4 selection order are retained; original R3c within-batch dataloader order is not independently established. R3c uses a shuffled full-pool epoch; R4 uses per-stage stratification and unseen-before-seen cell order followed by a stage permutation (`src/curriculum/failure_driven.py:106–146`). R4 reuses 7 warmup instances, R3c reuses none. R3c numbered dumps match the trajectory ID/reward multiset at all 80 steps. |
| 5. First post-fork U21 metrics | PRESENT | [u21_comparison.json](diagnostics/u21_comparison.json), all metric fields for both runs; reward, entropy, KL and gradient norm differ, while LR agrees. U21 data batches differ, so these are not paired-input comparisons. |
| Seed-noise contribution to the gap | NOT DETERMINABLE OFFLINE | No new training/evaluation or seed-variance experiment is authorized. Seed noise cannot be ruled out offline. |

RESTORE_EVIDENCE

U21_METRICS

The LR difference is visible in all stages and merits attention in any future contract. No existing experiment or conclusion has been edited to account for it retrospectively.

## Training diagnostics and attribution

Nine CSVs cover both R3c runs (U1–U80), six R4 branches (U21–U80), and the 2718 warm-up (U1–U20). Each contains reward, effective-group rate, KL, entropy, response length, gradient norm, clipping, LR, loss, and every available timing field. [Metric definitions](diagnostics/metric_definitions.json) identify exact source keys. Reward uses the trajectory-level binary field. Effective-group rate means reward heterogeneity, **not** non-zero Sign-advantage fraction: Sign advantage stays ±1 even for uniform-reward groups.

![Raw reward](diagnostics/reward_mean.png)
![Raw effective-group rate](diagnostics/effective_group_rate.png)
![Raw KL](diagnostics/kl.png)
![Raw entropy](diagnostics/entropy.png)
![Raw LR](diagnostics/lr.png)

Curves are raw and unsmoothed. R4 lines start at their actual parent U20 point. The 2718 warm-up also has a shorter LR horizon than the R3c parent runs, visible directly in its CSV and LR curve; cross-seed comparisons therefore do not isolate seed noise. Attributed records contain binary reward, task/step attribution and diagnostic flags, but no token-level mask arrays. Reward means and mixed-group rates agree with the trainer metrics for all 60 steps of every branch. This checks reward attribution; token-level mask correctness is not determinable from `attributed.jsonl`. See [attribution_summary.json](diagnostics/attribution_summary.json).

ATTRIBUTION_TABLE

Offload/sync timing is not isolated by the logged fields, so no causal timing attribution is made. `timing_s/gen`, `timing_s/update_actor`, and total step durations are retained, but are not labeled as pure synchronization or offload time. Summed step timings exclude startup and are not billed wall time.

RUN_TABLE

## Resume instructions

Use base Qwen3-4B at recorded revision `1cfa9a7208912126459214e8b04321603b3df60c` plus one local adapter, for example the actual warm-start directory `artifacts/self_improve/r5/checkpoints/r4_fixed_137_u80/`. Its source is `/root/autodl-tmp/octorl_r4/seed_137/fixed/stage_6/attempt_1/checkpoints/global_step_80/actor/lora_adapter`. Fork parents are `r3c_42_u20`, `r3c_137_u20`, and `r4_warmup_2718_u20`, as recorded in the manifest.

Full optimizer state was pruned from superseded R4 stage checkpoints. The R5 archive contains adapters only and cannot perform an exact optimizer-state resume. The read-only remote inventory still lists optimizer/extra files at final U80 and several R3c checkpoints; do not confuse intermediate pruning with absence of every remote optimizer file. An adapter warm start requires a new optimizer/scheduler and separate authorization for future training. The full-load example below is implemented but was **not run** for this contract:

```sh
.venv/bin/python scripts/self_improve/r5_load_adapter.py --check-only
# Future explicit full-load use; needs local base, torch, transformers and peft:
.venv/bin/python scripts/self_improve/r5_load_adapter.py --adapter artifacts/self_improve/r5/checkpoints/r4_fixed_137_u80 --base /absolute/path/to/Qwen3-4B --seed 42
```

## Numeric results (exact source values; rates are proportions)

<!-- numeric-results:start -->
| Result | Value | Source JSON | JSON pointer |
|---|---:|---|---|
NUMERIC_TABLE
<!-- numeric-results:end -->

Run `scripts/self_improve/r5_verify_report.py` to reject numerical drift. It reads only the existing R4 analysis and evaluation files.

## Inventory and reproducibility

[pull_inventory.json](pull_inventory.json) lists every remote/local file, byte size and SHA-256, including missing requested paths. [offline_remote_evidence.json](raw/offline_remote_evidence.json) stores projected numbered rollout records, source hashes and actual resume symlink targets. [scheduler_source_evidence.json](raw/scheduler_source_evidence.json) preserves source line numbers and hashes. No remote file was changed.

MISSING

Local reproduction (CPU; no training/evaluation): run `r5_diagnostics.py`, `r5_gap.py`, `r5_load_adapter.py --check-only`, `r5_report.py`, `r5_verify_report.py`, then `pytest tests/test_r5_*.py tests/test_r4_*.py`. Reacquisition scripts require key-only SSH to `autodl-r4`; they use read-only remote access. All new generated outputs stay under R5. Full-load dependencies are optional for check-only mode, which needs NumPy and the Python standard library.
'''
    restore=[]
    for stage in range(1,7):
        p=RAW/f'octorl_r4/seed_137/fixed/stage_{stage}/attempt_1/training.log'
        lines=[f'{i}: {line}' for i,line in enumerate(p.read_text().splitlines(),1) if 'Loaded optimizer from' in line or 'Loaded lr_scheduler from' in line]
        restore.append(f'- Stage {stage}: `{p.relative_to(ROOT)}`, lines '+', '.join(x.split(':',1)[0] for x in lines)+'.')
    metrics=['| U21 metric | R3c 137 | R4 fixed 137 |','|---|---:|---:|']
    for key in ('r3b/reward_mean','r3b/mixed_group_fraction','actor/kl_loss','actor/entropy','actor/grad_norm','actor/lr'):
        metrics.append(f'| `{key}` | {u21["r3c"][key]} | {u21["r4_fixed"][key]} |')
    at=['| Branch | Rollouts / groups | Binary reward | Effective groups | Diagnostic flags |','|---|---:|---:|---:|---|']
    for name,r in attrib.items():at.append(f'| {name} | {r["rollouts"]} / {r["groups"]} | {r["binary_reward_mean"]:.6f} | {r["effective_group_rate"]:.6f} | `{json.dumps(r["diagnostic_flags"])}` |')
    runs=['| Run | Completed steps | Final policy-gradient loss | Logged step seconds |','|---|---:|---:|---:|']
    for name,r in summary.items():runs.append(f'| {name} | {r["steps"]} | {r["last_pg_loss"]} | {r["logged_step_seconds"]:.3f} |')
    wall=inventory.get('remote_session_wall_seconds')
    repl={'CONTROL':str(control['shared_values_checked']),'NUMERIC_TABLE':'\n'.join(numeric),'RESTORE_EVIDENCE':'\n'.join(restore),'U21_METRICS':'\n'.join(metrics),'ATTRIBUTION_TABLE':'\n'.join(at),'RUN_TABLE':'\n'.join(runs),'MISSING':'Missing requested paths: '+ ('; '.join('`'+p+'`' for p in inventory['missing']) or 'none.') ,'REMOTE_TIME':f'R5 acquisition-session elapsed wall time: {wall:.3f} seconds, including transfers, retries and hash checks. The separate compact remote evidence query took '+str(json.loads((RAW/'offline_remote_evidence.json').read_text())['remote_query_wall_seconds'])+' seconds. These are CPU/network session times, not GPU hours.' if wall is not None else 'Acquisition still running; elapsed wall time not final.'}
    for key,value in repl.items():report=report.replace(key,value)
    (OUT/'R5_report.md').write_text(report)

if __name__=='__main__':main()
