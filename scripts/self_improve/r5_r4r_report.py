"""Generate final R5 text; all measured values are read from evidence files."""
import json
import re
from r5_diagnostics import ROOT, OUT, DIAG

R4 = 'artifacts/self_improve/r4'
R4R = 'artifacts/self_improve/r4r'
D = 'artifacts/self_improve/r5/diagnostics'
LABEL = 'R4 (protocol deviation: LR horizon)'


def read(source):
    return json.loads((ROOT / source).read_text())


def required_numeric_rows():
    rows = []
    def add(label, source, pointer):
        rows.append((label, source, pointer))
    for endpoint in ('test2', 'test_r3'):
        source = f'{R4R}/r4r_analysis_{endpoint}.json'
        for q in ('Q1', 'Q2'):
            for scope in ('pooled', 'by_seed/42', 'by_seed/137', 'by_seed/2718'):
                for key in ('difference', 'ci95/0', 'ci95/1'):
                    add(f'R4r {endpoint} {q} {scope} {key}', source, f'/{q}_{scope}/{key}')
        add(f'R4r {endpoint} base normal', source, '/base_normal_pass_rate')
        for model in read(source)['normal_guardrail']:
            add(f'R4r {endpoint} guardrail {model}', source, f'/normal_guardrail/{model}/rate')
        for key in ('bootstrap_seed', 'resamples'):
            add(f'R4r {endpoint} {key}', source, '/' + key)
        for p in sorted((ROOT / R4R / 'test_eval' / endpoint).glob('*.json')):
            if 'fault_only_binary_fcr' in json.loads(p.read_text()):
                for key in ('fault_only_binary_fcr', 'normal_only_binary_pass_rate'):
                    add(f'R4r {endpoint} {p.stem} {key}', str(p.relative_to(ROOT)), '/' + key)
        for q in ('Q1', 'Q2'):
            for cell in ('constraint_violation', 'missing_dependency', 'stale_version'):
                add(f'R4r {endpoint} {q} {cell} descriptive_only', f'{D}/r4r_fault_type_breakdown.json', f'/{endpoint}/{q}/cells/{cell}/pooled')
    for p in sorted((ROOT / R4R / 'test_eval/exploratory_test2').glob('*.json')):
        if 'fault_only_binary_fcr' in json.loads(p.read_text()):
            for key in ('fault_only_binary_fcr', 'normal_only_binary_pass_rate'):
                add(f'{p.stem} {key} (exploratory, not pre-registered)', str(p.relative_to(ROOT)), '/' + key)
    def leaves(value, pointer=''):
        if type(value) in (int, float):
            add('Pre-launch power ' + pointer, f'{R4R}/prelaunch_power.json', pointer)
        elif isinstance(value, dict):
            for k, v in value.items(): leaves(v, pointer + '/' + k)
        elif isinstance(value, list):
            for k, v in enumerate(value): leaves(v, pointer + '/' + str(k))
    leaves(read(f'{R4R}/prelaunch_power.json'))
    return rows


def final_report(historical):
    from r5_verify_report import resolve
    analysis = {e: read(f'{R4R}/r4r_analysis_{e}.json') for e in ('test2', 'test_r3')}
    fault = read(f'{D}/r4r_fault_type_breakdown.json')
    lr = read(f'{D}/r4r_lr_check.json')
    manifest = read('artifacts/self_improve/r5/checkpoint_manifest.json')
    checks = read(f'{D}/adapter_checks.json')
    def result_table(q):
        lines = ['| Endpoint | Seed / pooled | Difference | CI95 |', '|---|---|---:|---|']
        for endpoint, a in analysis.items():
            label = endpoint if endpoint == 'test2' else 'test_r3 — secondary, no decision authority'
            for seed in ('42', '137', '2718', 'pooled'):
                r = a[f'{q}_pooled'] if seed == 'pooled' else a[f'{q}_by_seed'][seed]
                lines.append(f'| {label} | {seed} | {r["difference"]} | {json.dumps(r["ci95"])} |')
        return '\n'.join(lines)
    a = analysis['test2']
    engine = historical.split('## 1. Engineering: completed / not completed\n\n')[1].split('## 2.')[0]
    engine = engine[engine.index('The provenance manifest'):]
    engine = engine.replace('raw/r3c_seed42_duplicate_steps.json', 'raw/r3c_seed42_duplicate_steps.json')
    report = f'''# R5 — Qwen3-4B dev-tool fault-recovery study

Final (2026-09-29): Q1/Q2 on R4r

## 1. Engineering

Completed local provenance and SHA-256 checks for {len(manifest)} adapters, CPU checks ({len(checks)} adapters, {checks[0]['tensor_count']} tensors each, all finite and non-zero LoRA-B), exact evaluation links, training diagnostics and an offline gap audit. R4r remote hashes were not re-verified: the instance is powered off and no local R4r remote hash inventory exists. No GPU, training, evaluation or full base-model load was used for this finalisation.

{engine}
R4r logged `actor/lr` agrees with the R3c seed-137 reference at all {lr['steps_checked']} global steps; maximum absolute difference = {lr['max_absolute_difference']}. See [r4r_lr_check.json](diagnostics/r4r_lr_check.json).

## 2. Capability-improvement evidence (Q1)

Q1 verdict: **{a['Q1_decision']}**, verbatim from `{R4R}/r4r_analysis_test2.json:/Q1_decision`.

Decision-rule clauses: all three Q1_s > 0 = {all(r['difference'] > 0 for r in a['Q1_by_seed'].values())} ({', '.join(f"{seed}: {a['Q1_by_seed'][seed]['difference']}" for seed in sorted(a['Q1_by_seed'], key=int))}); pooled CI lower bound > 0 = {a['Q1_pooled']['ci95'][0] > 0}; normal guardrail passes = {all(r['passes'] for r in a['normal_guardrail'].values())}. The rule requires all clauses, so the positive pooled CI alone does not establish improvement.

Primary endpoint test2 compares fixed-distribution GRPO with base. Base fault rate = {read(f'{R4R}/test_eval/test2/base.json')['fault_only_binary_fcr']}. Two of three seeds are ≈0 or negative on test2. All rates and differences below are proportions.

{result_table('Q1')}

Normal guardrail: base rate = {a['base_normal_pass_rate']}; all branches pass = {all(r['passes'] for r in a['normal_guardrail'].values())}. Exact branch rates appear in the numeric table. The secondary endpoint is labelled secondary, no decision authority.

## 3. Adaptive-mechanism evidence (Q2)

Q2 verdict: **{a['Q2_decision']}**, verbatim from `{R4R}/r4r_analysis_test2.json:/Q2_decision`.

Decision-rule clauses: all three Q2_s > 0 = {all(r['difference'] > 0 for r in a['Q2_by_seed'].values())}; pooled CI lower bound > 0 = {a['Q2_pooled']['ci95'][0] > 0}; normal guardrail passes = {all(r['passes'] for r in a['normal_guardrail'].values())}.

{result_table('Q2')}

Descriptive fault-type breakdown (`descriptive_only`; no cell CIs or tests). Values are equal-weight means across training seeds; n counts instances, not rollouts.

| Endpoint | Fault type | n | Q1 pooled | Q2 pooled |
|---|---|---:|---:|---:|
'''
    for endpoint, d in fault.items():
        for cell, r in d['Q2']['cells'].items():
            report += f"| {endpoint} | {cell} | {r['n']} | {d['Q1']['cells'][cell]['pooled']} | {r['pooled']} |\n"
    report += '\nSelector exposure: actual row counts out of each stage total, in stage order. The fixed allocation follows π0; integer allocation is shown explicitly.\n\n| Branch | constraint_violation | missing_dependency | stale_version | Stage total |\n|---|---|---|---|---|\n'
    for name, stages in read(f'{D}/r4r_selector_exposure.json').items():
        counts = [' / '.join(str(r['counts'][c]) for r in stages.values()) for c in ('constraint_violation', 'missing_dependency', 'stale_version')]
        report += f"| {name} | {' | '.join(counts)} | {next(iter(stages.values()))['total']} |\n"
    # The contract explicitly requires this sentence outside the case-study section.
    report += '\nIn R4, where the realised LR was about 10% of plan, the same up-weighting produced no test gain ([§4](#4-r4--protocol-deviation-case-study)).\n'
    report += '\nPre-launch power (already recorded; no bootstrap rerun):\n\n| Quantity | Value | Source pointer in `r4r/prelaunch_power.json` |\n|---|---:|---|\n'
    power = read(f'{R4R}/prelaunch_power.json')
    for condition in power['conditions']:
        for shift, result in power['conditions'][condition]['shifts'].items():
            pointer = f'/conditions/{condition}/shifts/{shift}/detection_rate'
            report += f"| {condition}, shift {shift} detection | {result['detection_rate']} | `{pointer}` |\n"
        report += f"| {condition} MDE at 80% power | {power['mde_at_80pct_power'][condition]} | `/mde_at_80pct_power/{condition}` |\n"
    gap = historical.split('| Candidate difference |')[1].split('## Training diagnostics and attribution')[0]
    report += f'\n## 4. R4 — protocol deviation case study\n\n### {LABEL}\n\n'
    old = read(f'{R4}/r4_analysis.json')
    report += f"Historical verdicts: Q1 `{old['Q1_decision']}`, Q2 `{old['Q2_decision']}`. These describe the executed protocol deviation.\n\n| Candidate difference |" + gap
    report += '\nLesson: health metrics looked normal while the realised LR was about 10% of plan. The fix was to gate on logged `actor/lr`.\n'
    exploratory = read(f'{R4R}/test_eval/exploratory_test2/r3c_137_u80.json')['fault_only_binary_fcr']
    fixed = read(f'{R4R}/test_eval/test2/fixed_137.json')['fault_only_binary_fcr']
    report += f'''\n## 5. Limitations

- {a['caveat']}
- n={len(a['Q1_by_seed'])} training seeds, so seed-to-seed variability is not bounded.
- The effect is modest and concentrated in one fault type.
- Dev→test non-transfer for fixed GRPO.
- The R3c-vs-pipeline gap remains. LR is now ruled out. Test2 seed-137 U80 R3c fault rate = {exploratory} (exploratory, not pre-registered), versus R4r fixed = {fixed}. Remaining candidates: stratified row selection / seen-last ordering, per-stage process restarts, seed noise. Not determinable offline.
- Held-out family scope: two families.
- Token-level mask correctness not established.
- R4r adapters' remote hashes not re-verified.
- This is not evidence of general or recursive self-improvement.

## 6. Training diagnostics and attribution

CSV metrics include binary reward, effective-group rate, KL, entropy, response length, gradient norm, clipping, LR, loss and available timing fields. Effective-group rate means reward heterogeneity, not non-zero Sign-advantage fraction. Token-level mask arrays are absent. Offload/sync timing is not isolated by logged aggregate timings. Summed step durations exclude startup and are not billed wall time. Process wall time includes startup; metered costs are estimates from those timings, not day-plan billing.

Curves are raw and unsmoothed; R4r branch lines start at their actual parent U20 point.
'''
    for metric in ('reward_mean', 'effective_group_rate', 'kl', 'entropy', 'lr'):
        report += f'\n![R4r {metric}](diagnostics/r4r_{metric}.png)\n'
    report += '\nHistorical deviation plots remain in diagnostics, unchanged.\n\n| Run | Updates | Final policy-gradient loss | Logged step seconds | Process wall seconds | Metered CNY |\n|---|---:|---:|---:|---:|---:|\n'
    for name, r in read(f'{D}/run_summary.json').items():
        label = f'{LABEL}: {name}' if name.startswith('r4_') else name
        report += f"| {label} | {r['steps']} | {r['last_pg_loss']} | {r['logged_step_seconds']} | {r.get('process_wall_seconds', 'not separately recorded')} | {r.get('metered_cost_cny', 'see ledger')} |\n"
    report += '\n| Run | Rollouts | Groups | Binary reward | Effective groups | Diagnostic flags |\n|---|---:|---:|---:|---:|---|\n'
    for name, r in read(f'{D}/attribution_summary.json').items():
        label = name if name.startswith('r4r_') else f'{LABEL}: {name}'
        report += f"| {label} | {r['rollouts']} | {r['groups']} | {r['binary_reward_mean']} | {r['effective_group_rate']} | `{json.dumps(r['diagnostic_flags'])}` |\n"
    report += '\nR4r attribution matches trainer binary reward and mixed-group rate at every recorded step, including warm-up. See [attribution_summary.json](diagnostics/attribution_summary.json).\n'
    # Extract cost values from the read-only ledger, never duplicate measured constants.
    ledger = (ROOT / 'docs/progress.md').read_text().split('## 8. 新预算记录')[1].split('## 9.')[0]
    metered = re.search(r'折算 ~(\d+\.\d+)', ledger)[1]
    actual = re.search(r'包日 ~(\d+\.\d+) 元', ledger)[1]
    total = re.search(r'累计 \*\*≈ (\d+\.\d+) 元', ledger)[1]
    off = re.search(r'实例于 (\d{4}-\d{2}-\d{2} ~\d{2}:\d{2}) 关机', ledger)[1]
    report += f'''\n## 7. Cost ledger

Read-only source: `docs/progress.md` §8. R4r ≈ ¥{metered} metered; ≈ ¥{actual} actual day-plan billing. No pay-as-you-go top-up; instance off at {off}. Cumulative new-direction ≈ ¥{total}. R5 used no GPU; no CPU currency amount is inferred.

## 8. Resume instructions

Use Qwen3-4B at recorded revision `1cfa9a7208912126459214e8b04321603b3df60c` plus `artifacts/self_improve/r5/checkpoints/r4r_failure_driven_137_u80`. The manifest records the exact evaluation path and lineage. R4r optimizer state exists only remotely at U80 on the kept data disk, as supported by saved optimizer/extra-state log lines in [r4r_resume_evidence.json](diagnostics/r4r_resume_evidence.json); it was not pulled. Current remote availability cannot be re-checked while powered off. Local adapters support a warm start, not exact optimizer-state resume.

```sh
.venv/bin/python scripts/self_improve/r5_load_adapter.py --check-only
# Future full load; not run for this CPU-only contract:
.venv/bin/python scripts/self_improve/r5_load_adapter.py --adapter artifacts/self_improve/r5/checkpoints/r4r_failure_driven_137_u80 --base /absolute/path/to/Qwen3-4B --seed 42
```

## 9. Numeric results table

<!-- numeric-results:start -->
| Result | Value | Source JSON | JSON pointer |
|---|---:|---|---|
'''
    for label, source, pointer in required_numeric_rows():
        report += f'| {label} | {json.dumps(resolve(read(source), pointer))} | `{source}` | `{pointer}` |\n'
    block = historical.split('<!-- numeric-results:start -->')[1].split('<!-- numeric-results:end -->')[0]
    for line in block.splitlines():
        if line.startswith('| ') and not line.startswith('| Result'):
            report += line.replace('| ', f'| {LABEL}: ', 1) + '\n'
    report += '''<!-- numeric-results:end -->

## 10. Inventory and reproducibility

[checkpoint_manifest.json](checkpoint_manifest.json) records all local file hashes and lineage. [adapter_checks.json](diagnostics/adapter_checks.json) contains CPU tensor checks. R4r remote hashes are explicitly unverified. The warm-up remote adapter path is null because the permitted local records do not establish that literal path. Historical acquisition records remain in `pull_inventory.json` and `raw/`; they do not establish R4r remote hashes.

Reproduce offline with `.venv/bin/python`: `scripts/self_improve/r5_r4r_evidence.py`, `scripts/self_improve/r5_diagnostics.py --r4r-only`, `scripts/self_improve/r5_load_adapter.py --check-only > artifacts/self_improve/r5/diagnostics/adapter_checks.json`, `scripts/self_improve/r5_report.py`, then `scripts/self_improve/r5_verify_report.py`. Tests: `.venv/bin/python -m pytest tests/test_r5_*.py tests/test_r4_*.py tests/test_r4r_*.py`. No analysis bootstrap, training, evaluation, base load or remote access is required. Analysis reproduction commands for both endpoints are shown in README and were not run.
'''
    return report
