#!/usr/bin/env python3
"""Derive R4r provenance and descriptive evidence from local immutable records."""
import collections
import hashlib
import json
import math
import re
from r5_diagnostics import ROOT, OUT, DIAG, RAW, FIELDS, read_jsonl, write_json

R4R = ROOT / 'artifacts/self_improve/r4r'
RECORDS = R4R / 'remote_records'
SEEDS = (42, 137, 2718)
ARMS = ('fixed', 'failure_driven')
CELLS = ('constraint_violation', 'missing_dependency', 'stale_version')
REMOTE_NOTE = 'instance powered off at finalisation; no R4r remote hash inventory exists locally (r4r/archive_remote_inventory.json covers octorl_r4, not octorl_r4r)'


def relative(path):
    return str(path.relative_to(ROOT))


def fault_breakdown(analysis):
    result = {'descriptive_only': True}
    for q in ('Q1', 'Q2'):
        cells = {}
        ids = None
        for seed in SEEDS:
            instances = analysis[f'{q}_by_seed'][str(seed)]['per_instance']
            assert ids is None or ids == set(instances)
            ids = set(instances)
            grouped = {cell: [] for cell in CELLS}
            for instance, value in instances.items():
                match = re.fullmatch(r'.+_\d+_(constraint_violation|missing_dependency|stale_version)', instance)
                assert match, instance
                grouped[match[1]].append(value)
            for cell, values in grouped.items():
                assert values
                entry = cells.setdefault(cell, {'n': len(values), 'by_seed': {}})
                assert entry['n'] == len(values)
                entry['by_seed'][str(seed)] = math.fsum(values) / len(values)
        for entry in cells.values():
            entry['pooled'] = math.fsum(entry['by_seed'].values()) / len(SEEDS)
        weighted = math.fsum(e['pooled'] * e['n'] for e in cells.values()) / sum(e['n'] for e in cells.values())
        expected = analysis[f'{q}_pooled']['difference']
        assert abs(weighted - expected) <= 1e-12
        result[q] = {'cells': cells, 'sanity': {'pooled_over_cells_weighted': weighted,
                     'analysis_pooled_difference': expected, 'matches_within_1e-12': True}}
    return result


def build_manifest():
    manifest = json.loads((OUT / 'checkpoint_manifest.json').read_text())
    manifest = [r for r in manifest if not r['name'].startswith('r4r_')]
    for row in manifest:
        if row['name'].startswith('r4_'):
            row['experiment'] = 'R4 (protocol deviation: LR horizon)'
    for directory in sorted((OUT / 'checkpoints').glob('r4r_*')):
        name = directory.name
        warm = name == 'r4r_warmup_2718_u20'
        model = name.removeprefix('r4r_').removesuffix('_u80')
        seed = int(name.split('_')[-2])
        source = R4R / 'test_eval/test2' / f'{model}.json'
        secondary = R4R / 'test_eval/test_r3' / f'{model}.json'
        evidence = []
        if warm:
            # Only literal adapter paths in the specified local records qualify.
            for p in (RECORDS / 'seed_2718/warmup/training.log', RECORDS / 'logs/queue.log'):
                for number, line in enumerate(p.read_text().splitlines(), 1):
                    for path in re.findall(r'/[^\s\'";,]+/lora_adapter', line):
                        if '/octorl_r4r/seed_2718/warmup/' in path:
                            evidence.append({'source': relative(p), 'line': number, 'path': path})
            paths = {r['path'] for r in evidence}
            assert len(paths) <= 1
            remote = next(iter(paths), None)
        else:
            remote = json.loads(source.read_text())['lora_path']
            assert remote == json.loads(secondary.read_text())['lora_path']
        row = {'name': name, 'local_path': relative(directory), 'remote_path': remote,
               'global_step': 20 if warm else 80,
               'lineage_parent': None if warm else (f'r3c_{seed}_u20' if seed != 2718 else 'r4r_warmup_2718_u20'),
               'files': {str(p.relative_to(directory)): hashlib.file_digest(p.open('rb'), 'sha256').hexdigest()
                         for p in sorted(directory.rglob('*')) if p.is_file()},
               'test_eval_json': None if warm else relative(source),
               'secondary_test_eval_json': None if warm else relative(secondary),
               'remote_sha256_verified': False, 'remote_sha256_note': REMOTE_NOTE, 'experiment': 'R4r'}
        if warm:
            row['remote_path_evidence'] = evidence
            row['remote_path_note'] = ('No literal warm-up adapter path in warmup/training.log or logs/queue.log; checkpoint parent is recorded, but the adapter path is not inferred.' if remote is None else 'Literal path recorded locally.')
        manifest.append(row)
    assert len(manifest) == len({r['name'] for r in manifest}) == 20
    write_json(OUT / 'checkpoint_manifest.json', manifest)


def deduplicate_metrics(paths):
    occurrences = collections.defaultdict(list)
    for path in paths:
        for line, row in enumerate(read_jsonl(path), 1):
            occurrences[row['step']].append({'source': relative(path), 'line': line, **row})
    return ([{'step': step, 'data': rows[-1]['data']} for step, rows in sorted(occurrences.items())],
            {str(step): rows for step, rows in sorted(occurrences.items()) if len(rows) > 1})


def lr_check(values, reference):
    return [{'step': r['step'], 'actor_lr': r['data']['actor/lr'], 'reference_lr': reference[r['step']],
             'absolute_difference': abs(r['data']['actor/lr'] - reference[r['step']])} for r in values]


def diagnostics():
    import csv
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    summaries = json.loads((DIAG / 'run_summary.json').read_text())
    attribution = json.loads((DIAG / 'attribution_summary.json').read_text())
    reference_path = RAW / 'octorl_r3c/seed_137/metrics_target_100.jsonl'
    reference = {r['step']: r['data']['actor/lr'] for r in read_jsonl(reference_path)}
    runs, duplicates, lr, exposure, resume = {}, {}, {}, {}, {}
    specs = [(f'r4r_{arm}_{seed}', RECORDS / f'seed_{seed}/{arm}', False) for seed in SEEDS for arm in ARMS]
    specs.append(('r4r_warmup_2718', RECORDS / 'seed_2718/warmup', True))
    for name, directory, warm in specs:
        metrics = sorted(directory.rglob('metrics_target_20.jsonl')) if warm else [directory / f'stage_{k}/attempt_1/metrics_target_{20+10*k}.jsonl' for k in range(1, 7)]
        assert not warm or len(metrics) == 1
        values, duplicates[name] = deduplicate_metrics(metrics)
        steps = list(range(1, 21) if warm else range(21, 81))
        assert [r['step'] for r in values] == steps
        runs[name] = values
        lr[name] = lr_check(values, reference)
        timing = sorted({k for r in values for k in r['data'] if k.startswith('timing_')})
        with (DIAG / f'{name}.csv').open('w') as f:
            writer = csv.DictWriter(f, fieldnames=['global_step', *FIELDS, *timing])
            writer.writeheader()
            for r in values:
                writer.writerow({'global_step': r['step'], **{k: r['data'].get(v, '') for k, v in FIELDS.items()}, **{k: r['data'].get(k, '') for k in timing}})
        timing_paths = [directory / 'timing.json'] if warm else [directory / f'stage_{k}/attempt_1/timing.json' for k in range(1, 7)]
        times = [json.loads(p.read_text()) for p in timing_paths]
        summaries[name] = {'steps': len(values), 'first_step': steps[0], 'last_step': steps[-1],
            'last_pg_loss': values[-1]['data']['actor/pg_loss'], 'logged_step_seconds': sum(r['data']['timing_s/step'] for r in values),
            'process_wall_seconds': sum(t['elapsed_seconds'] for t in times), 'metered_cost_cny': sum(t['cost_cny'] for t in times),
            'timing_sources': list(map(relative, timing_paths)), 'metric_sources': list(map(relative, metrics)), 'timing_fields': timing,
            'missing_fields': {k: [r['step'] for r in values if v not in r['data']] for k, v in FIELDS.items() if any(v not in r['data'] for r in values)}}
        sources = [directory / 'attributed.jsonl'] if warm else [directory / f'stage_{k}/attributed.jsonl' for k in range(1, 7)]
        rows = [r for p in sources for r in read_jsonl(p)]
        groups = collections.defaultdict(list)
        for r in rows:
            groups[r['global_step'], r['task_id']].append(r['binary_reward'])
        assert len(rows) == len(steps)*16 and len(groups) == len(steps)*4 and all(len(v) == 4 for v in groups.values())
        agreements = []
        for r in values:
            step_groups = [v for (s, _), v in groups.items() if s == r['step']]
            assert len(step_groups) == 4
            reward = sum(map(sum, step_groups))/16
            mixed = sum(len(set(v)) > 1 for v in step_groups)/4
            assert reward == r['data']['r3b/reward_mean'] and mixed == r['data']['r3b/mixed_group_fraction']
            agreements.append({'step': r['step'], 'reward_mean': reward, 'effective_group_rate': mixed, 'matches': True})
        attribution[name] = {'sources': list(map(relative, sources)), 'rollouts': len(rows), 'groups': len(groups),
            'binary_reward_mean': sum(r['binary_reward'] for r in rows)/len(rows),
            'effective_group_rate': sum(len(set(v)) > 1 for v in groups.values())/len(groups),
            'diagnostic_flags': dict(collections.Counter(flag for r in rows for flag in r['diagnostic_flags'])),
            'fault_rollout_counts': dict(collections.Counter(r['fault_type'] for r in rows)), 'per_step_agreement': agreements,
            'mask_evidence': 'Token-level mask correctness not established; token mask arrays absent.'}
        if not warm:
            exposure[name] = {}
            for k in range(1, 7):
                p = directory / f'stage_{k}/stage_distribution.json'
                d = json.loads(p.read_text())
                counts = {cell: d['cells'][cell]['count'] for cell in CELLS}
                assert sum(counts.values()) == 40
                exposure[name][str(k)] = {'counts': counts, 'total': sum(counts.values()), 'source': relative(p),
                    'probabilities': {cell: d['cells'][cell]['p_c'] for cell in CELLS}}
            p = directory / 'stage_6/attempt_1/training.log'
            saved = [{'line': n, 'text': line} for n, line in enumerate(p.read_text().splitlines(), 1) if 'Saved optim to' in line or 'Saved extra_state to' in line]
            assert len(saved) == 2 and all('global_step_80' in r['text'] for r in saved)
            resume[name] = {'source': relative(p), 'saved_state_evidence': saved, 'pulled': False}
    write_json(DIAG / 'run_summary.json', summaries)
    write_json(DIAG / 'attribution_summary.json', attribution)
    write_json(DIAG / 'r4r_duplicate_steps.json', duplicates)
    write_json(DIAG / 'r4r_lr_check.json', {'reference': relative(reference_path), 'runs': lr, 'steps_checked': sum(map(len, lr.values())), 'max_absolute_difference': max(r['absolute_difference'] for rows in lr.values() for r in rows)})
    write_json(DIAG / 'r4r_selector_exposure.json', exposure)
    write_json(DIAG / 'r4r_resume_evidence.json', resume)
    write_json(DIAG / 'r4r_fault_type_breakdown.json', {endpoint: fault_breakdown(json.loads((R4R / f'r4r_analysis_{endpoint}.json').read_text())) for endpoint in ('test2', 'test_r3')})
    for field in ('reward_mean', 'effective_group_rate', 'kl', 'entropy', 'lr'):
        fig, axes = plt.subplots(3, 1, figsize=(11, 10), sharex=True)
        for ax, seed in zip(axes, SEEDS):
            parent = runs['r4r_warmup_2718'] if seed == 2718 else read_jsonl(DIAG / 'r3c_seed42_metrics_reconstructed.jsonl' if seed == 42 else reference_path)
            ax.plot([r['step'] for r in parent], [r['data'][FIELDS[field]] for r in parent], label=f'parent {seed}', color='0.45', alpha=.7)
            for arm in ARMS:
                values = [next(r for r in parent if r['step'] == 20)] + runs[f'r4r_{arm}_{seed}']
                ax.plot([r['step'] for r in values], [r['data'][FIELDS[field]] for r in values], label=f'R4r {arm}', linewidth=1, alpha=.85)
            ax.axvline(20, color='black', linestyle=':', linewidth=.8)
            ax.set_ylabel(field); ax.set_title(f'Training seed {seed}'); ax.legend(fontsize=8); ax.grid(alpha=.2)
        axes[-1].set_xlabel('Global update (raw, unsmoothed)')
        fig.tight_layout(); fig.savefig(DIAG / f'r4r_{field}.png', dpi=150); plt.close(fig)


if __name__ == '__main__':
    build_manifest()
