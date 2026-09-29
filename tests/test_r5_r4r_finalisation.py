"""CPU-only finalisation contract controls, including independent source checks."""
import csv
import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts/self_improve'))
from r5_diagnostics import read_jsonl
from r5_r4r_evidence import CELLS, REMOTE_NOTE, deduplicate_metrics, fault_breakdown, lr_check
from r5_verify_report import sections, verify

OUT = ROOT / 'artifacts/self_improve/r5'
DIAG = OUT / 'diagnostics'
R4R = ROOT / 'artifacts/self_improve/r4r'


def load(path):
    return json.loads(path.read_text())


@pytest.mark.parametrize('mutation', ['number', 'verdict', 'r4_source', 'caveat', 'exploratory', 'prose_number'])
def test_verifier_negative_controls(tmp_path, mutation):
    text = (OUT / 'R5_report.md').read_text()
    if mutation == 'number':
        lines = text.splitlines()
        i = next(i for i, line in enumerate(lines) if line.startswith('| R4r test2 Q1 pooled difference |'))
        cells = lines[i].split('|'); cells[2] = str(float(cells[2]) + 1e-9)
        lines[i] = '|'.join(cells); text = '\n'.join(lines)
    elif mutation == 'verdict':
        text = text.replace('## 3. Adaptive-mechanism evidence (Q2)', '## 3. Adaptive-mechanism evidence (Q2)\n\nno_evidence_of_difference')
    elif mutation == 'r4_source':
        row = next(line for line in text.splitlines() if '| `artifacts/self_improve/r4/r4_analysis.json` |' in line)
        text = text.replace('## 2. Capability-improvement evidence (Q1)', '## 2. Capability-improvement evidence (Q1)\n\n' + row)
    elif mutation == 'caveat':
        text = text.replace(load(R4R / 'r4r_analysis_test2.json')['caveat'], '')
    elif mutation == 'exploratory':
        lines = text.splitlines()
        i = next(i for i, line in enumerate(lines) if '/exploratory_test2/' in line and line.startswith('|'))
        lines[i] = lines[i].replace('exploratory, not pre-registered', '')
        text = '\n'.join(lines)
    else:
        value = load(R4R / 'test_eval/test2/base.json')['fault_only_binary_fcr']
        text = text.replace(f'Base fault rate = {value}', f'Base fault rate = {value + 1e-9}')
    path = tmp_path / 'report.md'; path.write_text(text)
    result = subprocess.run([sys.executable, str(ROOT / 'scripts/self_improve/r5_verify_report.py'), '--report', str(path)], capture_output=True)
    assert result.returncode != 0


def test_q1_clause_false_to_true_rejected(tmp_path):
    text = (OUT / 'R5_report.md').read_text()
    clause = next(line for line in sections(text)[2].splitlines() if line.startswith('Decision-rule clauses:'))
    assert '= False' in clause
    changed = clause.replace('= False', '= True', 1)
    path = tmp_path / 'report.md'
    path.write_text(text.replace(clause, changed, 1))
    result = subprocess.run(
        [sys.executable, str(ROOT / 'scripts/self_improve/r5_verify_report.py'), '--report', str(path)],
        capture_output=True, text=True,
    )
    assert result.returncode != 0
    assert 'Q1 decision-rule clauses differ from source JSON' in result.stderr


def test_idempotent_report():
    command = [sys.executable, str(ROOT / 'scripts/self_improve/r5_report.py')]
    subprocess.run(command, check=True)
    first = (OUT / 'R5_report.md').read_bytes()
    subprocess.run(command, check=True)
    assert first == (OUT / 'R5_report.md').read_bytes()
    assert verify()['passed']


def test_fault_types_known_values_and_weighting():
    expected_pp = {'missing_dependency': (52, 9.46), 'constraint_violation': (60, -.14), 'stale_version': (48, .17)}
    stored = load(DIAG / 'r4r_fault_type_breakdown.json')
    for endpoint in ('test2', 'test_r3'):
        analysis = load(R4R / f'r4r_analysis_{endpoint}.json')
        actual = fault_breakdown(analysis)
        assert stored[endpoint] == actual and actual['descriptive_only']
        for q in ('Q1', 'Q2'):
            weighted = 0; n = 0
            for cell, entry in actual[q]['cells'].items():
                # Independent bucketing oracle: strip family and numeric instance seed.
                means = []
                for seed in (42, 137, 2718):
                    rows = analysis[f'{q}_by_seed'][str(seed)]['per_instance']
                    values = [v for name, v in rows.items() if re.split(r'_\d+_', name, maxsplit=1)[1] == cell]
                    assert len(values) == entry['n']
                    means.append(sum(values)/len(values))
                assert entry['pooled'] == pytest.approx(sum(means)/3, abs=1e-12)
                weighted += entry['pooled']*entry['n']; n += entry['n']
            assert weighted/n == pytest.approx(analysis[f'{q}_pooled']['difference'], abs=1e-12)
    for cell, (n, pp) in expected_pp.items():
        entry = stored['test2']['Q2']['cells'][cell]
        assert entry['n'] == n and abs(100*entry['pooled'] - pp) <= .005
    md = stored['test_r3']['Q2']['cells']['missing_dependency']
    assert md['n'] == 13 and abs(100*md['pooled'] - 8.97) <= .005
    assert stored['test2']['Q2']['sanity']['pooled_over_cells_weighted'] == pytest.approx(.03072916666666667, abs=1e-12)


def test_fault_type_synthetic_null_and_equal_seed_weight():
    analysis = {}
    for q in ('Q1', 'Q2'):
        analysis[q + '_by_seed'] = {}
        for seed, delta in zip((42, 137, 2718), (-.25, 0, .25)):
            analysis[q + '_by_seed'][str(seed)] = {'per_instance': {f'family_{i}_{cell}': delta for i, cell in enumerate(CELLS)}}
        analysis[q + '_pooled'] = {'difference': 0.0}
    result = fault_breakdown(analysis)
    assert all(e['pooled'] == 0 for q in ('Q1', 'Q2') for e in result[q]['cells'].values())


def test_r4r_manifest_and_adapter_checks():
    manifest = load(OUT / 'checkpoint_manifest.json')
    old = json.loads(subprocess.check_output(['git', 'show', 'HEAD:artifacts/self_improve/r5/checkpoint_manifest.json'], cwd=ROOT))
    assert [r for r in old if r['name'].startswith('r3c_')] == [r for r in manifest if r['name'].startswith('r3c_')]
    for row in manifest:
        if row['name'].startswith('r4_'):
            assert row['experiment'] == 'R4 (protocol deviation: LR horizon)'
        if not row['name'].startswith('r4r_'): continue
        assert row['experiment'] == 'R4r'
        assert row['remote_sha256_verified'] is False and row['remote_sha256_note'] == REMOTE_NOTE
        seed = int(row['name'].split('_')[-2])
        if row['global_step'] == 80:
            assert row['remote_path'] == load(ROOT / row['test_eval_json'])['lora_path'] == load(ROOT / row['secondary_test_eval_json'])['lora_path']
            assert row['lineage_parent'] == (f'r3c_{seed}_u20' if seed != 2718 else 'r4r_warmup_2718_u20')
            arm = 'failure_driven' if 'failure_driven' in row['name'] else 'fixed'
            assert f'/seed_{seed}/{arm}/' in row['remote_path']
        else:
            assert row['lineage_parent'] is None
            assert row['remote_path'] is None and row['remote_path_note']
    checks = load(DIAG / 'adapter_checks.json')
    assert len(checks) == 20 and all(r['tensor_count'] == 504 for r in checks)


def test_lr_coverage_and_historical_negative_control():
    check = load(DIAG / 'r4r_lr_check.json')
    reference = {r['step']: r['data']['actor/lr'] for r in read_jsonl(ROOT / check['reference'])}
    assert check['steps_checked'] == 380 and check['max_absolute_difference'] == 0.0
    assert len(check['runs']) == 7
    for name, rows in check['runs'].items():
        assert [r['step'] for r in rows] == list(range(1, 21) if 'warmup' in name else range(21, 81))
        with (DIAG / f'{name}.csv').open() as f: csvrows = list(csv.DictReader(f))
        assert [float(r['lr']) for r in csvrows] == [reference[r['step']] for r in rows]
        assert all(r['actor_lr'] == reference[r['step']] and r['absolute_difference'] == 0 for r in rows)
    with (DIAG / 'r4_fixed_137.csv').open() as f: rows = list(csv.DictReader(f))
    control = lr_check([{'step': int(r['global_step']), 'data': {'actor/lr': float(r['lr'])}} for r in rows], reference)
    assert next(r for r in control if r['step'] == 22)['absolute_difference'] != 0


def test_r4r_diagnostics_and_selector():
    summary = load(DIAG / 'run_summary.json')
    attribution = load(DIAG / 'attribution_summary.json')
    exposure = load(DIAG / 'r4r_selector_exposure.json')
    for name, record in summary.items():
        if not name.startswith('r4r_'): continue
        warm = 'warmup' in name
        steps = list(range(1, 21) if warm else range(21, 81))
        with (DIAG / f'{name}.csv').open() as f: rows = list(csv.DictReader(f))
        metrics, duplicates = deduplicate_metrics([ROOT / p for p in record['metric_sources']])
        assert duplicates == load(DIAG / 'r4r_duplicate_steps.json')[name]
        assert [int(r['global_step']) for r in rows] == steps
        assert float(rows[-1]['pg_loss']) == record['last_pg_loss']
        a = attribution[name]
        assert a['rollouts'] == 16*len(steps) and a['groups'] == 4*len(steps)
        source = [r for p in a['sources'] for r in read_jsonl(ROOT / p)]
        for metric in metrics:
            by_task = {}
            for r in source:
                if r['global_step'] == metric['step']: by_task.setdefault(r['task_id'], []).append(r['binary_reward'])
            assert len(by_task) == 4 and all(len(v) == 4 for v in by_task.values())
            assert sum(map(sum, by_task.values()))/16 == metric['data']['r3b/reward_mean']
            assert sum(len(set(v)) > 1 for v in by_task.values())/4 == metric['data']['r3b/mixed_group_fraction']
        assert len(a['per_step_agreement']) == len(steps) and all(r['matches'] for r in a['per_step_agreement'])
        if not warm:
            assert set(exposure[name]) == set(map(str, range(1, 7)))
            for r in exposure[name].values():
                assert sum(r['counts'].values()) == r['total'] == 40
                assert r['counts'] == {c: v['count'] for c, v in load(ROOT / r['source'])['cells'].items()}
                if '_fixed_' in name: assert r['counts'] == dict(zip(CELLS, (15, 13, 12)))
    for metric in ('reward_mean', 'effective_group_rate', 'kl', 'entropy', 'lr'):
        assert (DIAG / f'r4r_{metric}.png').read_bytes().startswith(b'\x89PNG\r\n\x1a\n')


def test_duplicate_metrics_last_occurrence(tmp_path, monkeypatch):
    import r5_r4r_evidence as evidence
    monkeypatch.setattr(evidence, 'ROOT', tmp_path)
    p = tmp_path / 'metrics.jsonl'
    p.write_text('\n'.join(json.dumps(r) for r in [{'step': 1, 'data': {'x': 0}}, {'step': 2, 'data': {'x': 2}}, {'step': 1, 'data': {'x': 1}}]))
    rows, duplicates = deduplicate_metrics([p])
    assert rows == [{'step': 1, 'data': {'x': 1}}, {'step': 2, 'data': {'x': 2}}]
    assert [r['line'] for r in duplicates['1']] == [1, 3]


def test_no_r4_sources_in_verdict_sections():
    sec = sections((OUT / 'R5_report.md').read_text())
    assert not re.search(r'artifacts/self_improve/r4/', sec[2] + sec[3])
    assert 'no_evidence_of_difference' not in sec[2] + sec[3]


def test_protected_inputs_unchanged():
    hashes = load(DIAG / 'r4r_protected_before.json')
    # Status lines in progress.md/roadmap.md are exempt from the docs freeze: roadmap.md got its
    # R5-complete status after this snapshot, and R6 keeps updating progress.md.
    for path, expected in hashes.items():
        if path in ('docs/progress.md', 'docs/roadmap.md'):
            continue
        with (ROOT / path).open('rb') as f: assert hashlib.file_digest(f, 'sha256').hexdigest() == expected, path
