#!/usr/bin/env python3
"""Verify exact numeric provenance, section authority and descriptive aggregation."""
import argparse
import json
import re
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
REPORT = ROOT / 'artifacts/self_improve/r5/R5_report.md'
LABEL = 'R4 (protocol deviation: LR horizon)'


def resolve(value, pointer):
    for part in pointer.strip('/').split('/'):
        part = part.replace('~1', '/').replace('~0', '~')
        value = value[int(part)] if isinstance(value, list) else value[part]
    return value


def sections(text):
    matches = list(re.finditer(r'^## (\d+)\. .+$', text, re.M))
    assert [int(m[1]) for m in matches] == list(range(1, 11)), 'section order'
    return {int(m[1]): text[m.end():matches[i+1].start() if i+1 < len(matches) else len(text)] for i, m in enumerate(matches)}


def allowed_source(source):
    if '..' in Path(source).parts:
        return False
    return bool(re.fullmatch(r'artifacts/self_improve/(?:r4/r4_analysis\.json|r4/test_eval/.+\.json|r4r/r4r_analysis_[^/]+\.json|r4r/test_eval/.+\.json|r4r/prelaunch_power\.json|r5/diagnostics/[^/]+\.json)', source))


def verify(path=REPORT):
    from r5_r4r_evidence import fault_breakdown
    from r5_r4r_report import required_numeric_rows
    text = path.read_text()
    sec = sections(text)
    primary = json.loads((ROOT / 'artifacts/self_improve/r4r/r4r_analysis_test2.json').read_text())
    for q, number, verdict in (('Q1', 2, 'no_evidence_of_improvement'), ('Q2', 3, 'Q2_failure_driven_better')):
        assert primary[q + '_decision'] == verdict
        assert f'{q} verdict: **{verdict}**' in sec[number], f'wrong {q} verdict or section'
    q1_seeds = primary['Q1_by_seed']
    seed_values = ', '.join(f"{seed}: {q1_seeds[seed]['difference']}" for seed in sorted(q1_seeds, key=int))
    expected_q1_clauses = (
        f"Decision-rule clauses: all three Q1_s > 0 = {all(r['difference'] > 0 for r in q1_seeds.values())} "
        f"({seed_values}); "
        f"pooled CI lower bound > 0 = {primary['Q1_pooled']['ci95'][0] > 0}; "
        f"normal guardrail passes = {all(r['passes'] for r in primary['normal_guardrail'].values())}. "
        'The rule requires all clauses, so the positive pooled CI alone does not establish improvement.'
    )
    clause_lines = [line for line in sec[2].splitlines() if line.startswith('Decision-rule clauses:')]
    assert clause_lines == [expected_q1_clauses], 'Q1 decision-rule clauses differ from source JSON'
    assert primary['caveat'] in sec[5], 'missing verbatim caveat'
    assert LABEL in sec[4], 'missing protocol deviation heading'
    for number, body in sec.items():
        if number == 4:
            continue
        for line in body.splitlines():
            if 'no_evidence_of_difference' in line or re.search(r'(?<![\w])r4/', line):
                assert number == 9 and line.startswith('| ' + LABEL + ':'), 'R4 evidence outside case study / labelled numeric row'
    block = sec[9].split('<!-- numeric-results:start -->', 1)[1].split('<!-- numeric-results:end -->', 1)[0]
    checked, seen = 0, set()
    for line in block.splitlines():
        if not line.startswith('| ') or line.startswith('| Result'):
            continue
        cells = [x.strip() for x in line.strip('|').split('|')]
        assert len(cells) == 4, 'malformed numeric row'
        label, value, source, pointer = cells
        source = source.strip('`'); pointer = pointer.strip('`')
        assert allowed_source(source), source
        expected = resolve(json.loads((ROOT / source).read_text()), pointer)
        actual = json.loads(value)
        assert type(expected) in (float, int) and type(actual) in (float, int), (label, 'non-numeric')
        assert actual == expected, (label, value, expected)
        if 'exploratory' in source or 'exploratory' in label:
            assert 'exploratory, not pre-registered' in label, 'exploratory label missing'
        if '/r4/' in source:
            assert label.startswith(LABEL + ':'), 'R4 label missing'
        assert (source, pointer) not in seen, 'duplicate numeric row'
        seen.add((source, pointer)); checked += 1
    required = {(source, pointer) for _, source, pointer in required_numeric_rows()}
    assert required <= seen, ('numeric table incomplete', required - seen)
    stored = json.loads((ROOT / 'artifacts/self_improve/r5/diagnostics/r4r_fault_type_breakdown.json').read_text())
    def compare(actual, expected):
        if isinstance(expected, dict):
            assert actual.keys() == expected.keys()
            for key in expected: compare(actual[key], expected[key])
        elif type(expected) is float:
            assert abs(actual - expected) <= 1e-12
        else:
            assert actual == expected
    for endpoint in ('test2', 'test_r3'):
        analysis = json.loads((ROOT / f'artifacts/self_improve/r4r/r4r_analysis_{endpoint}.json').read_text())
        compare(stored[endpoint], fault_breakdown(analysis))
    # Also reject drift in prose and diagnostic tables, not only the numeric block.
    from r5_report import historical_report
    from r5_r4r_report import final_report
    assert text == final_report(historical_report()), 'report differs from generated source evidence'
    return {'passed': True, 'numeric_values_checked': checked, 'fault_type_recomputed': True}


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--report', type=Path, default=REPORT)
    print(json.dumps(verify(p.parse_args().report), indent=2))
