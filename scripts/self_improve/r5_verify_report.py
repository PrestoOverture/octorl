#!/usr/bin/env python3
"""Assert exact source values for the R5 report's numeric results table."""
import argparse
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
REPORT=ROOT/'artifacts/self_improve/r5/R5_report.md'

def resolve(value,pointer):
    for part in pointer.strip('/').split('/'):
        part=part.replace('~1','/').replace('~0','~')
        value=value[int(part)] if isinstance(value,list) else value[part]
    return value

def verify(path=REPORT):
    text=path.read_text()
    for verdict in ('no_evidence_of_improvement','no_evidence_of_difference'):
        assert verdict in text, f'missing verdict {verdict}'
    block=text.split('<!-- numeric-results:start -->',1)[1].split('<!-- numeric-results:end -->',1)[0]
    checked=0
    for line in block.splitlines():
        if not line.startswith('| '): continue
        cells=[x.strip() for x in line.strip('|').split('|')]
        if cells[0]=='Result':continue
        label,value,source,pointer=cells
        source=source.strip('`');pointer=pointer.strip('`')
        assert source=='artifacts/self_improve/r4/r4_analysis.json' or (source.startswith('artifacts/self_improve/r4/test_eval/') and source.endswith('.json')),source
        expected=resolve(json.loads((ROOT/source).read_text()),pointer)
        assert type(expected) in (float,int), (label,'non-numeric source')
        assert json.loads(value)==expected,(label,value,expected)
        if '/exploratory_r3c/' in source: assert 'exploratory, not pre-registered' in label
        checked+=1
    assert checked>=30, 'numeric table incomplete'
    return {'passed':True,'numeric_values_checked':checked}

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--report',type=Path,default=REPORT)
    print(json.dumps(verify(p.parse_args().report),indent=2))
