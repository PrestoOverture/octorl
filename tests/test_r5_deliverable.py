"""R5 controls and artifact checks; CPU only, no model/base-weight load."""
import csv
import hashlib
import json
import struct
import subprocess
import sys
from pathlib import Path
import pytest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts/self_improve'))
from r5_diagnostics import parse_log,compare_control,read_jsonl
from r5_load_adapter import check_all,check_adapter
from r5_verify_report import verify

OUT=ROOT/'artifacts/self_improve/r5'

def test_manifest_integrity_and_eval_links():
    manifest=json.loads((OUT/'checkpoint_manifest.json').read_text())
    assert len(manifest)==13 and len({r['name'] for r in manifest})==13
    for row in manifest:
        directory=ROOT/row['local_path']
        assert {str(p.relative_to(directory)) for p in directory.rglob('*') if p.is_file()}==set(row['files'])
        for relative,digest in row['files'].items():
            with (directory/relative).open('rb') as f: assert hashlib.file_digest(f,'sha256').hexdigest()==digest
        if row['test_eval_json']:
            assert json.loads((ROOT/row['test_eval_json']).read_text())['lora_path']==row['remote_path']
        else: assert row['global_step']==20
        assert subprocess.run(['git','check-ignore','-q',str(directory/'adapter_model.safetensors')],cwd=ROOT).returncode==0

def test_all_adapters():
    results=check_all(OUT/'checkpoint_manifest.json')
    assert len(results)==13 and all(r['tensor_count']==504 for r in results)

def test_adapter_negative_controls(tmp_path):
    # Each control modifies a temporary copy of a real archived adapter.
    source=OUT/'checkpoints/r4_fixed_42_u80'
    (tmp_path/'adapter_config.json').write_bytes((source/'adapter_config.json').read_bytes())
    original=(source/'adapter_model.safetensors').read_bytes()
    target=tmp_path/'adapter_model.safetensors'
    target.write_bytes(original[:-1])
    command=[sys.executable,str(ROOT/'scripts/self_improve/r5_load_adapter.py'),'--check-only','--adapter',str(tmp_path)]
    assert subprocess.run(command,capture_output=True).returncode!=0
    blob=bytearray(original);size=struct.unpack('<Q',blob[:8])[0];header=json.loads(blob[8:8+size])
    key=next(k for k in header if '.lora_B.' in k);start,end=header[key]['data_offsets']
    blob[8+size+start:8+size+end]=bytes(end-start);target.write_bytes(blob)
    result=subprocess.run(command,capture_output=True,text=True)
    assert result.returncode!=0 and 'zero lora_B' in result.stderr

def test_parser_control_and_reconstruction():
    raw=OUT/'raw/octorl_r3c'
    control=compare_control(raw/'seed_137_train.log',raw/'seed_137/metrics_target_100.jsonl')
    assert control['passed'] and not control['jsonl_keys_absent_from_log']
    parsed,dups=parse_log(raw/'seed_42_train.log')
    assert parsed==read_jsonl(OUT/'diagnostics/r3c_seed42_metrics_reconstructed.jsonl')
    assert [r['step'] for r in parsed]==list(range(1,81))
    assert dups==json.loads((OUT/'raw/r3c_seed42_duplicate_steps.json').read_text())
    assert set(dups)=={'41','42','43','44'}
    for step,rows in dups.items():assert parsed[int(step)-1]['data']==rows[-1]['data']
    assert int((raw/'seed_42/checkpoints/latest_checkpointed_iteration.txt').read_text())==80
    for row in read_jsonl(raw/'seed_42/checkpoint_pruning.jsonl'):
        assert parsed[row['step']-1]['data']['timing_s/save_checkpoint']>0

def test_synthetic_parser_last_occurrence(tmp_path):
    p=tmp_path/'log'
    p.write_text('\x1b[36m(TaskRunner pid=1)\x1b[0m step:1 - reward:0.0 - tag:old\nstep:2 - reward:1e-3 - tag:second\nstep:1 - reward:1.0 - tag:new\n')
    rows,dups=parse_log(p)
    assert rows==[{'step':1,'data':{'reward':1.0,'tag':'new'}},{'step':2,'data':{'reward':.001,'tag':'second'}}]
    assert [r['line'] for r in dups['1']]==[1,3]

@pytest.mark.parametrize('name,start,end',[(f'r3c_{s}',1,80) for s in (42,137)]+[(f'r4_{a}_{s}',21,80) for s in (42,137,2718) for a in ('fixed','failure_driven')]+[('r4_warmup_2718',1,20)])
def test_csv_step_coverage(name,start,end):
    with (OUT/f'diagnostics/{name}.csv').open() as f:rows=list(csv.DictReader(f))
    assert [int(r['global_step']) for r in rows]==list(range(start,end+1))
    for row in rows:
        assert 0<=float(row['effective_group_rate'])<=1
        assert all(row[k]!='' for k in ('reward_mean','kl','entropy','lr','grad_norm','clip_fraction','step_seconds'))

def test_plots_and_report():
    for metric in ('reward_mean','effective_group_rate','kl','entropy','lr'):
        assert (OUT/f'diagnostics/{metric}.png').read_bytes().startswith(b'\x89PNG\r\n\x1a\n')
    assert verify()['passed']

def test_report_drift_rejected(tmp_path):
    original=(OUT/'R5_report.md').read_text()
    lines=original.splitlines()
    i=next(i for i,line in enumerate(lines) if line.startswith('| Q1 pooled difference |'))
    cells=lines[i].split('|');cells[2]=' 0.999 ';lines[i]='|'.join(cells)
    p=tmp_path/'report.md';p.write_text('\n'.join(lines))
    with pytest.raises(AssertionError):verify(p)

def test_config_and_scheduler_evidence():
    from r5_gap import config_dump
    source=OUT/'raw/octorl_r3c/seed_137_train.log'
    config,start,end=config_dump(source)
    assert start==5 and end>500 and config['data']['shuffle'] is True
    diffs=json.loads((OUT/'diagnostics/resolved_config_diff.json').read_text())
    assert set(diffs)=={str(s) for s in range(1,7)}
    assert all(row['differences']['data.shuffle']=={'r3c':True,'r4':False} for row in diffs.values())
    check=json.loads((OUT/'diagnostics/scheduler_formula_check.json').read_text())
    assert check['all_60_steps_match'] and len(check['predictions'])==60
    assert all(abs(r['predicted_r4_lr']-r['observed_r4_lr'])<1e-15 for r in check['predictions'])

def test_attribution_and_distribution():
    from collections import Counter
    d=json.loads((OUT/'diagnostics/row_distribution.json').read_text())
    assert len(d['r3c_ordered_tasks'])==240
    selected=[t for tasks in d['r4_stage_ordered_tasks'].values() for t in tasks]
    assert len(set(selected)&set(d['r3c_ordered_tasks']))==d['instance_overlap']==233
    a=json.loads((OUT/'diagnostics/attribution_summary.json').read_text())
    assert len(a)==6
    for record in a.values():
        rows=[r for src in record['sources'] for r in read_jsonl(ROOT/src)]
        assert len(rows)==record['rollouts']==960
        assert sum(r['binary_reward'] for r in rows)/960==record['binary_reward_mean']
        assert Counter(r['fault_type'] for r in rows)==record['fault_rollout_counts']

def test_parser_control_rejects_numeric_mismatch(tmp_path):
    log=tmp_path/'log';reference=tmp_path/'metrics.jsonl'
    log.write_text('step:1 - reward:0.0\n')
    reference.write_text('{"step":1,"data":{"reward":0.001}}\n')
    with pytest.raises(AssertionError,match='control mismatch'):
        compare_control(log,reference)

@pytest.mark.parametrize('dtype,width,nan_value', [('F32',4,0x7fc00000),('BF16',2,0x7fc0)])
def test_safetensors_nonfinite_control(tmp_path,dtype,width,nan_value):
    from r5_load_adapter import read_tensors
    header=json.dumps({'module.lora_B.weight':{'dtype':dtype,'shape':[1],'data_offsets':[0,width]}}).encode()
    p=tmp_path/'tensor.safetensors'
    p.write_bytes(struct.pack('<Q',len(header))+header+nan_value.to_bytes(width,'little'))
    with pytest.raises(ValueError,match='NaN/Inf'):read_tensors(p)
