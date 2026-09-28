import json
from pathlib import Path
import shutil
import pandas as pd
import pytest
from scripts.self_improve import r4_run_branch as r4
from scripts.self_improve.r4r_gates import prereg_gate, selection_gate, warmup_gate, PREREG, RECORDS
from scripts.self_improve.r4r_lr_gate import gate, REFERENCE

WORKSPACE=Path.cwd()
OLD=WORKSPACE/RECORDS
MARKER='R4R_LR_SCHEDULER total_steps=100 warmup=5 type=cosine\n'


@pytest.mark.parametrize('seed',[42,137])
def test_reference_passes(tmp_path,seed):
    log=tmp_path/'log';log.write_text(MARKER)
    metrics=REFERENCE if seed==137 else WORKSPACE/'artifacts/self_improve/r5/diagnostics/r3c_seed42_metrics_reconstructed.jsonl'
    assert gate(metrics,log,1,80)['steps']==80


def test_all_36_old_stages_and_warmup_fail_for_lr(tmp_path):
    log=tmp_path/'log'; log.write_text(MARKER)
    paths=sorted(OLD.glob('seed_*/*/stage_*/attempt_1/metrics_target_*.jsonl'))
    assert len(paths)==36
    for p in paths:
        end=int(p.stem.split('_')[-1])
        with pytest.raises(ValueError,match='LR mismatch'): gate(p,log,end-9,end)
    with pytest.raises(ValueError,match='LR mismatch'):
        gate(OLD/'seed_2718/warmup/metrics_target_20.jsonl',log,1,20)


def test_lr_missing_step_missing_marker_and_bad_reference(tmp_path):
    log=tmp_path/'log'; log.write_text(MARKER)
    metrics=tmp_path/'metrics'; metrics.write_text('\n'.join(REFERENCE.read_text().splitlines()[1:])+'\n')
    with pytest.raises(ValueError,match='missing'): gate(metrics,log,1,80)
    log.write_text('')
    with pytest.raises(ValueError,match='SCHEDULER'): gate(REFERENCE,log,1,80)
    with pytest.raises(ValueError,match='all 80'): gate(REFERENCE,log,1,80,metrics)


@pytest.mark.parametrize('seed,arm',[(s,'fixed') for s in (42,137,2718)]+[(s,'failure_driven') for s in (42,137)])
def test_offline_selection_identity_all_required_stages(tmp_path,seed,arm):
    warm=OLD/'seed_2718/warmup/attributed.jsonl' if seed==2718 else WORKSPACE/f'artifacts/self_improve/r4/warmup_records/seed_{seed}.jsonl'
    records=r4._read_jsonl(warm); sources=[(warm,records)];cursor=dict.fromkeys(r4.CELLS,0)
    for stage in range(1,7 if arm=='fixed' else 2):
        directory,cursor,_=r4.prepare_stage(seed=seed,arm=arm,stage=stage,workspace=WORKSPACE,root=tmp_path,
            record_sources=sources,cursor=cursor,seen_in_warmup={r['task_id'] for r in records})
        ref=OLD/f'seed_{seed}/{arm}/stage_{stage}'
        selection_gate(directory,ref)
        path=ref/'attributed.jsonl';sources.append((path,r4._read_jsonl(path)))
        df=pd.read_parquet(directory/'train.parquet'); df.iloc[[1,0]+list(range(2,40))].to_parquet(directory/'train.parquet',index=False)
        with pytest.raises(ValueError,match='order'): selection_gate(directory,ref)
        df.to_parquet(directory/'train.parquet',index=False)
        data=json.loads((directory/'selection.json').read_text());data['tasks'][0]='swapped_task'
        (directory/'selection.json').write_text(json.dumps(data))
        with pytest.raises(ValueError,match='bytes'): selection_gate(directory,ref)


def test_prereg_one_byte_and_record_drift(tmp_path):
    workspace=tmp_path/'workspace'; path=workspace/PREREG; path.parent.mkdir(parents=True)
    shutil.copyfile(WORKSPACE/PREREG,path)
    root=tmp_path/'run'; prereg_gate(workspace,root)
    path.write_bytes(path.read_bytes()+b' ')
    with pytest.raises(ValueError,match='hash'): prereg_gate(workspace,root)
    shutil.copyfile(WORKSPACE/PREREG,path)
    (root/'prereg_sha256.txt').write_text('changed')
    with pytest.raises(ValueError,match='launch hash'): prereg_gate(workspace,root)


def test_warmup_per_step_multiset(tmp_path):
    ref=OLD/'seed_2718/warmup/attributed.jsonl';warmup_gate(ref,ref)
    rows=r4._read_jsonl(ref);rows[0]['task_id']='swapped_task'
    bad=tmp_path/'bad';bad.write_text(''.join(json.dumps(r)+'\n' for r in rows))
    with pytest.raises(ValueError,match='multiset'): warmup_gate(bad,ref)


def test_preflight_evidence_gate(tmp_path):
    from scripts.self_improve.r4r_gates import preflight_gate,EXPECTED
    with pytest.raises(ValueError,match='has not passed'): preflight_gate(tmp_path)
    p=tmp_path/'preflight_pass.json'
    record=dict(prereg_sha256=EXPECTED,resume_steps=[21,22],warmup_steps=[1,2],optimizer_load=True,lr_scheduler_load=True,lr_gates=['PASS','PASS'])
    p.write_text(json.dumps(record));preflight_gate(tmp_path)
    record['lr_scheduler_load']=False;p.write_text(json.dumps(record))
    with pytest.raises(ValueError,match='incomplete'):preflight_gate(tmp_path)
