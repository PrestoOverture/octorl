"""Exercise the production queue and stage gates with a CPU trainer stub."""
import json
from pathlib import Path
import shutil
import subprocess
import sys
import pytest
from scripts.self_improve import r4r_run_branch as runner
from scripts.self_improve.r4r_gates import RECORDS
from scripts.self_improve.r4r_lr_gate import read_lrs,REFERENCE

WS=Path.cwd()


@pytest.fixture
def stub(monkeypatch):
    calls=[]
    monkeypatch.setattr(runner,"preflight_gate",lambda root:None)
    monkeypatch.setattr(runner.r4,'run_dev_health',lambda **kw:None)
    def train(command,*,log_path,max_seconds,env=None):
        calls.append(command)
        directory=log_path.parent
        is_warm='r4r_warmup.sh' in command[1]
        start=0 if is_warm else int(command[5]);end=20 if is_warm else start+10
        log_path.write_text('R4R_LR_SCHEDULER total_steps=100 warmup=5 type=cosine\n')
        ref=read_lrs(REFERENCE)
        (directory/f'metrics_target_{end}.jsonl').write_text(''.join(json.dumps({'step':s,'data':{
            'actor/lr':ref[s],'actor/pg_loss':.1,'actor/grad_norm':.2}})+'\n' for s in range(start+1,end+1)))
        actor=directory/f'checkpoints/global_step_{end}/actor';(actor/'lora_adapter').mkdir(parents=True)
        for n in ('optim_world_size_1_rank_0.pt','extra_state_world_size_1_rank_0.pt','lora_adapter/adapter_model.safetensors','lora_adapter/adapter_config.json'):
            (actor/n).write_text('CPU stub')
        if is_warm:
            shutil.copyfile(WS/RECORDS/'seed_2718/warmup/attributed.jsonl',directory/'attributed.jsonl')
        else:
            tasks=json.loads((directory.parent/'selection.json').read_text())['tasks']
            trajectories=[];(directory/'rollouts').mkdir()
            for offset,step in enumerate(range(start+1,end+1)):
                dumped=[]
                for task in tasks[offset*4:offset*4+4]:
                    for _ in range(4):
                        trajectories.append({'task_id':task,'instance_seed':0,'binary_reward':0,'diagnostic_flags':[],
                            'run_id':'stub','seed':int(command[2]),'arm':command[3],'stage':int(command[4])})
                        dumped.append({'gts':task,'score':0})
                (directory/f'rollouts/{step}.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in dumped))
            (directory/'trajectories.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in trajectories))
        return 0,.01
    monkeypatch.setattr(runner.r4,'_run_bounded',train)
    return calls


def test_queue_success_skip_done_and_half_finished(tmp_path,stub):
    runner.queue(workspace=WS,root=tmp_path)
    assert len(stub)==37
    assert len(list((tmp_path/'logs').glob('DONE_*')))==7
    runner.queue(workspace=WS,root=tmp_path);assert len(stub)==37
    (tmp_path/'logs/DONE_seed_42_fixed').unlink()
    with pytest.raises(RuntimeError,match='half-finished'):runner.queue(workspace=WS,root=tmp_path)
    assert len(stub)==37


@pytest.mark.parametrize('failure',['lr','selection','prereg','warmup','numerical','infra','budget','preflight'])
def test_queue_fail_stop_on_every_gate(tmp_path,monkeypatch,stub,failure):
    def fail(*a,**kw): raise ValueError('injected '+failure+' gate failure')
    expected_calls=0
    if failure=='preflight':monkeypatch.setattr(runner,'preflight_gate',fail)
    if failure=='lr': monkeypatch.setattr(runner,'lr_gate',fail);expected_calls=1
    if failure=='selection':monkeypatch.setattr(runner,'selection_gate',fail)
    if failure=='prereg':monkeypatch.setattr(runner,'prereg_gate',fail)
    if failure=='warmup':
        logs=tmp_path/'logs';logs.mkdir()
        for seed in (42,137):
            for arm in ('fixed','failure_driven'):(logs/f'DONE_seed_{seed}_{arm}').write_text('done')
        monkeypatch.setattr(runner,'warmup_gate',fail);expected_calls=1
    if failure=='numerical':monkeypatch.setattr(runner.r4,'verify_stage_metrics',fail);expected_calls=1
    if failure=='infra':  # prereg stops.infra: exactly one retry, then stop
        def infra(*a,**kw):stub.append('infra');return 1,.01
        monkeypatch.setattr(runner.r4,'_run_bounded',infra);expected_calls=2
    if failure=='budget':monkeypatch.setattr(runner.r4,'_cost_so_far',lambda *a:(80,15))
    with pytest.raises((ValueError,RuntimeError)):runner.queue(workspace=WS,root=tmp_path)
    assert len(stub)==expected_calls
    assert not (tmp_path/'logs/DONE_seed_2718_warmup' if failure=='warmup' else tmp_path/'logs/DONE_seed_42_fixed').exists()


def test_infra_retry_once_then_next_stage_resumes_from_attempt_2(tmp_path,monkeypatch,stub):
    original=runner.r4._run_bounded
    failed=[]
    def flaky(command,**kw):
        if not failed and 'r4r_train_stage.sh' in command[1] and command[4]=='2':
            failed.append(command);stub.append('infra');return 1,.01
        return original(command,**kw)
    monkeypatch.setattr(runner.r4,'_run_bounded',flaky)
    runner.execute(42,'fixed',workspace=WS,root=tmp_path)
    stage2=tmp_path/'seed_42/fixed/stage_2'
    assert (stage2/'attempt_1/INFRA_FAILED').exists() and (stage2/'attempt_2/checkpoints/global_step_40').is_dir()
    stage3=[c for c in stub if c!='infra' and c[4]=='3']
    assert len(stage3)==1 and stage3[0][7]==str(stage2/'attempt_2/checkpoints/global_step_40')
    assert len(stub)==7


def test_gate_failure_after_successful_process_is_not_retried(tmp_path,monkeypatch,stub):
    def fail(*a,**kw):raise ValueError('injected lr gate failure')
    monkeypatch.setattr(runner,'lr_gate',fail)
    with pytest.raises(ValueError):runner.execute(42,'fixed',workspace=WS,root=tmp_path)
    assert len(stub)==1 and not (tmp_path/'seed_42/fixed/stage_1/attempt_2').exists()


def test_preflight_dry_run_has_both_intervals_and_no_outputs(tmp_path):
    p=subprocess.run([sys.executable,'-m','scripts.self_improve.r4r_preflight','--dry-run'],capture_output=True,text=True,check=True)
    assert '--start 21 --end 22' in p.stdout and '--start 1 --end 2' in p.stdout
    assert 'optimizer and lr_scheduler' in p.stdout
    p=subprocess.run([sys.executable,'-m','scripts.self_improve.r4r_preflight','--dry-run','--scratch',str(tmp_path)],capture_output=True,text=True)
    assert p.returncode!=0
    p=subprocess.run([sys.executable,'-m','scripts.self_improve.r4r_launch','stage','--dry-run','--preflight',
        '137','fixed','1','20','/tmp/train','/tmp/previous','/root/autodl-tmp/r4r_preflight/checks/attempt'],capture_output=True,text=True,check=True)
    assert 'TARGET_STEPS=$((START_UPDATE + 2))' in p.stdout
    assert '+trainer.r4r_scheduler_horizon=100' in p.stdout


def test_prereg_change_between_process_and_next_stage_stops(tmp_path,monkeypatch,stub):
    original=runner.r4._run_bounded
    def change_launch_hash(*args,**kwargs):
        result=original(*args,**kwargs)
        (tmp_path/'prereg_sha256.txt').write_text('changed\n')
        return result
    monkeypatch.setattr(runner.r4,'_run_bounded',change_launch_hash)
    with pytest.raises(ValueError,match='launch hash changed'):
        runner.queue(workspace=WS,root=tmp_path)
    assert len(stub)==1
    assert not list((tmp_path/'logs').glob('DONE_*'))
