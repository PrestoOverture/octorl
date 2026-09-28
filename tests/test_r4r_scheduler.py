"""Model veRL's actual LambdaLR restore ordering, including the broken horizon."""
import copy
import csv
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
import libcst as cst
import pytest
import torch
from omegaconf import OmegaConf
from scripts.self_improve.r4r_lr_gate import read_lrs, REFERENCE
from scripts.self_improve.r4r_launch import render
from scripts.self_improve.r4r_gates import sha256

ART = Path('artifacts/self_improve/r4r')
spec=importlib.util.spec_from_file_location('verl_cosine',ART/'verl_cosine_schedule.py')
verl=importlib.util.module_from_spec(spec); spec.loader.exec_module(verl)


def simulation(fixed=True, warmup_horizon=100):
    result={}; saved=None
    for start,target in [(0,20)]+[(s,s+10) for s in range(20,80,10)]:
        optimizer=torch.optim.AdamW([torch.nn.Parameter(torch.zeros(1))],lr=2e-5)
        scheduler=verl.get_cosine_schedule_with_warmup(optimizer,5,100 if fixed else (warmup_horizon if start==0 else target))
        if saved:
            optimizer.load_state_dict(saved[0]); scheduler.load_state_dict(saved[1])
        for step in range(start+1,target+1):
            result[step]=scheduler.get_last_lr()[0]
            optimizer.step(); scheduler.step()
        saved=copy.deepcopy((optimizer.state_dict(),scheduler.state_dict()))
    return result


def test_fixed_curve_and_negative_controls():
    assert max(abs(simulation()[s]-v) for s,v in read_lrs(REFERENCE).items()) <= 1e-12
    broken=simulation(False)
    rows=list(csv.DictReader(open('artifacts/self_improve/r5/diagnostics/r4_fixed_137.csv')))
    assert len(rows)==60
    assert max(abs(broken[int(r['global_step'])]-float(r['lr'])) for r in rows) <= 1e-15
    warm=read_lrs('artifacts/self_improve/r4/remote_records/seed_2718/warmup/metrics_target_20.jsonl')
    assert max(abs(simulation(False,20)[s]-v) for s,v in warm.items()) <= 1e-15


def method(file, cls, method):
    tree=cst.parse_module((ART/'source_after'/file).read_text())
    node=next(n for n in tree.body if isinstance(n,cst.ClassDef) and n.name.value==cls)
    func=next(n for n in node.body.body if isinstance(n,cst.FunctionDef) and n.name.value==method)
    return tree.code_for_node(func)


@pytest.mark.parametrize('target',[20,30,40,50,60,70,80])
def test_actual_patched_init_to_worker_scheduler(target,capsys):
    class Base:
        def __init__(self, config):
            self.config=config
            config.actor_rollout_ref.actor.optim.total_training_steps=config.trainer.total_training_steps
    class WorkerBase:
        def _build_model_optimizer(self, *, optim_config):
            optimizer=torch.optim.AdamW([torch.nn.Parameter(torch.zeros(1))],lr=optim_config.lr)
            scheduler=verl.get_cosine_schedule_with_warmup(optimizer,optim_config.lr_warmup_steps,optim_config.total_training_steps)
            return None,optimizer,scheduler,None
    env={'Base':Base,'WorkerBase':WorkerBase,'AgentAdvantageEstimator':SimpleNamespace(GAE='gae',TOKEN_GAE='token_gae')}
    class Estimator:
        GAE='gae'; TOKEN_GAE='token_gae'
        def __new__(cls,x): return x
    env['AgentAdvantageEstimator']=Estimator
    for name,base,file,cls,meth in [('Trainer','Base','ray_trainer.py','RayAgentTrainer','__init__'),('Worker','WorkerBase','fsdp_workers.py','AsyncActorRolloutRefWorker','_build_model_optimizer')]:
        import textwrap
        exec(f'class {name}({base}):\n'+textwrap.indent(method(file,cls,meth),'    '),env)
    config=OmegaConf.create({'trainer':{'total_training_steps':target,'r4r_scheduler_horizon':100},'algorithm':{'adv_estimator':'sign'},
        'actor_rollout_ref':{'r4r_scheduler_log':True,'actor':{'optim':{'lr':2e-5,'lr_warmup_steps':5,'lr_scheduler_type':'cosine','total_training_steps':100}}}})
    before=OmegaConf.to_container(config)
    trainer=env['Trainer'](config); worker=env['Worker'](); worker.config=config.actor_rollout_ref
    scheduler=worker._build_model_optimizer(optim_config=config.actor_rollout_ref.actor.optim)[2]
    assert OmegaConf.to_container(config)==before
    assert config.trainer.total_training_steps==target
    assert scheduler.lr_lambdas[0](79)*2e-5==pytest.approx(read_lrs(REFERENCE)[80],abs=1e-12)
    assert 'R4R_LR_SCHEDULER total_steps=100 warmup=5 type=cosine' in capsys.readouterr().out
    script=render('warmup' if target==20 else 'stage',output='/tmp/warmup')
    assert '+trainer.r4r_scheduler_horizon=100' in script
    assert 'trainer.total_training_steps="$TARGET_STEPS"' in script


def test_verbatim_scheduler_source_hash():
    evidence=json.loads((ART/'patch_hashes.json').read_text())
    assert sha256(ART/'source_before/torch_functional.py')==evidence['scheduler_source_sha256']
    source=(ART/'verl_cosine_schedule.py').read_text().split('def get_cosine_schedule_with_warmup',1)[1]
    assert ('def get_cosine_schedule_with_warmup'+source) in (ART/'source_before/torch_functional.py').read_text()
