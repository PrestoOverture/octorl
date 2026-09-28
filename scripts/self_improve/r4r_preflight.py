"""GPU checks for later execution. --dry-run only validates and prints the plan."""
import argparse
import json
from pathlib import Path
import re
import shlex
import shutil
import subprocess
try:
    from scripts.self_improve import r4_run_branch as r4
except ModuleNotFoundError:
    import r4_run_branch as r4
try:
    from scripts.self_improve.r4r_gates import EXPECTED, runtime_input, prereg_gate, selection_gate, RECORDS
except ModuleNotFoundError:
    from r4r_gates import EXPECTED, runtime_input, prereg_gate, selection_gate, RECORDS
try:
    from scripts.self_improve.r4r_run_branch import WORKSPACE
except ModuleNotFoundError:
    from r4r_run_branch import WORKSPACE


def plan(scratch):
    stage=scratch/'resume/seed_137/fixed/stage_1'
    attempt=stage/'attempt_1'
    warmup=scratch/'warmup'
    prefix=['python3',str(WORKSPACE/'scripts/self_improve/r4r_launch.py')]
    commands=[prefix+['stage','--preflight','--root',str(scratch),'137','fixed','1','20',str(stage/'train.parquet'),
        '/root/autodl-tmp/octorl_r3c/seed_137/checkpoints/global_step_20',str(attempt)],
        prefix+['warmup','--preflight','--root',str(scratch),'--output',str(warmup)]]
    gates=[['python3','scripts/self_improve/r4r_lr_gate.py','--metrics',str(d/f'metrics_target_{end}.jsonl'),
            '--log',str(d/'training.log'),'--start',str(start),'--end',str(end)]
           for d,start,end in [(attempt,21,22),(warmup,1,2)]]
    return commands,gates,attempt,warmup


def verify_loads(path):
    text=path.read_text()
    for name in ('optimizer','lr_scheduler'):
        if not re.search(r'(?im)^.*(?:load(?:ed|ing)?).*'+name+r'.*$',text):
            raise ValueError(f'missing {name} load evidence')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    mode=p.add_mutually_exclusive_group(required=True)
    mode.add_argument('--dry-run',action='store_true'); mode.add_argument('--execute',action='store_true')
    p.add_argument('--run-root',type=Path,default=Path('/root/autodl-tmp/octorl_r4r'))
    p.add_argument('--scratch',type=Path,default=Path('/root/autodl-tmp/r4r_preflight/checks'))
    a=p.parse_args(); scratch=a.scratch
    if not scratch.is_absolute() or not scratch.is_relative_to('/root/autodl-tmp/r4r_preflight') or scratch==Path('/root/autodl-tmp/r4r_preflight') or '..' in scratch.parts:
        p.error('scratch must be a child of /root/autodl-tmp/r4r_preflight/')
    commands,gates,attempt,warmup=plan(scratch)
    print('Prepare fixed seed-137 stage 1 with the frozen selector and verify selection identity.')
    for command,gate,directory in zip(commands,gates,(attempt,warmup)):
        print(shlex.join(command)+' > '+shlex.quote(str(directory/'training.log'))+' 2>&1',flush=True)
        print(shlex.join(gate),flush=True)
    print('Require optimizer and lr_scheduler load lines in '+str(attempt/'training.log'))
    print('Print results, then delete only scratch '+str(scratch),flush=True)
    if a.dry_run: return
    if scratch.exists(): raise RuntimeError('refusing existing preflight scratch')
    prereg_gate(WORKSPACE,scratch)
    path=r4.warmup_records_path(137,workspace=WORKSPACE,root=scratch)
    records=r4._read_jsonl(path)
    stage,_,_=r4.prepare_stage(seed=137,arm='fixed',stage=1,workspace=WORKSPACE,root=scratch/'resume',
        record_sources=[(path,records)],cursor=dict.fromkeys(r4.CELLS,0),seen_in_warmup={r['task_id'] for r in records})
    selection_gate(stage,runtime_input(WORKSPACE,RECORDS)/'seed_137/fixed/stage_1')
    results=[]
    for command,gate,directory in zip(commands,gates,(attempt,warmup)):
        prereg_gate(WORKSPACE,scratch)
        directory.mkdir(parents=True,exist_ok=True)
        with (directory/'training.log').open('w') as log:
            subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,check=True)
        subprocess.run(gate,check=True)
        if directory==attempt: verify_loads(directory/'training.log')
        results.append({'command':command,'lr_gate':'PASS','loads':'PASS' if directory==attempt else 'fresh base'})
    prereg_gate(WORKSPACE,scratch)
    print(json.dumps(results,indent=2),flush=True)
    prereg_gate(WORKSPACE,a.run_root)
    # Keep the first-hand evidence (metrics + logs) before the scratch checkpoints are deleted.
    evidence=a.run_root/'preflight_evidence'
    for name,directory,end in (('resume',attempt,22),('warmup',warmup,2)):
        (evidence/name).mkdir(parents=True,exist_ok=True)
        for source in (directory/f'metrics_target_{end}.jsonl',directory/'training.log'):
            shutil.copy2(source,evidence/name/source.name)
    (a.run_root/'preflight_pass.json').write_text(json.dumps(dict(prereg_sha256=EXPECTED,
        resume_steps=[21,22],warmup_steps=[1,2],optimizer_load=True,lr_scheduler_load=True,lr_gates=['PASS','PASS'],results=results),indent=2)+'\n')
    shutil.rmtree(scratch)


if __name__=='__main__': main()
