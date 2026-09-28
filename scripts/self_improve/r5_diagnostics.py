#!/usr/bin/env python3
"""Offline log reconstruction and nine-run diagnostics. Never trains/evaluates."""
from __future__ import annotations
import collections
import csv
import json
import math
import re
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'artifacts/self_improve/r5'
RAW=OUT/'raw'
DIAG=OUT/'diagnostics'
ANSI=re.compile(r'\x1b\[[0-?]*[ -/]*[@-~]')

def clean(line):
    return re.sub(r'\(TaskRunner pid=\d+\)\s*','',ANSI.sub('',line))

def read_jsonl(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]

def write_json(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')

def parse_log(path):
    occurrences=collections.defaultdict(list)
    for number,line in enumerate(path.read_text(errors='replace').splitlines(),1):
        line=clean(line)
        match=re.search(r'\bstep:(\d+)\s+-\s+',line)
        if not match: continue
        step=int(match[1]); data={}
        for item in line[match.end():].split(' - '):
            if ':' not in item: raise ValueError(f'malformed metric at {path}:{number}')
            key,value=item.split(':',1)
            try: value=json.loads(value)
            except json.JSONDecodeError: value=value.strip()
            data[key.strip()]=value
        occurrences[step].append({'line':number,'step':step,'data':data})
    records=[{'step':step,'data':rows[-1]['data']} for step,rows in sorted(occurrences.items())]
    duplicates={str(step):rows for step,rows in sorted(occurrences.items()) if len(rows)>1}
    return records,duplicates

def compare_control(log,jsonl):
    parsed,_=parse_log(log); expected=read_jsonl(jsonl)
    a={r['step']:r['data'] for r in parsed}; b={r['step']:r['data'] for r in expected}
    if set(a)!=set(b) or len(b)!=len(expected): raise AssertionError('control step set mismatch')
    missing={}; checked=0
    for step in a:
        missing[str(step)]=sorted(b[step].keys()-a[step].keys())
        for key in a[step].keys() & b[step].keys():
            x,y=a[step][key],b[step][key]
            if isinstance(x,(int,float)) and isinstance(y,(int,float)):
                same=math.isfinite(x) and math.isfinite(y) and abs(x-y)<=1e-6*max(1,abs(y))
            else: same=x==y
            if not same: raise AssertionError(f'control mismatch U{step} {key}: {x} != {y}')
            checked+=1
    return {'passed':True,'steps':sorted(a),'shared_values_checked':checked,'jsonl_keys_absent_from_log':{s:k for s,k in missing.items() if k},'tolerance':'1e-6 * max(1, abs(source))','log':str(log.relative_to(ROOT)),'jsonl':str(jsonl.relative_to(ROOT))}

FIELDS={'reward_mean':'r3b/reward_mean','effective_group_rate':'r3b/mixed_group_fraction','kl':'actor/kl_loss','entropy':'actor/entropy','response_length_mean':'response_length/mean','response_length_trajectory_mean':'response_length/trajectory/mean','grad_norm':'actor/grad_norm','clip_fraction':'actor/pg_clipfrac','lr':'actor/lr','step_seconds':'timing_s/step','pg_loss':'actor/pg_loss','advantage_zero_fraction':'r3b/advantage_zero_frac'}

def main():
    DIAG.mkdir(parents=True,exist_ok=True)
    r3=RAW/'octorl_r3c'
    control=compare_control(r3/'seed_137_train.log',r3/'seed_137/metrics_target_100.jsonl')
    write_json(DIAG/'parser_control.json',control) # MUST precede seed-42 reconstruction
    records,dups=parse_log(r3/'seed_42_train.log')
    assert [r['step'] for r in records]==list(range(1,81))
    write_json(RAW/'r3c_seed42_duplicate_steps.json',dups)
    (DIAG/'r3c_seed42_metrics_reconstructed.jsonl').write_text(''.join(json.dumps(r,allow_nan=False)+'\n' for r in records))
    pruning=read_jsonl(r3/'seed_42/checkpoint_pruning.jsonl')
    latest=int((r3/'seed_42/checkpoints/latest_checkpointed_iteration.txt').read_text())
    assert latest==80 and all(p['step'] in {r['step'] for r in records} for p in pruning)
    # A logged save must have a corresponding completed step. Save timing is positive.
    saves={r['step']:r['data'].get('timing_s/save_checkpoint') for r in records if 'timing_s/save_checkpoint' in r['data']}
    assert all(saves.get(p['step'],0)>0 for p in pruning)
    write_json(DIAG/'reconstruction_check.json',{'steps':80,'log_occurrences':80+sum(len(v)-1 for v in dups.values()),'duplicate_steps':list(map(int,dups)),'resolution':'last occurrence in file order','latest_checkpointed_iteration':latest,'pruning_steps':sorted({p['step'] for p in pruning}),'save_timing_seconds':saves})
    runs={'r3c_42':records,'r3c_137':read_jsonl(r3/'seed_137/metrics_target_100.jsonl')}
    for seed in (42,137,2718):
        for arm in ('fixed','failure_driven'):
            values=[]
            for stage in range(1,7): values+=read_jsonl(RAW/f'octorl_r4/seed_{seed}/{arm}/stage_{stage}/attempt_1/metrics_target_{20+10*stage}.jsonl')
            assert [r['step'] for r in values]==list(range(21,81))
            runs[f'r4_{arm}_{seed}']=values
    runs['r4_warmup_2718']=read_jsonl(RAW/'octorl_r4/seed_2718/warmup/metrics_target_20.jsonl')
    summaries={}
    for name,values in runs.items():
        timing=sorted({k for r in values for k in r['data'] if k.startswith('timing_')})
        columns=['global_step',*FIELDS,*timing]
        with (DIAG/f'{name}.csv').open('w') as f:
            writer=csv.DictWriter(f,fieldnames=columns);writer.writeheader()
            for row in values: writer.writerow({'global_step':row['step'],**{k:row['data'].get(v,'') for k,v in FIELDS.items()},**{k:row['data'].get(k,'') for k in timing}})
        summaries[name]={'steps':len(values),'first_step':values[0]['step'],'last_step':values[-1]['step'],'last_pg_loss':values[-1]['data'].get('actor/pg_loss'),'logged_step_seconds':sum(r['data']['timing_s/step'] for r in values),'timing_fields':timing,'missing_fields':{k:[r['step'] for r in values if v not in r['data']] for k,v in FIELDS.items() if any(v not in r['data'] for r in values)}}
    write_json(DIAG/'run_summary.json',summaries)
    write_json(DIAG/'metric_definitions.json',{'columns':FIELDS,'effective_group_rate':'Fraction of original reward groups with non-identical rewards. This is NOT the fraction of non-zero Sign advantages: A=2r-1 remains non-zero in uniform groups.','reward_mean':'Trajectory-level binary reward, not token/turn-averaged critic/rewards/mean.','kl':'actor/kl_loss (low_var_kl), not rollout importance-sampling KL.','timing':'Only logged timing fields. No isolated offload/sync time is inferred from aggregate generation or actor time.'})
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    for field in ('reward_mean','effective_group_rate','kl','entropy','lr'):
        fig,axes=plt.subplots(3,1,figsize=(11,10),sharex=True)
        for ax,seed in zip(axes,(42,137,2718)):
            parent=runs[f'r3c_{seed}'] if seed!=2718 else runs['r4_warmup_2718']
            ax.plot([r['step'] for r in parent],[r['data'][FIELDS[field]] for r in parent],label=f'R3c {seed}' if seed!=2718 else 'warm-up 2718',color='0.45',alpha=.7)
            for arm in ('fixed','failure_driven'):
                values=[next(r for r in parent if r['step']==20)]+runs[f'r4_{arm}_{seed}']
                ax.plot([r['step'] for r in values],[r['data'][FIELDS[field]] for r in values],label=f'R4 {arm}',linewidth=1,alpha=.85)
            ax.axvline(20,color='black',linestyle=':',linewidth=.8);ax.set_ylabel(field);ax.set_title(f'Training seed {seed}');ax.legend(fontsize=8);ax.grid(alpha=.2)
        axes[-1].set_xlabel('Global update (raw, unsmoothed)');fig.tight_layout();fig.savefig(DIAG/f'{field}.png',dpi=150);plt.close(fig)

if __name__=='__main__': main()
