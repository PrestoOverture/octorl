#!/usr/bin/env python3
"""Descriptive offline evidence, config differences and row-distribution audit."""
import ast  # literal_eval only: reads config dumps; never rewrites source.
import collections
import json
import math
from pathlib import Path
from r5_diagnostics import ROOT,OUT,RAW,DIAG,clean,read_jsonl,write_json

R4=ROOT/'artifacts/self_improve/r4/remote_records'

def config_dump(path):
    lines=path.read_text().splitlines(); start=None; parts=[]
    for i,line in enumerate(lines,1):
        line=clean(line)
        if start is None:
            if not line.startswith("{'actor_rollout_ref':"): continue
            start=i
        if not line or line[0] not in "'\"{}[]0123456789-": continue
        parts.append(line)
        try: value=ast.literal_eval('\n'.join(parts))
        except (SyntaxError,ValueError): continue
        if isinstance(value,dict): return value,start,i
    raise ValueError('no resolved config in '+str(path))

def flatten(value,prefix=''):
    result={}
    for k,v in value.items():
        key=prefix+'.'+k if prefix else k
        if isinstance(v,dict):result.update(flatten(v,key))
        else: result[key]=v
    return result

def task_parts(task):
    for fault in ('constraint_violation','missing_dependency','stale_version'):
        if task.endswith('_'+fault):
            family=task.removesuffix('_'+fault).rsplit('_',1)[0]
            return family,fault
    raise ValueError(task)

def distribution(tasks):
    count=len(tasks)
    return {'rows':count,'unique_instances':len(set(tasks)),'fault_share':{k:v/count for k,v in sorted(collections.Counter(task_parts(t)[1] for t in tasks).items())},'family_share':{k:v/count for k,v in sorted(collections.Counter(task_parts(t)[0] for t in tasks).items())}}

def main():
    base_path=RAW/'octorl_r3c/seed_137_train.log'; base,start,end=config_dump(base_path);base=flatten(base)
    diffs={}
    for stage in range(1,7):
        p=RAW/f'octorl_r4/seed_137/fixed/stage_{stage}/attempt_1/training.log'
        config,a,b=config_dump(p); config=flatten(config)
        diffs[str(stage)]={'r3c_source':f'{base_path.relative_to(ROOT)}:{start}-{end}','r4_source':f'{p.relative_to(ROOT)}:{a}-{b}','differences':{k:{'r3c':base.get(k),'r4':config.get(k)} for k in sorted(base.keys()|config.keys()) if base.get(k)!=config.get(k)}}
    write_json(DIAG/'resolved_config_diff.json',diffs)
    r3={r['step']:r['data'] for r in read_jsonl(RAW/'octorl_r3c/seed_137/metrics_target_100.jsonl')}
    r4={r['step']:r['data'] for stage in range(1,7) for r in read_jsonl(RAW/f'octorl_r4/seed_137/fixed/stage_{stage}/attempt_1/metrics_target_{20+10*stage}.jsonl')}
    lr=[{'step':s,'r3c':r3[s]['actor/lr'],'r4':r4[s]['actor/lr'],'difference':r4[s]['actor/lr']-r3[s]['actor/lr']} for s in range(21,81)]
    write_json(DIAG/'lr_comparison.json',lr)
    def cosine(step, horizon):
        return 2e-5 * 0.5 * (1 + math.cos(math.pi * (step-5)/(horizon-5)))
    predictions=[]
    for row in lr:
        step=row['step'];stage=(step-21)//10+1;start=20+10*(stage-1)
        horizon=(100 if stage==1 else start) if step==start+1 else start+10
        predicted=cosine(step-1,horizon)
        assert abs(predicted-row['r4'])<1e-15
        assert abs(cosine(step-1,100)-row['r3c'])<1e-15
        predictions.append({'step':step,'effective_horizon':horizon,'predicted_r4_lr':predicted,'observed_r4_lr':row['r4']})
    write_json(DIAG/'scheduler_formula_check.json',{'source':'raw/scheduler_source_evidence.json','tolerance':1e-15,'all_60_steps_match':True,'predictions':predictions})
    write_json(DIAG/'u21_comparison.json',{'r3c':r3[21],'r4_fixed':r4[21]})
    evidence=json.loads((RAW/'offline_remote_evidence.json').read_text())
    dumped={r['step']:r for r in evidence['r3c_137_rollout_rows']}
    assert set(range(1,81))<=dumped.keys()
    trajectories=read_jsonl(RAW/'octorl_r3c/seed_137/trajectories.jsonl')
    assert len(trajectories)==1280
    original=[];warm=[]
    for step in range(1,81):
        rows=dumped[step]['rows'];assert len(rows)==16
        actual=collections.Counter((r['task_id'],r['binary_reward']) for r in trajectories[(step-1)*16:step*16])
        assert actual==collections.Counter((r['gts'],r['score']) for r in rows)
        counts=collections.Counter(r['gts'] for r in rows)
        assert len(counts)==4 and set(counts.values())=={4}
        # Preserve first-seen group order in the numbered dump. Worker balancing
        # may reorder within a batch; this is not a claim about original dataloader order.
        tasks=list(dict.fromkeys(r['gts'] for r in rows))
        (warm if step<=20 else original).extend(tasks)
    fixed=[];stage_tasks={}
    for stage in range(1,7):
        p=R4/f'seed_137/fixed/stage_{stage}/selection.json'
        tasks=json.loads(p.read_text())['tasks'];assert len(tasks)==40
        fixed.extend(tasks);stage_tasks[str(stage)]=tasks
    dist={'r3c':distribution(original),'r4_fixed':distribution(fixed),'instance_overlap':len(set(original)&set(fixed)),'same_order':original==fixed,'warmup_unique':len(set(warm)),'r3c_warmup_reused_rows':sum(t in set(warm) for t in original),'r4_warmup_reused_rows':sum(t in set(warm) for t in fixed),'r3c_order_basis':'first-seen groups in numbered dumps; within-batch dataloader order is not independently established','r3c_ordered_tasks':original,'r4_stage_ordered_tasks':stage_tasks,'r3c_dump_trajectory_multiset_check':'all 80 batches match; 16 trajectories, four unique tasks with four rollouts each'}
    write_json(DIAG/'row_distribution.json',dist)
    table=['# U21–U80 training-row distribution: seed 137', '',
           'Descriptive only; row shares do not identify a cause of the test gap. R3c test evidence is exploratory, not pre-registered.', '',
           '| Dimension | Category | R3c share | R4 fixed share |', '|---|---|---:|---:|']
    for dimension in ('fault_share','family_share'):
        for key in dist['r3c'][dimension]:
            table.append(f"| {dimension} | {key} | {dist['r3c'][dimension][key]} | {dist['r4_fixed'][dimension][key]} |")
    table += ['', 'Source: `row_distribution.json`, including R3c dump-observed group order and R4 selection order; original R3c within-batch dataloader order is not independently established. Each run has 240 unique instances; overlap is 233. R4 reuses 7 warm-up instances and R3c reuses none. R4 stratifies each stage by fault type, orders unseen instances before seen instances inside each cell, then permutes the stage rows. Seed noise cannot be ruled out offline.']
    (DIAG/'row_distribution.md').write_text('\n'.join(table)+'\n')
    attribution={}
    for seed in (42,137,2718):
        for arm in ('fixed','failure_driven'):
            rows=[];sources=[]
            for stage in range(1,7):
                p=R4/f'seed_{seed}/{arm}/stage_{stage}/attributed.jsonl';rows+=read_jsonl(p);sources.append(str(p.relative_to(ROOT)))
            groups=collections.defaultdict(list)
            for r in rows:groups[(r['global_step'],r['task_id'])].append(r['binary_reward'])
            assert len(rows)==960 and len(groups)==240 and all(len(v)==4 for v in groups.values())
            metrics={r['step']:r['data'] for stage in range(1,7) for r in read_jsonl(RAW/f'octorl_r4/seed_{seed}/{arm}/stage_{stage}/attempt_1/metrics_target_{20+10*stage}.jsonl')}
            for step in range(21,81):
                batch=[r['binary_reward'] for r in rows if r['global_step']==step]
                mixed=sum(len(set(v))>1 for (s,t),v in groups.items() if s==step)/4
                assert abs(sum(batch)/16-metrics[step]['r3b/reward_mean'])<1e-6
                assert abs(mixed-metrics[step]['r3b/mixed_group_fraction'])<1e-6
            attribution[f'{arm}_{seed}']={'sources':sources,'rollouts':len(rows),'groups':len(groups),'binary_reward_mean':sum(r['binary_reward'] for r in rows)/len(rows),'effective_group_rate':sum(len(set(v))>1 for v in groups.values())/len(groups),'diagnostic_flags':dict(collections.Counter(flag for r in rows for flag in r['diagnostic_flags'])),'mask_evidence':'attributed.jsonl contains no token mask arrays; diagnostic_flags and binary rewards only. Token-level mask correctness is NOT DETERMINABLE OFFLINE from these records.','reward_metric_agreement':'all 60 steps match r3b/reward_mean and r3b/mixed_group_fraction','fault_rollout_counts':dict(collections.Counter(r['fault_type'] for r in rows))}
    write_json(DIAG/'attribution_summary.json',attribution)
    print(json.dumps({'config_diff_count':{k:len(v['differences']) for k,v in diffs.items()},'lr_max_difference':max(abs(r['difference']) for r in lr),'distribution':{k:v for k,v in dist.items() if 'tasks' not in k}},indent=2))

if __name__=='__main__':main()
