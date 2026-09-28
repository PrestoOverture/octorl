"""R4r serial, fail-stop runner using the unchanged R4 selector and process limits."""
import argparse
import json
from pathlib import Path
import shutil
import sys
try:
    from scripts.self_improve import r4_run_branch as r4
except ModuleNotFoundError:
    import r4_run_branch as r4
try:
    from scripts.self_improve.r4r_gates import preflight_gate, runtime_input, prereg_gate, selection_gate, warmup_gate, RECORDS
except ModuleNotFoundError:
    from r4r_gates import preflight_gate, runtime_input, prereg_gate, selection_gate, warmup_gate, RECORDS
try:
    from scripts.self_improve.r4r_lr_gate import gate as lr_gate
except ModuleNotFoundError:
    from r4r_lr_gate import gate as lr_gate

WORKSPACE = Path(__file__).resolve().parents[2]
ROOT = Path('/root/autodl-tmp/octorl_r4r')


def stage_command(seed, arm, stage, *, workspace=WORKSPACE, root=ROOT, previous_checkpoint=None, attempt=1):
    command = r4.stage_command(seed, arm, stage, workspace=workspace, root=root,
                               previous_checkpoint=previous_checkpoint, attempt=attempt)
    command[1] = str(workspace/'scripts/self_improve/r4r_train_stage.sh')
    return command+["--root",str(root)]


def bounded(command, directory, *, workspace, root, branch, timeout_seconds=r4.STAGE_TIMEOUT_SECONDS,
            allow_failure=False):
    """Run one trainer process; with allow_failure a non-zero exit is returned for the infra-retry rule."""
    prereg_gate(workspace, root)
    global_cost, branch_cost = r4._cost_so_far(root, branch)
    available = min(r4.GLOBAL_LIMIT_CNY-global_cost, r4.BRANCH_LIMIT_CNY-branch_cost)
    if available <= 0:
        raise RuntimeError('budget stop')
    if shutil.disk_usage(root).free < r4.MIN_FREE_BYTES:
        raise RuntimeError('disk stop')
    directory.mkdir(parents=True, exist_ok=True)
    code, elapsed = r4._run_bounded(command, log_path=directory/'training.log',
        max_seconds=min(timeout_seconds, available*3600/r4.RATE_CNY_PER_HOUR))
    (directory/'timing.json').write_text(json.dumps(dict(elapsed_seconds=elapsed,
        cost_cny=elapsed*r4.RATE_CNY_PER_HOUR/3600, exit_code=code, command=command), indent=2)+'\n')
    prereg_gate(workspace, root)
    if elapsed*r4.RATE_CNY_PER_HOUR/3600 > available:
        raise RuntimeError('budget stop after process')
    if code and not allow_failure:
        raise RuntimeError(f'process failed with exit {code}; queue stopped without retry')
    return code


def verify_checkpoint(path):
    for name in ('optim_world_size_1_rank_0.pt','extra_state_world_size_1_rank_0.pt','lora_adapter/adapter_model.safetensors','lora_adapter/adapter_config.json'):
        if not (path/'actor'/name).is_file():
            raise RuntimeError(f'missing checkpoint {path/name}')
    if list(path.rglob('model_world_size_*.pt')):
        raise RuntimeError('unexpected full model shard')


def execute(seed, arm, *, workspace=WORKSPACE, root=ROOT):
    prereg_gate(workspace, root)
    branch = root/f'seed_{seed}'/arm
    warmup = r4.warmup_records_path(seed, workspace=workspace, root=root)
    records = r4._read_jsonl(warmup)
    if len(records) != 320:
        raise ValueError('warm-up must contain 320 rollouts')
    if seed == 2718:
        warmup_gate(warmup, runtime_input(workspace,RECORDS)/'seed_2718/warmup/attributed.jsonl')
    sources, cursor = [(warmup, records)], dict.fromkeys(r4.CELLS, 0)
    previous=r4.warmup_checkpoint(seed,root)
    for stage in range(1,7):
        prereg_gate(workspace, root)
        start = 20+10*(stage-1)
        stage_dir, next_cursor, _ = r4.prepare_stage(seed=seed, arm=arm, stage=stage,
            workspace=workspace, root=root, record_sources=sources, cursor=cursor,
            seen_in_warmup={r['task_id'] for r in records})
        if arm == 'fixed' or stage == 1 and seed in (42,137):
            selection_gate(stage_dir, runtime_input(workspace,RECORDS)/f'seed_{seed}/{arm}/stage_{stage}')
        # Inherited stops.infra: a stage that fails to complete its 10 updates twice stops the branch.
        # Only a failed or checkpoint-less process is retried; every gate below stops without retry.
        for number in (1, 2):
            attempt = stage_dir/f'attempt_{number}'
            code = bounded(stage_command(seed,arm,stage,workspace=workspace,root=root,
                                         previous_checkpoint=previous,attempt=number), attempt,
                           workspace=workspace,root=root,branch=branch,allow_failure=True)
            if code == 0 and (attempt/f'checkpoints/global_step_{start+10}').is_dir():
                break
            (attempt/'INFRA_FAILED').write_text(f'exit_code={code}\n')
            print(f'INFRA_FAILED seed_{seed}_{arm} stage_{stage} attempt_{number} exit={code}', flush=True)
        else:
            raise RuntimeError(f'stage {stage} failed twice; branch stopped at U{start}')
        lr_gate(attempt/f'metrics_target_{start+10}.jsonl', attempt/'training.log',start+1,start+10,
                runtime_input(workspace,'artifacts/self_improve/r5/raw/octorl_r3c/seed_137/metrics_target_100.jsonl'))
        r4.verify_stage_metrics(attempt, start_update=start)
        checkpoint = attempt/f'checkpoints/global_step_{start+10}'
        verify_checkpoint(checkpoint)
        attributed = r4.attribute(attempt,start_step=start+1,end_step=start+10)
        if len(attributed) != 160 or {r.get('stage') for r in attributed} != {stage}:
            raise ValueError('stage attribution is incomplete')
        output = stage_dir/'attributed.jsonl'
        r4.write_records(output,attributed)
        sources.append((output,attributed)); cursor=next_cursor
        # The inherited future-training retention policy applies only within R4r.
        pruned=r4._prune_optimizer_state(previous,root)
        if pruned:
            (stage_dir/'pruned_previous.json').write_text(json.dumps(pruned,indent=2)+'\n')
        previous=checkpoint
        prereg_gate(workspace, root)
        r4.run_dev_health(seed=seed,arm=arm,stage=stage,checkpoint=checkpoint,
                          stage_dir=stage_dir,workspace=workspace,root=root)


def warmup(*,workspace=WORKSPACE,root=ROOT):
    directory = root/'seed_2718/warmup'
    command = ['bash',str(workspace/'scripts/self_improve/r4r_warmup.sh'),'--root',str(root)]
    bounded(command,directory,workspace=workspace,root=root,branch=directory,
            timeout_seconds=r4.BRANCH_LIMIT_CNY*3600/r4.RATE_CNY_PER_HOUR)
    lr_gate(directory/'metrics_target_20.jsonl',directory/'training.log',1,20,
            runtime_input(workspace,'artifacts/self_improve/r5/raw/octorl_r3c/seed_137/metrics_target_100.jsonl'))
    # The same numerical checks apply separately to each ten-update window.
    rows = r4._read_jsonl(directory/'metrics_target_20.jsonl')
    for row in rows:
        import math
        data = row['data']; losses=[v for k,v in data.items() if 'loss' in k.lower()]
        grad=float(data.get('actor/grad_norm',float('nan')))
        if not math.isfinite(grad) or grad > 100 or not losses or not all(math.isfinite(float(v)) for v in losses):
            raise RuntimeError('warm-up numerical stop')
    verify_checkpoint(directory/'checkpoints/global_step_20')
    warmup_gate(directory/'attributed.jsonl',runtime_input(workspace,RECORDS)/'seed_2718/warmup/attributed.jsonl')


def queue(*, workspace=WORKSPACE, root=ROOT):
    prereg_gate(workspace,root)
    preflight_gate(root)
    logs=root/'logs'; logs.mkdir(exist_ok=True)
    for seed,arm in [(42,'fixed'),(42,'failure_driven'),(137,'fixed'),(137,'failure_driven'),(2718,'warmup'),(2718,'fixed'),(2718,'failure_driven')]:
        prereg_gate(workspace,root)
        name=f'seed_{seed}_{arm}'; done=logs/f'DONE_{name}'; directory=root/f'seed_{seed}'/arm
        if done.exists():
            print('SKIP',name,flush=True); continue
        if directory.exists():
            raise RuntimeError(f'REFUSE half-finished {name}')
        print('START',name,flush=True)
        if arm=='warmup': warmup(workspace=workspace,root=root)
        else: execute(seed,arm,workspace=workspace,root=root)
        prereg_gate(workspace,root)
        done.write_text('gates passed\n')
    print('ALL_DONE',flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    mode=p.add_mutually_exclusive_group(required=True)
    mode.add_argument('--execute',action='store_true'); mode.add_argument('--dry-run',action='store_true')
    p.add_argument('--root',type=Path,default=ROOT)
    a=p.parse_args()
    if a.dry_run:
        import shlex
        for seed in (42,137,2718):
            for arm in ('fixed','failure_driven'):
                for stage in range(1,7): print(shlex.join(stage_command(seed,arm,stage,root=a.root)))
        print('warmup: bash scripts/self_improve/r4r_warmup.sh --root '+str(a.root))
    else: queue(root=a.root)


if __name__ == '__main__': main()
