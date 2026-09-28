"""Reuse the frozen R4 shell paths with checked, minimal R4r substitutions."""
import argparse
from pathlib import Path
import subprocess
import sys
try:
    from scripts.self_improve.r4r_gates import runtime_input, prereg_gate, selection_gate, RECORDS
except ModuleNotFoundError:
    from r4r_gates import runtime_input, prereg_gate, selection_gate, RECORDS

WORKSPACE = Path(__file__).resolve().parents[2]
ROOT = Path('/root/autodl-tmp/octorl_r4r')


def replace_once(text, old, new):
    if text.count(old) != 1:
        raise ValueError(f'R4 source changed: expected one occurrence of {old!r}')
    return text.replace(old, new)


def render(kind, *, output=None, preflight=False):
    source = WORKSPACE/f'scripts/self_improve/r4_{"train_stage" if kind == "stage" else "warmup"}.sh'
    text = source.read_text()
    text = replace_once(text, '    actor_rollout_ref.actor.optim.total_training_steps=100 \\\n',
        '    actor_rollout_ref.actor.optim.total_training_steps=100 \\\n    +trainer.r4r_scheduler_horizon=100 \\\n    +actor_rollout_ref.r4r_scheduler_log=true \\\n')
    text = text.replace('trainer.project_name=octorl_r4', 'trainer.project_name=octorl_r4r')
    text = text.replace('r4_seed_', 'r4r_seed_')
    if kind == 'warmup':
        import shlex
        text = replace_once(text, 'RUN_ROOT=/root/autodl-tmp/octorl_r4/seed_${SEED}/warmup',
                            'RUN_ROOT='+shlex.quote(str(output)))
        if preflight:
            text = replace_once(text, 'TARGET_STEPS=20', 'TARGET_STEPS=2')
            text = replace_once(text, 'SAVE_FREQ=20', 'SAVE_FREQ=2')
            text = text.replace('global_step_20', 'global_step_2').replace('--end-step 20', '--end-step 2')
    elif preflight:
        text = replace_once(text, 'TARGET_STEPS=$((START_UPDATE + 10))', 'TARGET_STEPS=$((START_UPDATE + 2))')
        text = replace_once(text, 'SAVE_FREQ=10', 'SAVE_FREQ=2')
    return text


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('kind', choices=('stage','warmup'))
    p.add_argument('--dry-run', action='store_true')
    p.add_argument('--preflight', action='store_true')
    p.add_argument('--root', type=Path, default=ROOT)
    p.add_argument('--output', type=Path)
    p.add_argument('stage_args', nargs='*')
    a = p.parse_intermixed_args()
    if a.kind == 'stage':
        if len(a.stage_args) != 7:
            p.error('stage requires SEED ARM STAGE START TRAIN_PARQUET PREVIOUS_CKPT RUN_ROOT')
        seed, arm, stage, start, parquet, previous, output = a.stage_args
        if int(seed) not in (42,137,2718) or arm not in ('fixed','failure_driven') or not 1 <= int(stage) <= 6 or int(start) != 20+10*(int(stage)-1):
            p.error('invalid stage arguments')
    else:
        if a.stage_args:
            p.error('warmup does not accept stage arguments')
        output = str(a.output or a.root/'seed_2718/warmup')
    if a.preflight and not Path(output).is_relative_to('/root/autodl-tmp/r4r_preflight'):
        p.error('preflight outputs must be under /root/autodl-tmp/r4r_preflight/')
    script = render(a.kind, output=output, preflight=a.preflight)
    if a.dry_run:
        print(script)
        return
    prereg_gate(WORKSPACE, a.root)
    if a.kind == 'stage' and (arm == 'fixed' or int(stage) == 1 and int(seed) in (42,137)):
        selection_gate(Path(parquet).parent, runtime_input(WORKSPACE,RECORDS)/f'seed_{seed}/{arm}/stage_{stage}')
    sys.exit(subprocess.run(['bash','-s','--',*a.stage_args], input=script, text=True).returncode)


if __name__ == '__main__':
    main()
