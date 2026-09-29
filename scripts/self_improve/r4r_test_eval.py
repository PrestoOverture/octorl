"""R4r sealed-test evaluation queue (prereg r4r-lr-fixed-rerun-v1, `evaluation` block).

Refuses to run until the training queue has finished (unseal condition). Serial, skip-done, and it stops at the
first failed evaluation. Order: primary test2 (7 models, eval seed 510000) → secondary R3 test (7 models,
410000) → exploratory R3c U80 on test2.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
try:
    from scripts.self_improve.r4r_gates import runtime_input, PREREG
except ModuleNotFoundError:
    from r4r_gates import runtime_input, PREREG

WORKSPACE = Path(__file__).resolve().parents[2]
ROOT = Path('/root/autodl-tmp/octorl_r4r')
PREREG_SHA = '46d844d862eb55048c3be346ecf4ebc56f8f1133596e54bd4221891a256f78a4'
TEST2_SHA = '1b729b717c3dfffde8579cf19bb2eb1fae9b446aba6890d3fd75e96f47f38bb3'
BRANCHES = [f'{arm}_{seed}' for arm in ('fixed', 'failure_driven') for seed in (42, 137, 2718)]
DONE = ['seed_42_fixed', 'seed_42_failure_driven', 'seed_137_fixed', 'seed_137_failure_driven',
        'seed_2718_warmup', 'seed_2718_fixed', 'seed_2718_failure_driven']


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def adapter(root, name):
    arm, seed = name.rsplit('_', 1)
    found = sorted((root / f'seed_{seed}' / arm / 'stage_6').glob('attempt_*/checkpoints/global_step_80/actor/lora_adapter'))
    if len(found) != 1 or not (found[0] / 'adapter_model.safetensors').is_file():
        raise RuntimeError(f'{name}: expected exactly one U80 adapter, found {found}')
    return found[0]


def plan(root, workspace):
    test2 = workspace / 'artifacts/self_improve/r4r/data/test2_manifest.json'
    old = workspace / 'artifacts/self_improve/r3/data/test_manifest.json'
    jobs = []
    for split, manifest, seed in (('test2', test2, 510000), ('test_r3', old, 410000)):
        jobs.append((split, 'base', manifest, seed, None))
        jobs += [(split, name, manifest, seed, name) for name in BRANCHES]
    for s in (42, 137):
        jobs.append(('exploratory_test2', f'r3c_{s}_u80', test2, 510000,
                     Path(f'/root/autodl-tmp/octorl_r3c/seed_{s}/checkpoints/global_step_80/actor/lora_adapter')))
    return jobs


def unsealed(root):
    log = (root / 'logs/queue.log').read_text()
    missing = [d for d in DONE if not (root / 'logs' / f'DONE_{d}').is_file()]
    if 'ALL_DONE' not in log or missing:
        raise RuntimeError(f'test stays sealed: ALL_DONE={"ALL_DONE" in log}, missing DONE markers {missing}')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    mode = p.add_mutually_exclusive_group(required=True)
    mode.add_argument('--dry-run', action='store_true'); mode.add_argument('--execute', action='store_true')
    p.add_argument('--root', type=Path, default=ROOT)
    a = p.parse_args()
    jobs = plan(a.root, WORKSPACE)
    if sha256(runtime_input(WORKSPACE, PREREG)) != PREREG_SHA:
        raise RuntimeError('prereg hash differs')
    if sha256(jobs[0][2]) != TEST2_SHA:
        raise RuntimeError('test2 manifest hash differs from the seal')
    out = a.root / 'test_eval'
    for split, name, manifest, seed, lora in jobs:
        lora_path = adapter(a.root, lora) if isinstance(lora, str) and not a.dry_run else lora
        cmd = [sys.executable, str(WORKSPACE / 'scripts/self_improve/r3b_eval.py'), '--manifest', str(manifest),
               '--output', str(out / split / f'{name}.json'), '--trajectories', str(out / split / f'{name}_trajectories.jsonl'),
               '--eval-seed', str(seed)] + (['--lora-path', str(lora_path)] if lora_path else [])
        print(split, name, ' '.join(cmd), flush=True)
    if a.dry_run:
        return
    unsealed(a.root)
    progress = out / 'progress.log'; out.mkdir(parents=True, exist_ok=True)
    env = {k: v for k, v in os.environ.items() if k.lower() not in ('http_proxy', 'https_proxy')}
    env.update(PYTHONPATH=f'{WORKSPACE}:/root/Agent-R1', HF_ENDPOINT='https://hf-mirror.com', CUDA_VISIBLE_DEVICES='0')
    for split, name, manifest, seed, lora in jobs:
        result = out / split / f'{name}.json'
        if result.is_file():
            print('SKIP', split, name, flush=True); continue
        lora_path = adapter(a.root, lora) if isinstance(lora, str) else lora
        result.parent.mkdir(parents=True, exist_ok=True)
        cmd = [sys.executable, str(WORKSPACE / 'scripts/self_improve/r3b_eval.py'), '--manifest', str(manifest),
               '--output', str(result), '--trajectories', str(out / split / f'{name}_trajectories.jsonl'),
               '--eval-seed', str(seed)] + (['--lora-path', str(lora_path)] if lora_path else [])
        with progress.open('a') as log:
            log.write(f'{time.strftime("%Y-%m-%dT%H:%M:%S")} START {split} {name}\n')
        with (out / split / f'{name}.log').open('w') as log:
            code = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, env=env).returncode
        with progress.open('a') as log:
            log.write(f'{time.strftime("%Y-%m-%dT%H:%M:%S")} EXIT={code} {split} {name}\n')
        if code or not result.is_file():
            raise RuntimeError(f'{split} {name} failed with exit {code}; evaluation queue stopped')
        record = json.loads(result.read_text())
        if Path(record['manifest_path']).resolve() != Path(manifest).resolve() or (lora_path and record['lora_path'] != str(lora_path)):
            raise RuntimeError(f'{split} {name}: result provenance mismatch')
    print('EVAL_ALL_DONE', flush=True)


if __name__ == '__main__':
    main()
