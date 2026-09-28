#!/usr/bin/env python3
"""Read-only R5 acquisition, with retries, remote hashes and resumable transfers."""
import hashlib
import json
import shlex
import subprocess
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'artifacts/self_improve/r5'
SSH = ['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=20', '-i', str(Path.home()/'.ssh/autodl_octorl'), 'autodl-r4']

def remote(code):
    for attempt in range(5):
        result = subprocess.run(SSH + ['export PATH=/root/miniconda3/bin:$PATH; python3 -c ' + shlex.quote(code)], capture_output=True, text=True)
        if result.returncode == 0:
            return json.loads(result.stdout)
        if attempt == 4:
            raise RuntimeError(result.stderr)
        time.sleep(2)

def sha(path):
    with path.open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()

def adapters():
    entries = []
    for seed in (42, 137, 2718):
        parent = f'r3c_{seed}_u20' if seed != 2718 else 'r4_warmup_2718_u20'
        for branch in ('fixed', 'failure_driven'):
            entries.append((f'r4_{branch}_{seed}_u80', f'/root/autodl-tmp/octorl_r4/seed_{seed}/{branch}/stage_6/attempt_1/checkpoints/global_step_80/actor/lora_adapter', 80, parent))
    for seed in (42, 137):
        for step in (20, 50, 80):
            entries.append((f'r3c_{seed}_u{step}', f'/root/autodl-tmp/octorl_r3c/seed_{seed}/checkpoints/global_step_{step}/actor/lora_adapter', step, f'r3c_{seed}_u20' if step > 20 else 'Qwen3-4B@1cfa9a7'))
    entries.append(('r4_warmup_2718_u20', '/root/autodl-tmp/octorl_r4/seed_2718/warmup/checkpoints/global_step_20/actor/lora_adapter', 20, 'Qwen3-4B@1cfa9a7'))
    return entries

def main():
    OUT.mkdir(parents=True, exist_ok=True)
    started = time.time()
    patterns = [f'/root/autodl-tmp/octorl_r4/seed_*/{arm}/stage_*/attempt_1/{file}' for arm in ('fixed','failure_driven') for file in ('metrics_target_*.jsonl','training.log')]
    patterns += ['/root/autodl-tmp/octorl_r4/seed_2718/warmup/metrics_target_20.jsonl']
    patterns += ['/root/autodl-tmp/octorl_r3c/'+f for f in ['seed_42_train.log',*[f'seed_42_train_run{x}.log' for x in ('4','5','5b','6','7','8')],'seed_137_train.log','seed_137/metrics_target_100.jsonl','seed_42/exit.txt','seed_42/checkpoint_pruning.jsonl','seed_137/checkpoint_pruning.jsonl']]
    patterns += ['/root/autodl-tmp/octorl_r3c/seed_*/checkpoints/latest_checkpointed_iteration.txt', '/root/autodl-tmp/octorl_r3c/seed_*/trajectories.jsonl', '/root/autodl-tmp/octorl_r4/seed_*/fixed/stage_*/attempt_1/resume_view/lora_adapter/adapter_config.json']
    spec = adapters()
    code = 'import glob,hashlib,json,pathlib\npatterns='+repr(patterns)+'\nadapters='+repr(spec)+'''\nfiles = {}; missing = []
for pattern in patterns:
    matches = [p for p in glob.glob(pattern) if pathlib.Path(p).is_file()]
    if not matches: missing.append(pattern)
    for p in matches: files[p] = 'raw/' + p.removeprefix('/root/autodl-tmp/')
for name, directory, step, parent in adapters:
    matches = [p for p in pathlib.Path(directory).rglob('*') if p.is_file()]
    if not matches: missing.append(directory)
    for p in matches: files[str(p)] = 'checkpoints/'+name+'/'+str(p.relative_to(directory))
rows=[]
for src, dst in files.items():
    with open(src,'rb') as f: digest=hashlib.file_digest(f,'sha256').hexdigest()
    rows.append(dict(remote_path=src,local_path=dst,sha256=digest,bytes=pathlib.Path(src).stat().st_size))
print(json.dumps(dict(files=rows,missing=missing)))
'''
    inventory = remote(code)
    (OUT/'pull_inventory.json').write_text(json.dumps(inventory, indent=2)+'\n')
    for i, row in enumerate(inventory['files']):
        dest = OUT / row['local_path']; dest.parent.mkdir(parents=True, exist_ok=True)
        if not dest.exists() or sha(dest) != row['sha256']:
            for attempt in range(5):
                result = subprocess.run(['rsync','--partial','-a','-e',shlex.join(SSH[:-1]),'autodl-r4:'+row['remote_path'], str(dest)])
                if result.returncode == 0 and sha(dest) == row['sha256']: break
                if attempt == 4: raise RuntimeError('Transfer failed: '+row['remote_path'])
                time.sleep(2)
        print(f'{i+1}/{len(inventory["files"])} {row["local_path"]}', flush=True)
    manifest = []
    evals = [(p, json.loads(p.read_text())) for p in (ROOT/'artifacts/self_improve/r4/test_eval').rglob('*.json')]
    for name, remote_path, step, parent in spec:
        linked = [str(p.relative_to(ROOT)) for p,d in evals if d.get('lora_path') == remote_path]
        if step != 20 and len(linked) != 1: raise ValueError((name, linked))
        files = {r['local_path'].split(name+'/',1)[1]: r['sha256'] for r in inventory['files'] if r['local_path'].startswith('checkpoints/'+name+'/')}
        manifest.append(dict(name=name,local_path='artifacts/self_improve/r5/checkpoints/'+name,remote_path=remote_path,global_step=step,lineage_parent=parent,files=files,test_eval_json=linked[0] if linked else None))
    (OUT/'checkpoint_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    inventory.update(started_unix=started, finished_unix=time.time(), remote_session_wall_seconds=time.time()-started)
    (OUT/'pull_inventory.json').write_text(json.dumps(inventory,indent=2)+'\n')

if __name__ == '__main__': main()
