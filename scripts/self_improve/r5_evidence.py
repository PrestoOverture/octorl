#!/usr/bin/env python3
"""Read-only compact remote evidence for R5; no model imports."""
import json
import time
from r5_pull import remote, OUT

def main():
    start=time.time()
    value=remote('''import pathlib,json,hashlib,os
root=pathlib.Path('/root/autodl-tmp')
result={'r3c_137_rollout_rows':[], 'resume_views':[], 'checkpoint_state_files':[]}
for p in sorted((root/'octorl_r3c/seed_137/rollouts').glob('*.jsonl'),key=lambda p:int(p.stem)):
    with p.open('rb') as f: digest=hashlib.file_digest(f,'sha256').hexdigest()
    rows=[json.loads(line) for line in p.read_text().splitlines()]
    result['r3c_137_rollout_rows'].append({'path':str(p),'sha256':digest,'step':int(p.stem),'rows':[{'gts':r['gts'],'score':r['score']} for r in rows]})
for p in sorted((root/'octorl_r4/seed_137/fixed').glob('stage_*/attempt_1/resume_view/**/*')):
    if p.is_symlink(): result['resume_views'].append({'path':str(p),'target':os.readlink(p),'target_exists':p.exists()})
for study in ('octorl_r3c','octorl_r4'):
    for p in (root/study).rglob('*.pt'):
        if 'checkpoint' in str(p) and ('optim_' in p.name or 'extra_state' in p.name): result['checkpoint_state_files'].append(str(p))
print(json.dumps(result))''')
    value['remote_query_wall_seconds']=time.time()-start
    (OUT/'raw/offline_remote_evidence.json').write_text(json.dumps(value,indent=2)+'\n')

if __name__=='__main__': main()
