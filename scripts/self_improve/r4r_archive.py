import subprocess,json,hashlib,time
from pathlib import Path
root=Path('artifacts/self_improve'); out=root/'r4/checkpoint_archive'; out.mkdir(parents=True,exist_ok=True)
scopes=['seed_42','seed_137','seed_2718','validation']
script='''import pathlib,hashlib,json,subprocess
root=pathlib.Path('/root/autodl-tmp/octorl_r4'); scopes=['seed_42','seed_137','seed_2718','validation']; rows=[]; links=[]
for scope in scopes:
 for p in sorted((root/scope).rglob('*')):
  if p.is_symlink():
   links.append(dict(path=str(p.relative_to(root)),target=str(p.readlink())))
  elif p.is_file():
   h=hashlib.sha256()
   with p.open('rb') as f:
    for c in iter(lambda:f.read(8*1024*1024),b''): h.update(c)
   rows.append(dict(path=str(p.relative_to(root)),size=p.stat().st_size,remote_sha256=h.hexdigest(),symlink=p.is_symlink()))
counts={s:int(subprocess.check_output('find '+str(root/s)+' -type f | wc -l',shell=True)) for s in scopes}
print(json.dumps(dict(files=rows,symlinks=links,remote_find_counts=counts)))'''
p=subprocess.run(['ssh','-o','BatchMode=yes','autodl-r4','export PATH=/root/miniconda3/bin:$PATH; python3 -'],input=script,text=True,capture_output=True,check=True)
manifest=json.loads(p.stdout); (root/'r4r/archive_remote_inventory.json').write_text(json.dumps(manifest,indent=2)+'\n')
# Reuse existing local bytes only after matching a remote SHA-256. This reduces
# transfer of the already archived R5 U80 adapters and identical validation files.
import shutil
index={}
for base in (root/'r5/checkpoints',out):
 for candidate in base.rglob('*'):
  if candidate.is_file() and not candidate.is_symlink():
   h=hashlib.sha256()
   with candidate.open('rb') as f:
    for c in iter(lambda:f.read(8*1024*1024),b''):h.update(c)
   index[h.hexdigest()]=candidate
reused=0
for row in manifest['files']:
 source=index.get(row['remote_sha256']); dest=out/row['path']
 if source is not None and source!=dest:
  dest.parent.mkdir(parents=True,exist_ok=True)
  shutil.copy2(source,dest);reused+=row['size']
print(f'Locally reused {reused} verified bytes',flush=True)
def transfer(scope):
 for attempt in range(1,11):
  print(f'rsync {scope} attempt {attempt}',flush=True)
  p=subprocess.run(['/opt/homebrew/bin/rsync','-rltzc','--partial','--timeout=600','--info=stats2','-e','ssh -o BatchMode=yes -o ServerAliveInterval=20','autodl-r4:/root/autodl-tmp/octorl_r4/'+scope,str(out)+'/'])
  if p.returncode==0: break
 else: raise RuntimeError('rsync retries exhausted')
from concurrent.futures import ThreadPoolExecutor
with ThreadPoolExecutor(max_workers=4) as pool:
 list(pool.map(transfer,scopes))
for row in manifest['files']:
 p=out/row['path']; h=hashlib.sha256()
 with p.open('rb') as f:
  for c in iter(lambda:f.read(8*1024*1024),b''): h.update(c)
 row['local_sha256']=h.hexdigest()
 assert p.stat().st_size==row['size'] and row['local_sha256']==row['remote_sha256'],row['path']
assert set(str(p.relative_to(out)) for p in out.rglob('*') if p.is_file() and not p.is_symlink())=={r['path'] for r in manifest['files']}
for link in manifest['symlinks']:
 p=out/link['path']; assert p.is_symlink() and str(p.readlink())==link['target']
latest=json.loads(subprocess.run(['ssh','-o','BatchMode=yes','autodl-r4','export PATH=/root/miniconda3/bin:$PATH; python3 -'],input=script,text=True,capture_output=True,check=True).stdout)
assert latest['files']==[{k:v for k,v in row.items() if k!='local_sha256'} for row in manifest['files']]
assert latest['symlinks']==manifest['symlinks'] and latest['remote_find_counts']==manifest['remote_find_counts']
manifest['remote_rechecked_after_transfer']=True
manifest['total_files']=len(manifest['files']); manifest['total_bytes']=sum(r['size'] for r in manifest['files']); manifest['all_matched']=True
for s in scopes: assert sum(r['path'].startswith(s+'/') and not r['symlink'] for r in manifest['files'])==manifest['remote_find_counts'][s]
(root/'r4/checkpoint_archive_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
(root/'r4r/deletion_list.txt').write_text(''.join('/root/autodl-tmp/octorl_r4/'+s+'\n' for s in scopes))
print(manifest['total_files'],manifest['total_bytes'],flush=True)
