#!/usr/bin/env python3
"""Read remote scheduler source and preserve numbered evidence, without imports."""
import json
from r5_pull import remote,OUT
value=remote('''import pathlib,json,re,hashlib
roots=[pathlib.Path('/root/Agent-R1'),pathlib.Path('/root/miniconda3/lib/python3.12/site-packages/verl')]
files=[]
for root in roots:
 for name in ('ray_trainer.py','fsdp_workers.py','torch_functional.py'):
  for p in root.rglob(name):
   lines=p.read_text().splitlines(); selected=set()
   for i,line in enumerate(lines):
    if re.search(r'total_training_steps|lr_scheduler.step|get_last_lr|num_training_steps|load_state_dict',line): selected.update(range(max(0,i-5),min(len(lines),i+8)))
   files.append({'path':str(p),'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'lines':[{'line':i+1,'text':lines[i]} for i in sorted(selected)]})
print(json.dumps(files))''')
(OUT/'raw/scheduler_source_evidence.json').write_text(json.dumps(value,indent=2)+'\n')
for file in value:
 print(file['path'])
 for line in file['lines']:
  print(str(line['line'])+': '+line['text'])
