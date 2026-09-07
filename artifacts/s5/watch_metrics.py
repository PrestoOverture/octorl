import time,json,re
from pathlib import Path
D=Path('/root/autodl-tmp/s5'); offsets={};buffers={}
with (D/'metric_events.jsonl').open('a',buffering=1) as out:
 while True:
  for row in ['A1','A4','A8']:
   p=D/f'{row}.log'
   if not p.exists():continue
   with p.open() as f:
    f.seek(offsets.get(row,0)); text=f.read();offsets[row]=f.tell()
   text=buffers.get(row,'')+text
   lines=text.split('\n');buffers[row]=lines.pop()
   for line in lines:
    if 'timing_s/step' in line or 'timing_raw/step' in line:
     out.write(json.dumps({'observed_at':time.time(),'row':row,'line':line})+'\n')
  time.sleep(.2)
