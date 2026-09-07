import json,time,subprocess,os
from pathlib import Path
import psutil
D=Path('/root/autodl-tmp/s5')
for row,args in [('A1',['bash',str(D/'run_a.sh'),'A1','1','1','5']),('A4',['bash',str(D/'run_a.sh'),'A4','4','1','5']),('A8',['bash',str(D/'run_a.sh'),'A8','4','2','3']),('R8',['python',str(D/'run_r.py'),'8']),('R16',['python',str(D/'run_r.py'),'16'])]:
 subprocess.run(['ray','stop','--force'],stdout=subprocess.DEVNULL,stderr=subprocess.STDOUT)
 time.sleep(3)
 start=time.time()
 with (D/f'{row}.log').open('w') as f:
  result=subprocess.run(args,stdout=f,stderr=subprocess.STDOUT)
 (D/f'{row}.exit').write_text(str(result.returncode))
 (D/f'{row}_time.json').write_text(json.dumps({'start':start,'end':time.time(),'returncode':result.returncode,'seed':20260906}))
 print(row,result.returncode,flush=True)
 if result.returncode: break
