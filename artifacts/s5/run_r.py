import os
os.environ['VLLM_WORKER_MULTIPROC_METHOD']='spawn'
os.environ['PYTHONHASHSEED']='20260906'
import asyncio,json,time,sys,random
from pathlib import Path
sys.path.insert(0,'/root/Agent-R1')

def main():
 import numpy as np, torch, pyarrow.parquet as pq
 from vllm import LLM,SamplingParams
 from transformers import AutoTokenizer
 import recipes.gsm8k
 from agent_r1.env import AgentEnv
 from agent_r1.env.base import Action
 n=int(sys.argv[1]); seed=20260906
 random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
 model='/root/autodl-tmp/models/Qwen3-4B'
 tok=AutoTokenizer.from_pretrained(model)
 llm=LLM(model=model,dtype='bfloat16',gpu_memory_utilization=.92,enforce_eager=True,max_model_len=3072,max_num_seqs=n,seed=seed)
 data=pq.read_table('/root/data/gsm8k_tool/train.parquet').to_pylist()
 batches=[]; trajectories=[]
 start=time.time()
 for batch_idx in range((20+n-1)//n):
  active=[]; begin=time.time()
  for j in range(n):
   idx=batch_idx*n+j; row=data[idx]
   env=AgentEnv.from_config('tool',tools=['calc_gsm8k_reward'],tool_format='hermes',**json.loads(row['env_kwargs']))
   obs=env.reset(raw_prompt=row['prompt'])
   active.append(dict(idx=idx,env=env,obs=obs,turns=0,tokens=0,start=time.time()))
  for turn in range(12):
   ready=[]; prompts=[]; params=[]
   for s in active:
    ids=tok.apply_chat_template(s['obs'].messages,tools=s['env'].tool_schemas,add_generation_prompt=True,tokenize=True)
    if len(ids)>2048:
     s['stop']='prompt_limit'; trajectories.append({k:v for k,v in s.items() if k not in ('env','obs')}); continue
    ready.append(s); prompts.append({'prompt_token_ids':ids})
    params.append(SamplingParams(temperature=1.,top_p=1.,top_k=-1,max_tokens=1024,seed=seed+s['idx']*12+turn))
   if not ready: break
   outputs=llm.generate(prompts,params,use_tqdm=False)
   active=[]
   for s,out in zip(ready,outputs):
    ids=out.outputs[0].token_ids
    text=tok.decode(ids,skip_special_tokens=True)
    obs,reward,done,info=asyncio.run(s['env'].step(Action(text=text,token_ids=ids)))
    s['obs']=obs;s['turns']+=1;s['tokens']+=len(ids)
    if done or turn==11:
     s['stop']='done' if done else 'step_limit';s['wall_seconds']=time.time()-s['start']
     trajectories.append({k:v for k,v in s.items() if k not in ('env','obs')})
    else: active.append(s)
  batches.append(dict(start=begin,end=time.time(),trajectories=n))
  print(json.dumps({'batch':batch_idx,'N':n,'wall':batches[-1]['end']-begin}),flush=True)
 end=time.time()
 Path(f'/root/autodl-tmp/s5/R{n}_raw.json').write_text(json.dumps(dict(seed=seed,start=start,end=end,batches=batches,trajectories=trajectories),indent=2))
if __name__=='__main__': main()
