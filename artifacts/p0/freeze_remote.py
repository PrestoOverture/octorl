from pathlib import Path
import hashlib, importlib.metadata as m, json, subprocess, urllib.request
ROOT=Path('/root/autodl-tmp/p0')
versions={k:m.version(k) for k in ['torch','vllm','verl','flash-attn','transformers','peft','accelerate','datasets','jsonschema']}
import torch
result={'versions':versions,'cuda':torch.version.cuda,'driver':subprocess.check_output(['nvidia-smi','--query-gpu=driver_version','--format=csv,noheader'],text=True).strip(),'agent_r1_commit':subprocess.check_output(['git','-C','/root/Agent-R1','rev-parse','HEAD'],text=True).strip(),'verifiers_commit':'25debce78aff23fca201cefe9c6f72dc65176d06'}
for endpoint in ['https://hf-mirror.com/api/models/Qwen/Qwen3-4B?blobs=true','https://huggingface.co/api/models/Qwen/Qwen3-4B?blobs=true']:
    try:
        info=json.load(urllib.request.urlopen(endpoint,timeout=25)); break
    except Exception as exc: print(str(exc))
else: info={}
result['model_revision']=info.get('sha')
model=Path('/root/autodl-tmp/models/Qwen3-4B')
result['model_files']={}
for p in sorted(model.glob('*.safetensors')):
    digest=hashlib.file_digest(p.open('rb'),'sha256').hexdigest()
    remote=next((x for x in info.get('siblings',[]) if x['rfilename']==p.name),{})
    result['model_files'][p.name]={'sha256':digest,'hub_sha256':remote.get('lfs',{}).get('sha256'),'matches':digest==remote.get('lfs',{}).get('sha256')}
result['model_config_sha256']=hashlib.sha256((model/'config.json').read_bytes()).hexdigest()
result['model_config_local_override']='max_position_embeddings=3072; original config retained in config.json.orig'
ROOT.joinpath('version_manifest.partial.json').write_text(json.dumps(result,indent=2))
ROOT.joinpath('pip_freeze.txt').write_text(subprocess.check_output(['python','-m','pip','freeze'],text=True))
print(json.dumps(result,indent=2))
