#!/usr/bin/env python3
"""CPU adapter integrity check; optional offline base+PEFT load and generation."""
from __future__ import annotations
import argparse
import json
import math
import re
import struct
import tempfile
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MANIFEST = ROOT/'artifacts/self_improve/r5/checkpoint_manifest.json'

def normalize_modules(value):
    if isinstance(value, list) and value and all(isinstance(x, str) and len(x) == 1 for x in value):
        return ''.join(value)
    return value

def read_tensors(path):
    """Validate the safetensors header, bounds, shapes and complete payload."""
    blob = path.read_bytes()
    if len(blob) < 8:
        raise ValueError('truncated safetensors header')
    size = struct.unpack('<Q', blob[:8])[0]
    if size < 2 or size > len(blob)-8:
        raise ValueError('truncated safetensors header')
    def unique(pairs):
        out = {}
        for k,v in pairs:
            if k in out: raise ValueError('duplicate header key: '+k)
            out[k] = v
        return out
    header = json.loads(blob[8:8+size], object_pairs_hook=unique)
    payload = memoryview(blob)[8+size:]
    spans, tensors = [], {}
    dtypes = {'F64':'<f8','F32':'<f4','F16':'<f2','BF16':'<u2'}
    for key, entry in header.items():
        if key == '__metadata__': continue
        dtype = entry['dtype']; shape = entry['shape']; start,end = entry['data_offsets']
        if dtype not in dtypes or any(type(d) is not int or d <= 0 for d in shape):
            raise ValueError('unsupported tensor dtype/shape: '+key)
        dt = np.dtype(dtypes[dtype])
        if not 0 <= start <= end <= len(payload) or end-start != math.prod(shape)*dt.itemsize:
            raise ValueError('invalid/truncated tensor: '+key)
        values = np.frombuffer(payload[start:end],dtype=dt).reshape(shape)
        if dtype == 'BF16': values = (values.astype(np.uint32)<<16).view(np.float32)
        if not np.isfinite(values).all(): raise ValueError('NaN/Inf: '+key)
        tensors[key] = values
        spans.append((start,end))
    cursor=0
    for start,end in sorted(spans):
        if start != cursor: raise ValueError('overlapping or gapped tensor data')
        cursor=end
    if not tensors or cursor != len(payload): raise ValueError('empty or trailing tensor data')
    return tensors

def check_adapter(directory):
    config = json.loads((directory/'adapter_config.json').read_text())
    if config.get('r') != 16 or config.get('lora_alpha') != 32:
        raise ValueError('expected r=16 and alpha=32')
    modules = normalize_modules(config.get('target_modules'))
    if not modules: raise ValueError('missing target modules')
    tensors = read_tensors(directory/'adapter_model.safetensors')
    pairs = {}
    for key, values in tensors.items():
        match = re.fullmatch(r'(.+)\.lora_([AB])(?:\.default)?\.weight',key)
        if not match: raise ValueError('unexpected tensor key: '+key)
        module, side = match.groups()
        mapped = re.fullmatch(modules,module) if isinstance(modules,str) else any(module == m or module.endswith('.'+m) for m in modules)
        if not mapped: raise ValueError('unmapped target: '+key)
        if values.ndim != 2 or values.shape[0 if side == 'A' else 1] != 16:
            raise ValueError('rank mismatch: '+key)
        if side == 'B' and not np.any(values != 0): raise ValueError('zero lora_B: '+key)
        pairs.setdefault(module,set()).add(side)
    if any(sides != {'A','B'} for sides in pairs.values()): raise ValueError('unpaired LoRA tensors')
    return {'path':str(directory), 'tensor_count':len(tensors), 'base_model_name_or_path':config.get('base_model_name_or_path'), 'revision':config.get('revision'), 'target_modules':modules, 'serialized_character_list_repaired_in_memory':modules != config.get('target_modules')}

def check_all(manifest):
    results = [check_adapter(ROOT/row['local_path']) for row in json.loads(manifest.read_text())]
    for field in ('base_model_name_or_path','revision','target_modules'):
        if any(row[field] != results[0][field] for row in results): raise ValueError('inconsistent '+field)
    return results

def full_load(directory, base, seed, prompt):
    """Explicit opt-in only; local files, CPU, deterministic greedy generation."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
    from peft import PeftModel
    set_seed(seed)
    check_adapter(directory)
    config=json.loads((directory/'adapter_config.json').read_text())
    config['target_modules']=normalize_modules(config['target_modules'])
    with tempfile.TemporaryDirectory(prefix='r5_adapter_') as tmp:
        view=Path(tmp)
        (view/'adapter_config.json').write_text(json.dumps(config))
        (view/'adapter_model.safetensors').symlink_to((directory/'adapter_model.safetensors').resolve())
        tokenizer=AutoTokenizer.from_pretrained(base,local_files_only=True)
        model=AutoModelForCausalLM.from_pretrained(base,local_files_only=True,torch_dtype=torch.float32)
        model=PeftModel.from_pretrained(model,view,local_files_only=True).eval()
        inputs=tokenizer(prompt,return_tensors='pt')
        with torch.no_grad(): output=model.generate(**inputs,max_new_tokens=32,do_sample=False)
        print(tokenizer.decode(output[0],skip_special_tokens=True))

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check-only',action='store_true')
    parser.add_argument('--manifest',type=Path,default=DEFAULT_MANIFEST)
    parser.add_argument('--adapter',type=Path)
    parser.add_argument('--base',type=Path)
    parser.add_argument('--seed',type=int,default=42)
    parser.add_argument('--prompt',default='Describe how to recover from a failed tool call.')
    args=parser.parse_args()
    if args.check_only:
        print(json.dumps(check_adapter(args.adapter) if args.adapter else check_all(args.manifest),indent=2))
    else:
        if args.adapter is None or args.base is None: parser.error('full mode requires --adapter and --base')
        full_load(args.adapter,args.base,args.seed,args.prompt)

if __name__ == '__main__': main()
