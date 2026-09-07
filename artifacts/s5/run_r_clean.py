"""S5 R-row: vLLM standalone inference throughput measurement.
Does NOT import Agent-R1 to avoid the subprocess-spawning bug in run_r.py.
Implements the GSM8K agent loop directly with vLLM + tokenizer.
"""
import os
os.environ['VLLM_WORKER_MULTIPROC_METHOD'] = 'spawn'
os.environ['PYTHONHASHSEED'] = '20260906'
import json, time, sys, random, re
from pathlib import Path
import numpy as np, torch, pyarrow.parquet as pq
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer

TOOLS = [{"type": "function", "function": {
    "name": "calculate", "description": "Calculate a math expression",
    "parameters": {"type": "object",
                   "properties": {"expression": {"type": "string", "description": "Math expression to evaluate"}},
                   "required": ["expression"]}}}]

def calc(expr):
    try:
        return str(eval(expr, {"__builtins__": {}}, {}))
    except Exception as e:
        return f"Error: {e}"

def parse_tool_call(text):
    m = re.search(r'<tool_call>\s*(\{.*?\})\s*</tool_call>', text, re.DOTALL)
    if not m:
        return None
    try:
        tc = json.loads(m.group(1))
        return tc
    except json.JSONDecodeError:
        return None

def main():
    n = int(sys.argv[1])
    seed = 20260906
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)

    model = '/root/autodl-tmp/models/Qwen3-4B'
    tok = AutoTokenizer.from_pretrained(model)
    llm = LLM(model=model, dtype='bfloat16', gpu_memory_utilization=0.92,
              enforce_eager=True, max_model_len=3072, max_num_seqs=n, seed=seed)

    data = pq.read_table('/root/data/gsm8k_tool/train.parquet').to_pylist()
    min_traj = 20
    batches = []
    trajectories = []
    total_start = time.time()

    for batch_idx in range((min_traj + n - 1) // n):
        active = []
        batch_start = time.time()
        for j in range(n):
            idx = batch_idx * n + j
            if idx >= len(data):
                break
            row = data[idx]
            msgs = [{"role": "user", "content": row['prompt']}]
            active.append({"idx": idx, "msgs": msgs, "turns": 0, "tokens": 0, "start": time.time()})

        for turn in range(12):
            if not active:
                break
            prompts = []; params_list = []; ready = []
            for s in active:
                ids = tok.apply_chat_template(s['msgs'], tools=TOOLS,
                                              add_generation_prompt=True, tokenize=True)
                if len(ids) > 2048:
                    s['stop'] = 'prompt_limit'
                    s['wall'] = time.time() - s['start']
                    trajectories.append({k: v for k, v in s.items() if k != 'msgs'})
                    continue
                ready.append(s)
                prompts.append({'prompt_token_ids': ids})
                params_list.append(SamplingParams(temperature=1.0, top_p=1.0, top_k=-1,
                                                  max_tokens=1024, seed=seed + s['idx'] * 12 + turn))
            if not ready:
                break
            outputs = llm.generate(prompts, params_list, use_tqdm=False)
            active = []
            for s, out in zip(ready, outputs):
                ids = out.outputs[0].token_ids
                text = tok.decode(ids, skip_special_tokens=False)
                s['turns'] += 1
                s['tokens'] += len(ids)
                tc = parse_tool_call(text)
                if tc and turn < 11:
                    expr = tc.get('arguments', {}).get('expression', '0')
                    result = calc(expr)
                    s['msgs'].append({"role": "assistant", "content": text})
                    s['msgs'].append({"role": "tool", "content": result})
                    active.append(s)
                else:
                    s['stop'] = 'done' if not tc else 'step_limit'
                    s['wall'] = time.time() - s['start']
                    trajectories.append({k: v for k, v in s.items() if k != 'msgs'})

        batch_wall = time.time() - batch_start
        batches.append({"start": batch_start, "end": time.time(),
                        "n_traj": sum(1 for t in trajectories if t.get('idx', -1) >= batch_idx * n)})
        print(json.dumps({"batch": batch_idx, "N": n, "wall": round(batch_wall, 2)}), flush=True)

    total_end = time.time()
    Path(f'/root/autodl-tmp/s5/R{n}_raw.json').write_text(json.dumps({
        "seed": seed, "start": total_start, "end": total_end,
        "batches": batches,
        "trajectories": [{"idx": t["idx"], "turns": t["turns"], "tokens": t["tokens"],
                          "wall": t["wall"], "stop": t["stop"]} for t in trajectories]
    }, indent=2))
    print(json.dumps({"done": True, "total_wall": round(total_end - total_start, 2),
                      "total_traj": len(trajectories)}), flush=True)

if __name__ == '__main__':
    main()
