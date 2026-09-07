"""Minimal vLLM throughput benchmark for S5 R-row.
Measures raw generation throughput at N=8 and N=16 concurrency.
No Agent-R1, no agent loop, no tool calling — pure vLLM generate.
Simulates multi-turn by generating 3 batches of N sequences.
"""
import os, json, time, sys
os.environ['VLLM_WORKER_MULTIPROC_METHOD'] = 'spawn'
from pathlib import Path
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
import pyarrow.parquet as pq

def main():
    n = int(sys.argv[1])
    model = '/root/autodl-tmp/models/Qwen3-4B'
    tok = AutoTokenizer.from_pretrained(model)
    llm = LLM(model=model, dtype='bfloat16', gpu_memory_utilization=0.92,
              enforce_eager=True, max_model_len=3072, max_num_seqs=n, seed=42)
    data = pq.read_table('/root/data/gsm8k_tool/train.parquet').to_pylist()
    prompts = []
    for i in range(max(20, n * 3)):
        ids = tok.apply_chat_template(
            [{"role": "user", "content": data[i]['prompt']}],
            add_generation_prompt=True, tokenize=True)
        prompts.append({"prompt_token_ids": ids})
    params = SamplingParams(temperature=1.0, top_p=1.0, max_tokens=1024, seed=42)
    # Warmup
    print("warmup", flush=True)
    llm.generate(prompts[:1], params, use_tqdm=False)
    # Measure: 3 rounds of N-concurrent generation
    results = []
    total_tokens = 0
    total_start = time.time()
    for rd in range(3):
        batch = prompts[rd * n:(rd + 1) * n]
        if not batch:
            break
        t0 = time.time()
        outs = llm.generate(batch, params, use_tqdm=False)
        t1 = time.time()
        toks = sum(len(o.outputs[0].token_ids) for o in outs)
        total_tokens += toks
        r = {"round": rd, "n": len(batch), "wall_s": round(t1 - t0, 2),
             "tokens": toks, "tok_per_s": round(toks / (t1 - t0), 1)}
        results.append(r)
        print(json.dumps(r), flush=True)
    total_wall = time.time() - total_start
    summary = {"N": n, "rounds": len(results), "total_tokens": total_tokens,
               "total_wall_s": round(total_wall, 2),
               "avg_tok_per_s": round(total_tokens / total_wall, 1),
               "results": results}
    Path(f'/root/autodl-tmp/s5/bench_R{n}.json').write_text(json.dumps(summary, indent=2))
    print(json.dumps({"done": True, **{k: v for k, v in summary.items() if k != "results"}}), flush=True)

if __name__ == '__main__':
    main()
