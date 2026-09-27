#!/usr/bin/env python3
"""R4 load-only gate, replacing the cross-backend HF logprob threshold.

adapter-equal: the LoRA the resumed FSDP actor saved must be bitwise equal to the source adapter
               (fails if PEFT skipped or mis-mapped any module on load). CPU only.
base-tokens:   greedy tokens of the BASE model (no LoRA) on the validation prompts must differ from
               the U20 reference tokens, otherwise the training-vLLM token match proves nothing. GPU.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _normalise(key: str) -> str:
    # veRL/PEFT exports may differ only in the wrapper prefix and the adapter name segment.
    for prefix in ("base_model.model.", "model."):
        if key.startswith(prefix):
            key = key[len(prefix):]
    return key.replace(".default.", ".")


def adapter_equal(source: Path, saved: Path) -> dict:
    import torch
    from safetensors.torch import load_file

    a = {_normalise(k): v for k, v in load_file(source / "adapter_model.safetensors").items()}
    b = {_normalise(k): v for k, v in load_file(saved / "adapter_model.safetensors").items()}
    only_a, only_b = sorted(set(a) - set(b)), sorted(set(b) - set(a))
    mismatched = [k for k in sorted(set(a) & set(b))
                  if a[k].shape != b[k].shape or a[k].dtype != b[k].dtype or not torch.equal(a[k], b[k])]
    nonzero_b = sum(int(v.abs().sum().item() > 0) for k, v in a.items() if "lora_B" in k)
    return {"tensors": len(a), "only_in_source": only_a[:5], "only_in_saved": only_b[:5],
            "mismatched": mismatched[:5], "n_mismatched": len(mismatched),
            "nonzero_lora_B_tensors": nonzero_b,
            "pass": not only_a and not only_b and not mismatched and nonzero_b > 0}


def base_tokens(validation: Path, model: Path, seed: int) -> dict:
    from vllm import LLM, SamplingParams
    from vllm.inputs import TokensPrompt

    result = json.loads(validation.read_text(encoding="utf-8"))
    llm = LLM(model=str(model), trust_remote_code=True, max_model_len=8192,
              gpu_memory_utilization=0.85, enforce_eager=True, seed=seed)
    outs = llm.generate([TokensPrompt(prompt_token_ids=x["prompt_ids"]) for x in result["examples"]],
                        sampling_params=SamplingParams(temperature=0.0, max_tokens=64, ignore_eos=True),
                        use_tqdm=False)
    base = [list(o.outputs[0].token_ids) for o in outs]
    train, ref = result["training_vllm_tokens"], result["reference_vllm_tokens"]
    first_diff = [next((i for i, (x, y) in enumerate(zip(b, r)) if x != y), None) for b, r in zip(base, ref)]
    return {"base_vllm_tokens": base, "first_base_vs_u20_divergence": first_diff,
            "training_equals_u20": train == ref,
            "base_differs_from_u20": any(i is not None for i in first_diff),
            "pass": train == ref and any(i is not None for i in first_diff)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="mode", required=True)
    p = sub.add_parser("adapter-equal")
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--saved", type=Path, required=True)
    p = sub.add_parser("base-tokens")
    p.add_argument("--validation", type=Path, required=True)
    p.add_argument("--model", type=Path, default=Path("/root/autodl-tmp/models/Qwen3-4B"))
    p.add_argument("--seed", type=int, required=True)
    args = parser.parse_args()
    out = (adapter_equal(args.source, args.saved) if args.mode == "adapter-equal"
           else base_tokens(args.validation, args.model, args.seed))
    print(json.dumps({k: v for k, v in out.items() if k != "base_vllm_tokens"}, sort_keys=True))
    sys.exit(0 if out["pass"] else 1)


if __name__ == "__main__":
    main()
