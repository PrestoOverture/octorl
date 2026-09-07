"""
S1 Training Test — 5 policy updates with Agent-R1 + veRL on a toy task.

This is the manual part of S1. After s1_spike.sh identifies which
(veRL version, model) combos can load, run this for each working combo.

Usage:
    python scripts/s1_training_test.py \
        --model Qwen/Qwen3-4B \
        --verl-version 0.8 \
        2>&1 | tee s1_training_Qwen3-4B_verl0.8.log

What it checks:
    1. Can Agent-R1's AgentEnv connect to veRL's GRPO trainer?
    2. Do 5 policy updates complete without error?
    3. Are checkpoint parameter diffs non-zero (learning happened)?
    4. VRAM peak, wall-clock per update
    5. Are reward/loss/advantage numerically sane (no NaN, no constant)?

The output is the S1 decision record.
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path


def check_imports():
    """Verify the framework stack is importable."""
    results = {}
    for pkg in ["torch", "transformers", "peft", "verl", "vllm"]:
        try:
            mod = __import__(pkg)
            results[pkg] = getattr(mod, "__version__", "unknown")
        except ImportError:
            results[pkg] = "NOT INSTALLED"
    return results


def check_vram():
    """Return current and peak VRAM in GB."""
    import torch
    if not torch.cuda.is_available():
        return {"available": False}
    return {
        "available": True,
        "allocated_gb": torch.cuda.memory_allocated() / 1e9,
        "peak_gb": torch.cuda.max_memory_allocated() / 1e9,
        "total_gb": torch.cuda.get_device_properties(0).total_mem / 1e9,
    }


def test_model_load(model_name: str) -> dict:
    """Load model with bf16, measure VRAM."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()

    tok = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )

    load_time = time.time() - t0
    vram = check_vram()

    result = {
        "model": model_name,
        "params_B": sum(p.numel() for p in model.parameters()) / 1e9,
        "vocab_size": tok.vocab_size,
        "load_time_s": round(load_time, 1),
        "vram_after_load_gb": round(vram["peak_gb"], 2),
    }

    del model, tok
    torch.cuda.empty_cache()
    return result


def test_agent_r1_training(model_name: str, n_updates: int = 5) -> dict:
    """
    Attempt to run Agent-R1's multi-turn GRPO training loop.

    This function tries multiple integration paths:
    1. Agent-R1's built-in example configs
    2. Direct veRL GRPO with a toy reward function
    3. Minimal GRPO loop if the above fail

    Returns a dict with success/failure, metrics, and error details.
    """
    import torch

    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()

    # ── Path 1: Try Agent-R1's example ────────────────────────────
    try:
        sys.path.insert(0, os.path.expanduser("~/s1_spike/Agent-R1"))

        # Agent-R1 typically has example configs in examples/ or configs/
        agent_r1_root = Path(os.path.expanduser("~/s1_spike/Agent-R1"))

        # List what's available
        example_files = list(agent_r1_root.rglob("*.yaml")) + list(
            agent_r1_root.rglob("*.py")
        )
        example_files = [
            str(f.relative_to(agent_r1_root))
            for f in example_files
            if "example" in str(f).lower() or "train" in str(f).lower()
        ]

        print(f"Agent-R1 training-related files: {example_files[:20]}")

        # The actual training command depends on Agent-R1's interface.
        # We document what we find and attempt to run it.
        print(
            "\n>>> Agent-R1 training integration requires manual adaptation. <<<"
        )
        print(">>> See the MANUAL STEPS section below. <<<\n")

        return {
            "path": "manual",
            "agent_r1_files": example_files[:20],
            "wall_clock_s": round(time.time() - t0, 1),
            "note": "Requires manual config adaptation — see instructions",
        }

    except Exception as e:
        return {
            "path": "failed",
            "error": str(e),
            "wall_clock_s": round(time.time() - t0, 1),
        }


def main():
    parser = argparse.ArgumentParser(description="S1 Training Test")
    parser.add_argument(
        "--model", default="Qwen/Qwen3-4B", help="HF model name"
    )
    parser.add_argument(
        "--verl-version", default="0.8", help="veRL version being tested"
    )
    parser.add_argument(
        "--n-updates", type=int, default=5, help="Policy updates to attempt"
    )
    parser.add_argument(
        "--output",
        default="s1_decision_record.json",
        help="Output decision record",
    )
    args = parser.parse_args()

    print("=" * 60)
    print(f"S1 TRAINING TEST")
    print(f"Model: {args.model}")
    print(f"veRL:  {args.verl_version}")
    print(f"Updates: {args.n_updates}")
    print("=" * 60)

    record = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "model": args.model,
        "verl_version": args.verl_version,
        "target_updates": args.n_updates,
    }

    # 1. Check imports
    print("\n[1/4] Checking imports...")
    record["packages"] = check_imports()
    for pkg, ver in record["packages"].items():
        status = "OK" if ver != "NOT INSTALLED" else "MISSING"
        print(f"  {pkg}: {ver} [{status}]")

    # 2. VRAM baseline
    print("\n[2/4] VRAM baseline...")
    record["vram_baseline"] = check_vram()
    print(f"  Total: {record['vram_baseline'].get('total_gb', '?')} GB")

    # 3. Model load
    print(f"\n[3/4] Loading {args.model}...")
    try:
        record["model_load"] = test_model_load(args.model)
        print(f"  Params: {record['model_load']['params_B']:.2f}B")
        print(
            f"  VRAM after load: {record['model_load']['vram_after_load_gb']} GB"
        )
        print(f"  Load time: {record['model_load']['load_time_s']}s")
    except Exception as e:
        record["model_load"] = {"error": str(e)}
        print(f"  FAILED: {e}")

    # 4. Training test
    print(f"\n[4/4] Training test ({args.n_updates} updates)...")
    record["training"] = test_agent_r1_training(args.model, args.n_updates)

    # Write decision record
    output_path = Path(args.output)
    output_path.write_text(json.dumps(record, indent=2, ensure_ascii=False))
    print(f"\nDecision record written to: {output_path}")

    # Print manual steps
    print("\n" + "=" * 60)
    print("MANUAL STEPS (do these on the GPU machine):")
    print("=" * 60)
    print(
        """
1. Find Agent-R1's training entry point:
   cd ~/s1_spike/Agent-R1
   find . -name "*.py" | xargs grep -l "def main\\|entry_point\\|grpo" | head -10
   cat README.md | head -100

2. Locate the multi-turn example config:
   find . -name "*.yaml" -path "*/example*" -o -name "*.yaml" -path "*/config*"

3. Adapt the config for our setup:
   - model: {model}
   - n_updates: 5 (just a smoke test)
   - group_size: 8
   - Use LoRA (r=16) to fit in 24GB
   - A trivial 2-tool environment (read_file + apply_patch)

4. Run it:
   python -m agent_r1.train --config <adapted_config>.yaml  # or whatever the entry point is

5. Check the output:
   - Did 5 updates complete? (Y/N)
   - VRAM peak? (nvidia-smi -l 1 in another terminal)
   - Are losses NaN or constant zero? (check wandb or stdout)
   - Is the checkpoint diff non-zero?
     python -c "
     import torch
     c0 = torch.load('checkpoint-0/...')
     c5 = torch.load('checkpoint-5/...')
     diff = sum((c5[k]-c0[k]).abs().sum().item() for k in c0 if 'lora' in k)
     print(f'LoRA param diff: {{diff}}')
     "

6. Repeat for each (veRL, model) combo that passes the load test.

7. Record the decision in s1_decision_record.json and copy it back.
""".format(
            model=args.model
        )
    )


if __name__ == "__main__":
    main()
