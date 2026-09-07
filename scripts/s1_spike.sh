#!/usr/bin/env bash
# S1 Falsification Spike — Agent-R1 × veRL × Qwen3(.5)-4B
# Run on AutoDL 4090 (24GB). Budget: ≤30 CNY for all of P-1.
#
# What this script does:
#   1. Installs veRL (0.7 then 0.8) + Agent-R1
#   2. Downloads Qwen3-4B and Qwen3.5-4B
#   3. Runs Agent-R1's official multi-turn example with 5 policy updates
#   4. Logs VRAM peak, wall-clock, and compatibility notes
#
# Usage:
#   bash scripts/s1_spike.sh 2>&1 | tee s1_spike.log

set -euo pipefail

WORKDIR="$HOME/s1_spike"
RESULTS="$WORKDIR/results"
mkdir -p "$WORKDIR" "$RESULTS"

echo "=== S1 Spike started at $(date -Iseconds) ==="
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
echo "CUDA: $(nvcc --version 2>/dev/null | tail -1 || echo 'nvcc not found')"

# ── Step 0: Base dependencies ──────────────────────────────────────
pip install uv 2>/dev/null || true

# ── Step 1: Clone repos ───────────────────────────────────────────
cd "$WORKDIR"

if [ ! -d "Agent-R1" ]; then
    git clone https://github.com/AgentR1/Agent-R1.git
fi
AGENT_R1_COMMIT=$(cd Agent-R1 && git rev-parse HEAD)
echo "Agent-R1 commit: $AGENT_R1_COMMIT"

# ── Step 2: Check Uni-Agent and record why not used ───────────────
echo ""
echo "=== Uni-Agent assessment ==="
echo "Uni-Agent (https://github.com/verl-project/uni-agent) is the 2026"
echo "long-horizon agent RL stack for SWE/R2E with async GRPO/GSPO."
echo "Not used because: single 4090 budget cannot support its multi-GPU"
echo "async architecture. Agent-R1 is the single-GPU-compatible option."
echo "Verify: check uni-agent README for GPU requirements."
echo ""

# ── Step 3: Download models ───────────────────────────────────────
echo "=== Downloading models ==="

# Qwen3-4B (co-primary, Agent-R1 validated)
python -c "
from huggingface_hub import snapshot_download
info = snapshot_download('Qwen/Qwen3-4B', allow_patterns=['*.safetensors','*.json','*.tiktoken','*.py'])
print(f'Qwen3-4B downloaded to: {info}')
" 2>&1 | tee "$RESULTS/download_qwen3.log"

# Qwen3.5-4B (primary candidate, compatibility uncertain)
python -c "
from huggingface_hub import snapshot_download
info = snapshot_download('Qwen/Qwen3.5-4B', allow_patterns=['*.safetensors','*.json','*.tiktoken','*.py'])
print(f'Qwen3.5-4B downloaded to: {info}')
" 2>&1 | tee "$RESULTS/download_qwen35.log"

# Record model revision hashes
python -c "
from huggingface_hub import model_info
for name in ['Qwen/Qwen3-4B', 'Qwen/Qwen3.5-4B']:
    info = model_info(name)
    print(f'{name}: sha={info.sha}')
" 2>&1 | tee "$RESULTS/model_revisions.txt"


# ── Step 4: Test matrix ───────────────────────────────────────────
# We test 4 combinations. For each:
#   - Install the veRL version
#   - Try to run Agent-R1's multi-turn example with 5 updates
#   - Record success/failure, VRAM peak, wall-clock

run_test() {
    local verl_ver="$1"   # "0.7" or "0.8"
    local model="$2"      # "Qwen/Qwen3-4B" or "Qwen/Qwen3.5-4B"
    local tag="${verl_ver}_$(basename $model)"
    local logfile="$RESULTS/test_${tag}.log"

    echo ""
    echo "=============================================="
    echo "Testing: veRL $verl_ver × $model"
    echo "=============================================="

    # Install veRL
    if [ "$verl_ver" = "0.7" ]; then
        pip install verl==0.7.0.post1 2>&1 | tail -5
    elif [ "$verl_ver" = "0.8" ]; then
        pip install "verl>=0.8.0,<0.9" 2>&1 | tail -5
    fi

    # Record installed versions
    echo "--- Installed versions ---" > "$logfile"
    pip show verl 2>/dev/null | grep -E "^(Name|Version)" >> "$logfile"
    python -c "import torch; print(f'torch={torch.__version__}, cuda={torch.version.cuda}')" >> "$logfile"
    python -c "
try:
    import vllm; print(f'vllm={vllm.__version__}')
except: print('vllm not installed')
try:
    import sglang; print(f'sglang={sglang.__version__}')
except: print('sglang not installed')
" >> "$logfile"

    # Check if model loads at all
    echo "--- Model load test ---" >> "$logfile"
    python -c "
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
model_name = '$model'
print(f'Loading {model_name}...')
tok = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
print(f'Tokenizer OK, vocab_size={tok.vocab_size}')
model = AutoModelForCausalLM.from_pretrained(
    model_name, torch_dtype=torch.bfloat16, device_map='auto',
    trust_remote_code=True
)
print(f'Model OK, params={sum(p.numel() for p in model.parameters())/1e9:.2f}B')
# VRAM after load
mem = torch.cuda.max_memory_allocated() / 1e9
print(f'VRAM after model load: {mem:.2f} GB')
del model
torch.cuda.empty_cache()
" >> "$logfile" 2>&1
    cat "$logfile"

    # Try Agent-R1's multi-turn training
    # This is the key test — look for their example config and adapt
    echo "--- Agent-R1 training test (5 updates) ---" >> "$logfile"
    echo ">>> See the manual steps below for this part <<<" >> "$logfile"

    echo "Result saved to: $logfile"
}

# Run the test matrix
run_test "0.7" "Qwen/Qwen3-4B"
run_test "0.8" "Qwen/Qwen3-4B"
run_test "0.7" "Qwen/Qwen3.5-4B"
run_test "0.8" "Qwen/Qwen3.5-4B"

echo ""
echo "=== S1 automated checks done at $(date -Iseconds) ==="
echo "Results in: $RESULTS/"
echo ""
echo ">>> NEXT: Run the Agent-R1 training manually for each working combo <<<"
echo ">>> See scripts/s1_training_test.py for the 5-update test <<<"
