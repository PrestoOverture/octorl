#!/usr/bin/env bash
# OctoRL S1 — AutoDL environment build (verified working 2026-09-02)
#
# Image REQUIRED: PyTorch 2.8.0 / Python 3.12 / CUDA 12.8 / Ubuntu 22.04
#   The image's torch version is the binding constraint: vLLM 0.10.2 pins
#   torch==2.8.0 exactly. On a torch 2.5.1 image no vLLM release satisfies
#   veRL 0.7.0's v1-engine imports, and hand-written shims fail at runtime.
#   Pick the image first, then the vLLM release whose torch pin matches.
#
# Usage:  bash s1_setup_autodl.sh 2>&1 | tee ~/s1_setup.log
set -euo pipefail

MIRROR="https://mirrors.aliyun.com/pypi/simple/"
DATA_DISK=/root/autodl-tmp          # survives shutdown; system disk is only 30 GB
MODEL_DIR="$DATA_DISK/models/Qwen3-4B"

echo "=== S1 setup $(date '+%F %T') ==="

# ── 0. Verify the image before touching anything ──────────────────────
python3 - <<'PY'
import torch
print(f"  torch={torch.__version__}  cuda={torch.version.cuda}  gpu={torch.cuda.get_device_name(0)}")
assert torch.cuda.is_available(), "CUDA unavailable"
assert torch.__version__.startswith("2.8.0"), \
    f"expected torch 2.8.0 (vLLM 0.10.2 pins it exactly), got {torch.__version__}"
PY

export HF_ENDPOINT=https://hf-mirror.com
grep -q HF_ENDPOINT ~/.bashrc 2>/dev/null || echo 'export HF_ENDPOINT=https://hf-mirror.com' >> ~/.bashrc

# ── 1. Python stack ───────────────────────────────────────────────────
# Note: /etc/network_turbo accelerates GitHub/HuggingFace but makes the pip
# mirrors slower, so keep the proxy OFF for pip and ON only for GitHub.
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY

echo ""
echo "[1/6] vLLM 0.10.2 + veRL 0.7.0"
pip install "vllm==0.10.2" "verl==0.7.0" -i "$MIRROR" --retries 10 --timeout 300

# datasets 2.14.4 (what the resolver picks) is broken against pyarrow>=14:
# it references pa.PyExtensionType, removed upstream. Force a 3.x.
pip install -U "datasets>=3.0,<4.0" "pandas" -i "$MIRROR" --retries 10 --timeout 120

# libsndfile is a system lib pulled in transitively via mistral_common[audio];
# without it `import verl.workers.rollout...` dies with an OSError.
echo ""
echo "[2/6] libsndfile1"
DEBIAN_FRONTEND=noninteractive apt-get update -qq
DEBIAN_FRONTEND=noninteractive apt-get install -y libsndfile1

# ── 2. flash-attn: use the prebuilt wheel, never build from source ─────
echo ""
echo "[3/6] flash-attn (prebuilt: torch2.8 / cxx11abiTRUE / cp312)"
FA=flash_attn-2.8.3.post1+cu12torch2.8cxx11abiTRUE-cp312-cp312-linux_x86_64.whl
if ! python3 -c "import flash_attn" 2>/dev/null; then
    (source /etc/network_turbo >/dev/null 2>&1 || true
     curl -sSL -C - --retry 10 --retry-all-errors -o "/root/$FA" \
       "https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3.post1/$FA")
    pip install --no-deps "/root/$FA"
fi

# ── 3. Agent-R1 (run from the repo; it is not a pip package) ──────────
echo ""
echo "[4/6] Agent-R1 @ b124aa4"
if [ ! -d /root/Agent-R1 ]; then
    (source /etc/network_turbo >/dev/null 2>&1 || true
     git clone https://github.com/AgentR1/Agent-R1.git /root/Agent-R1)
fi
git -C /root/Agent-R1 checkout -q b124aa4

# ── 4. Model weights onto the data disk ───────────────────────────────
echo ""
echo "[5/6] Qwen3-4B -> $MODEL_DIR"
mkdir -p "$MODEL_DIR" && cd "$MODEL_DIR"
BASE=https://hf-mirror.com/Qwen/Qwen3-4B/resolve/main
for f in config.json generation_config.json merges.txt vocab.json tokenizer.json \
         tokenizer_config.json model.safetensors.index.json \
         model-0000{1,2,3}-of-00003.safetensors; do
    [ -s "$f" ] || curl -sSL -C - --retry 10 --retry-all-errors --retry-delay 2 -o "$f" "$BASE/$f"
done

# veRL discards actor_rollout_ref.rollout.max_model_len — vllm_async_server.py:198
# overwrites it with hf_config.max_position_embeddings. The model config is the
# only working control. rope_scaling is null, so this just bounds the position
# range and is a no-op for sequences below the limit.
python3 - <<'PY'
import json, os, shutil
p = os.path.join(os.environ.get("MODEL_DIR", "/root/autodl-tmp/models/Qwen3-4B"), "config.json")
if not os.path.exists(p + ".orig"):
    shutil.copy(p, p + ".orig")
c = json.load(open(p))
c["max_position_embeddings"] = 3072
json.dump(c, open(p, "w"), indent=2)
print(f"  max_position_embeddings -> 3072 (original kept at config.json.orig)")
PY

python3 - <<'PY'
import json, os
from safetensors import safe_open
D = os.environ.get("MODEL_DIR", "/root/autodl-tmp/models/Qwen3-4B")
idx = json.load(open(f"{D}/model.safetensors.index.json"))
shards = sorted(set(idx["weight_map"].values()))
n = sum(len(safe_open(os.path.join(D, s), framework="pt").keys()) for s in shards)
assert n == len(idx["weight_map"]), f"tensor count {n} != index {len(idx['weight_map'])}"
print(f"  {len(shards)} shards, {n} tensors verified")
PY

# ── 5. GSM8K tool dataset (use the recipe's own preprocessor) ─────────
echo ""
echo "[6/6] GSM8K tool data"
cd /root/Agent-R1
python3 -m recipes.gsm8k.data_preprocess.process_gsm8k_tool --local_save_dir /root/data/gsm8k_tool

# ── Verify the import chain that actually matters ─────────────────────
echo ""
echo "=== verification ==="
python3 - <<'PY'
import torch, vllm, verl, transformers, flash_attn, ray
for m in (torch, vllm, verl, transformers, flash_attn, ray):
    print(f"  {m.__name__:14} {m.__version__}")
# This exact import is what defeated the vLLM 0.7.3 attempt. If it succeeds,
# no shims or site-packages patches are needed.
from verl.workers.rollout.vllm_rollout.vllm_async_server import vLLMReplica
print(f"  vLLMReplica    import OK  <- the decisive check")
PY

echo ""
echo "=== setup complete. Next: bash scripts/s1_run_train.sh ==="
