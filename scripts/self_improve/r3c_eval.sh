#!/bin/bash
set -euo pipefail

# R3c evaluation: runs both the original dev manifest and the normal guardrail.
# Usage: r3c_eval.sh SEED STEP [--lora-path PATH]

if [ "$#" -lt 2 ]; then
    echo "usage: r3c_eval.sh SEED STEP [--lora-path PATH]" >&2
    exit 2
fi

SEED="$1"
STEP="$2"
shift 2

LORA_ARG=""
CHECKPOINT_LABEL="base"
while [ "$#" -gt 0 ]; do
    case "$1" in
        --lora-path) LORA_ARG="--lora-path $2"; CHECKPOINT_LABEL="seed_${SEED}_u${STEP}"; shift 2;;
        *) echo "unknown arg: $1" >&2; exit 2;;
    esac
done

R3_ROOT=/root/octorl_r3
EVAL_ROOT=/root/autodl-tmp/octorl_r3c/evals/${CHECKPOINT_LABEL}

export PATH=/root/miniconda3/bin:$PATH
export HF_ENDPOINT=https://hf-mirror.com
export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH="$R3_ROOT:/root/Agent-R1"
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY

mkdir -p "$EVAL_ROOT"
date -Iseconds > "$EVAL_ROOT/started_at.txt"
cd /root/Agent-R1

echo "=== Dev evaluation (50 instances) ==="
python3 "$R3_ROOT/scripts/self_improve/r3b_eval.py" \
    --manifest "$R3_ROOT/artifacts/self_improve/r3/data/dev_manifest.json" \
    --output "$EVAL_ROOT/dev.json" \
    --trajectories "$EVAL_ROOT/dev_trajectories.jsonl" \
    --eval-seed 310000 \
    $LORA_ARG

echo "=== Normal guardrail evaluation (30 instances) ==="
python3 "$R3_ROOT/scripts/self_improve/r3b_eval.py" \
    --manifest "$R3_ROOT/artifacts/self_improve/r3c/data/dev_normal_guardrail_manifest.json" \
    --output "$EVAL_ROOT/normal_guardrail.json" \
    --trajectories "$EVAL_ROOT/normal_guardrail_trajectories.jsonl" \
    --eval-seed 320000 \
    $LORA_ARG

date -Iseconds > "$EVAL_ROOT/finished_at.txt"

echo "=== Summary ==="
python3 -c "
import json
dev = json.load(open('$EVAL_ROOT/dev.json'))
ng = json.load(open('$EVAL_ROOT/normal_guardrail.json'))
print(json.dumps({
    'checkpoint': '$CHECKPOINT_LABEL',
    'dev_fault_fcr': dev.get('fault_only_binary_fcr', dev.get('binary_fcr')),
    'dev_continuous': dev['continuous_reward_mean'],
    'dev_full_binary': dev.get('full_dev_binary_pass_rate', 'N/A'),
    'normal_guardrail_binary_npr': ng.get('normal_only_binary_pass_rate', ng.get('binary_npr', 'N/A')),
    'normal_guardrail_continuous': ng['continuous_reward_mean'],
    'normal_guardrail_rollouts': ng['rollout_count'],
}, indent=2))
"

echo "=== Done ==="
