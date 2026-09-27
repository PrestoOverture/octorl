#!/bin/bash
set -euo pipefail

# R4: one 10-update stage after a shared U20 warm-up.
# A resume view exposes optimizer/extra only; data.pt is deliberately absent.

if [ "$#" -ne 7 ]; then
    echo "usage: r4_train_stage.sh SEED ARM STAGE START_UPDATE TRAIN_PARQUET PREVIOUS_CKPT RUN_ROOT" >&2
    exit 2
fi

SEED="$1"
ARM="$2"
STAGE="$3"
START_UPDATE="$4"
TRAIN_PARQUET="$5"
PREVIOUS_CKPT="$6"
RUN_ROOT="$7"
case "$ARM" in fixed|failure_driven) ;; *) echo "invalid arm: $ARM" >&2; exit 2;; esac
case "$SEED" in 42|137|2718) ;; *) echo "invalid seed: $SEED" >&2; exit 2;; esac
if [ "$STAGE" -lt 1 ] || [ "$STAGE" -gt 6 ] || [ "$START_UPDATE" -ne $((20 + 10 * (STAGE - 1))) ]; then
    echo "invalid stage boundary: stage=$STAGE start=$START_UPDATE" >&2
    exit 2
fi
test -f "$TRAIN_PARQUET"
TARGET_STEPS=$((START_UPDATE + 10))
# Agent-R1 computes current_epoch = global_steps // len(stage_dataloader).
# With 40 rows and batch 4, this value is START_UPDATE/10.  Permit one epoch.
TOTAL_EPOCHS=$((START_UPDATE / 10 + 1))
RESUME_MODE=resume_path
RESUME_VIEW="$RUN_ROOT/resume_view/global_step_${START_UPDATE}"
test -f "$PREVIOUS_CKPT/actor/optim_world_size_1_rank_0.pt"
test -f "$PREVIOUS_CKPT/actor/extra_state_world_size_1_rank_0.pt"
test -f "$PREVIOUS_CKPT/actor/lora_adapter/adapter_model.safetensors"
mkdir -p "$RESUME_VIEW/actor"
ln -sfn "$PREVIOUS_CKPT/actor/optim_world_size_1_rank_0.pt" "$RESUME_VIEW/actor/optim_world_size_1_rank_0.pt"
ln -sfn "$PREVIOUS_CKPT/actor/extra_state_world_size_1_rank_0.pt" "$RESUME_VIEW/actor/extra_state_world_size_1_rank_0.pt"
test ! -e "$RESUME_VIEW/data.pt"
# veRL serialized a regex target_modules string as a list of characters in
# R3c's adapter_config.json.  Repair only the private resume view, never the
# checkpoint, so PEFT loads the exact previous adapter weights.
ADAPTER_VIEW="$RUN_ROOT/resume_view/lora_adapter"
mkdir -p "$ADAPTER_VIEW"
ln -sfn "$PREVIOUS_CKPT/actor/lora_adapter/adapter_model.safetensors" "$ADAPTER_VIEW/adapter_model.safetensors"
python3 - "$PREVIOUS_CKPT/actor/lora_adapter/adapter_config.json" "$ADAPTER_VIEW/adapter_config.json" <<'PY'
import json
import sys
from pathlib import Path
config = json.loads(Path(sys.argv[1]).read_text())
modules = config.get("target_modules")
if isinstance(modules, list) and modules and all(isinstance(x, str) and len(x) == 1 for x in modules):
    config["target_modules"] = "".join(modules)
Path(sys.argv[2]).write_text(json.dumps(config, indent=2) + "\n")
PY
LORA_ARGS=("actor_rollout_ref.model.lora_adapter_path=$ADAPTER_VIEW")
RESUME_ARGS=("trainer.resume_from_path=$RESUME_VIEW")
SAVE_FREQ=10
R3_ROOT=/root/octorl_r3
MODEL=/root/autodl-tmp/models/Qwen3-4B
CHECKPOINT_ROOT="$RUN_ROOT/checkpoints"

export PATH=/root/miniconda3/bin:$PATH
export HF_ENDPOINT=https://hf-mirror.com
export CUDA_VISIBLE_DEVICES=0
export PYTHONHASHSEED="$SEED"
export PYTHONPATH="$R3_ROOT:/root/Agent-R1"
export R3_TRAJECTORY_PATH="$RUN_ROOT/trajectories.jsonl"
export R4_RUN_ID="r4_seed_${SEED}_${ARM}"
export R4_ARM="$ARM"
export R4_SEED="$SEED"
export R4_STAGE="$STAGE"
export VERL_FILE_LOGGER_PATH="$RUN_ROOT/metrics_target_${TARGET_STEPS}.jsonl"
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY

mkdir -p "$RUN_ROOT" "$CHECKPOINT_ROOT" "$RUN_ROOT/rollouts"
date -Iseconds > "$RUN_ROOT/started_at.txt"
cd /root/Agent-R1

VALIDATION_ARGS=()
if [ "${R4_VALIDATE_ONLY:-0}" = 1 ]; then
    VALIDATION_ARGS=(+trainer.r4_validation_only=true "+trainer.r4_validation_output=$RUN_ROOT/validation.json")
fi

set +e
python3 -m agent_r1.trainer.main_agent_ppo \
    algorithm.adv_estimator=sign \
    algorithm.use_kl_in_reward=False \
    data.train_files="$TRAIN_PARQUET" \
    data.val_files="$R3_ROOT/artifacts/self_improve/r3/data/dev.parquet" \
    data.train_batch_size=4 \
    data.max_prompt_length=4096 \
    data.max_response_length=4096 \
    data.filter_overlong_prompts=True \
    data.truncation=error \
    data.return_raw_chat=True \
    data.shuffle=False \
    data.seed="$SEED" \
    actor_rollout_ref.model.path="$MODEL" \
    actor_rollout_ref.model.trust_remote_code=True \
    actor_rollout_ref.model.lora_rank=16 \
    actor_rollout_ref.model.lora_alpha=32 \
    "${LORA_ARGS[@]}" \
    'actor_rollout_ref.model.target_modules=".*\.(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)$"' \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.strategy=fsdp2 \
    actor_rollout_ref.actor.optim.lr=2e-5 \
    actor_rollout_ref.actor.optim.lr_scheduler_type=cosine \
    actor_rollout_ref.actor.optim.lr_warmup_steps=5 \
    actor_rollout_ref.actor.optim.total_training_steps=100 \
    actor_rollout_ref.actor.optim.clip_grad=1.0 \
    actor_rollout_ref.actor.ppo_mini_batch_size=4 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.actor.clip_ratio=0.2 \
    actor_rollout_ref.actor.clip_ratio_low=0.2 \
    actor_rollout_ref.actor.clip_ratio_high=0.2 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.04 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0.0 \
    actor_rollout_ref.actor.data_loader_seed="$SEED" \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.offload_policy=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.actor.fsdp_config.model_dtype=bfloat16 \
    'actor_rollout_ref.actor.checkpoint.load_contents=[optimizer,extra]' \
    'actor_rollout_ref.actor.checkpoint.save_contents=[optimizer,extra]' \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.load_format=safetensors \
    actor_rollout_ref.rollout.layered_summon=True \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.85 \
    actor_rollout_ref.rollout.max_model_len=8192 \
    actor_rollout_ref.rollout.enforce_eager=True \
    actor_rollout_ref.rollout.max_num_batched_tokens=4096 \
    actor_rollout_ref.rollout.max_num_seqs=16 \
    actor_rollout_ref.rollout.n=4 \
    actor_rollout_ref.rollout.temperature=1.5 \
    actor_rollout_ref.rollout.top_p=0.95 \
    actor_rollout_ref.rollout.prompt_length=4096 \
    actor_rollout_ref.rollout.response_length=4096 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.rollout.free_cache_engine=True \
    actor_rollout_ref.rollout.calculate_log_probs=True \
    +actor_rollout_ref.rollout.engine_kwargs.vllm.seed="$SEED" \
    actor_rollout_ref.rollout.agent.agent_flow_config_path="$R3_ROOT/artifacts/self_improve/contracts/r3_agent_flow.yaml" \
    actor_rollout_ref.rollout.agent.default_agent_flow=octorl_tool_recovery \
    actor_rollout_ref.rollout.agent.max_steps=5 \
    actor_rollout_ref.rollout.agent.num_workers=16 \
    custom_reward_function.path="$R3_ROOT/scripts/self_improve/r3_zero_reward.py" \
    custom_reward_function.name=compute_score \
    trainer.project_name=octorl_r4 \
    trainer.experiment_name="r4_seed_${SEED}_${ARM}_stage_${STAGE}" \
    trainer.logger='["console","file"]' \
    trainer.n_gpus_per_node=1 \
    trainer.nnodes=1 \
    trainer.save_freq="$SAVE_FREQ" \
    trainer.test_freq=-1 \
    trainer.total_training_steps="$TARGET_STEPS" \
    trainer.total_epochs="$TOTAL_EPOCHS" \
    trainer.val_before_train=False \
    trainer.resume_mode="$RESUME_MODE" \
    "${RESUME_ARGS[@]}" \
    trainer.default_local_dir="$CHECKPOINT_ROOT" \
    trainer.rollout_data_dir="$RUN_ROOT/rollouts" \
    "${VALIDATION_ARGS[@]}"

TRAIN_EXIT=$?
set -e
echo "TRAIN_EXIT=$TRAIN_EXIT" > "$RUN_ROOT/exit.txt"
date -Iseconds > "$RUN_ROOT/finished_at.txt"
exit "$TRAIN_EXIT"
