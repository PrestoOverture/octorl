#!/bin/bash
set -euo pipefail

# R4 seed-2718 shared warm-up: R3c training distribution, U1-U20 only.

SEED=2718
RESUME_MODE=disable
SAVE_FREQ=20
TARGET_STEPS=20
R3_ROOT=/root/octorl_r3
MODEL=/root/autodl-tmp/models/Qwen3-4B
RUN_ROOT=/root/autodl-tmp/octorl_r4/seed_${SEED}/warmup
CHECKPOINT_ROOT="$RUN_ROOT/checkpoints"

export PATH=/root/miniconda3/bin:$PATH
export HF_ENDPOINT=https://hf-mirror.com
export CUDA_VISIBLE_DEVICES=0
export PYTHONHASHSEED="$SEED"
export PYTHONPATH="$R3_ROOT:/root/Agent-R1"
export R3_TRAJECTORY_PATH="$RUN_ROOT/trajectories.jsonl"
export R4_RUN_ID="r4_seed_${SEED}_warmup"
export R4_ARM=warmup
export R4_SEED="$SEED"
export R4_STAGE=0
export VERL_FILE_LOGGER_PATH="$RUN_ROOT/metrics_target_${TARGET_STEPS}.jsonl"
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY

mkdir -p "$RUN_ROOT" "$CHECKPOINT_ROOT" "$RUN_ROOT/rollouts"
BEGAN_UNIX=$(date +%s)
date -Iseconds > "$RUN_ROOT/started_at.txt"
cd /root/Agent-R1

set +e
python3 -m agent_r1.trainer.main_agent_ppo \
    algorithm.adv_estimator=sign \
    algorithm.use_kl_in_reward=False \
    data.train_files="$R3_ROOT/artifacts/self_improve/r3/data/train_fault_only.parquet" \
    data.val_files="$R3_ROOT/artifacts/self_improve/r3/data/dev.parquet" \
    data.train_batch_size=4 \
    data.max_prompt_length=4096 \
    data.max_response_length=4096 \
    data.filter_overlong_prompts=True \
    data.truncation=error \
    data.return_raw_chat=True \
    data.shuffle=True \
    data.seed="$SEED" \
    actor_rollout_ref.model.path="$MODEL" \
    actor_rollout_ref.model.trust_remote_code=True \
    actor_rollout_ref.model.lora_rank=16 \
    actor_rollout_ref.model.lora_alpha=32 \
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
    trainer.experiment_name="r4_seed_${SEED}_warmup" \
    trainer.logger='["console","file"]' \
    trainer.n_gpus_per_node=1 \
    trainer.nnodes=1 \
    trainer.save_freq="$SAVE_FREQ" \
    trainer.test_freq=-1 \
    trainer.total_training_steps="$TARGET_STEPS" \
    trainer.total_epochs=1 \
    trainer.val_before_train=False \
    trainer.resume_mode="$RESUME_MODE" \
    trainer.default_local_dir="$CHECKPOINT_ROOT" \
    trainer.rollout_data_dir="$RUN_ROOT/rollouts"

TRAIN_EXIT=$?
set -e
echo "TRAIN_EXIT=$TRAIN_EXIT" > "$RUN_ROOT/exit.txt"
date -Iseconds > "$RUN_ROOT/finished_at.txt"
ENDED_UNIX=$(date +%s)
python3 - "$RUN_ROOT/timing.json" "$BEGAN_UNIX" "$ENDED_UNIX" "$TRAIN_EXIT" <<'PY'
import json
import sys
from pathlib import Path
elapsed = int(sys.argv[3]) - int(sys.argv[2])
Path(sys.argv[1]).write_text(json.dumps({
    "elapsed_seconds": elapsed,
    "cost_cny": elapsed * 2.18 / 3600,
    "exit_code": int(sys.argv[4]),
}, indent=2) + "\n")
PY
if [ "$TRAIN_EXIT" -eq 0 ]; then
    test -f "$CHECKPOINT_ROOT/global_step_20/actor/optim_world_size_1_rank_0.pt"
    test -f "$CHECKPOINT_ROOT/global_step_20/actor/extra_state_world_size_1_rank_0.pt"
    test -f "$CHECKPOINT_ROOT/global_step_20/actor/lora_adapter/adapter_model.safetensors"
    if find "$CHECKPOINT_ROOT/global_step_20/actor" -maxdepth 1 -name 'model_world_size_*.pt' | grep -q .; then
        echo 'warm-up wrote an unexpected full model shard' >&2
        exit 1
    fi
    python3 "$R3_ROOT/scripts/self_improve/r4_attribute_steps.py" \
        --run-dir "$RUN_ROOT" --output "$RUN_ROOT/attributed.jsonl" \
        --start-step 1 --end-step 20
fi
exit "$TRAIN_EXIT"
