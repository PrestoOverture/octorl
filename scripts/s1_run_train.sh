#!/bin/bash
# OctoRL S1 Falsification Spike — FSDP2 + offload (v2.6 config)
# Qwen3-4B + Agent-R1@b124aa4 + veRL 0.7.0 + vllm 0.10.2, single RTX 4090D (24GB)
# LoRA r=32 (veRL recommends >=32; r=16 converges slowly).
# FSDP2 + param_offload: the FSDP1 offload was a measured no-op (s1_decision_record.md §3a).
export PATH=/root/miniconda3/bin:$PATH
export HF_ENDPOINT=https://hf-mirror.com
export CUDA_VISIBLE_DEVICES=0
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY

MODEL=/root/autodl-tmp/models/Qwen3-4B
cd /root/Agent-R1

python3 -m agent_r1.trainer.main_agent_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files=/root/data/gsm8k_tool/train.parquet \
    data.val_files=/root/data/gsm8k_tool/test.parquet \
    data.train_batch_size=4 \
    data.max_prompt_length=2048 \
    data.max_response_length=384 \
    data.filter_overlong_prompts=True \
    data.truncation=error \
    data.return_raw_chat=True \
    actor_rollout_ref.model.path=$MODEL \
    actor_rollout_ref.model.lora_rank=32 \
    actor_rollout_ref.model.lora_alpha=64 \
    actor_rollout_ref.model.target_modules=all-linear \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.strategy=fsdp2 \
    actor_rollout_ref.actor.optim.lr=1e-5 \
    actor_rollout_ref.actor.ppo_mini_batch_size=2 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=2 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.fsdp_config.strategy=fsdp2 \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.offload_policy=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.actor.fsdp_config.model_dtype=bfloat16 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=2 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.load_format=safetensors \
    actor_rollout_ref.rollout.layered_summon=True \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.85 \
    actor_rollout_ref.rollout.max_model_len=3072 \
    actor_rollout_ref.rollout.enforce_eager=True \
    actor_rollout_ref.rollout.max_num_batched_tokens=2048 \
    actor_rollout_ref.rollout.n=4 \
    actor_rollout_ref.rollout.prompt_length=2048 \
    actor_rollout_ref.rollout.response_length=384 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=2 \
    actor_rollout_ref.rollout.agent.agent_flow_config_path=/root/Agent-R1/recipes/gsm8k/base.yaml \
    actor_rollout_ref.rollout.agent.default_agent_flow=gsm8k_tool \
    actor_rollout_ref.rollout.agent.max_steps=3 \
    custom_reward_function.path=recipes/gsm8k/reward_fn.py \
    custom_reward_function.name=compute_score \
    trainer.project_name=s1_spike \
    trainer.experiment_name=qwen3_4b_grpo_fsdp2 \
    trainer.logger='["console"]' \
    trainer.n_gpus_per_node=1 \
    trainer.nnodes=1 \
    trainer.save_freq=-1 \
    trainer.test_freq=-1 \
    trainer.total_training_steps=5 \
    trainer.total_epochs=1
