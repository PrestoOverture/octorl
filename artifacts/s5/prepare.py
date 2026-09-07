from pathlib import Path
p=Path('/root/run_s1.sh').read_text()
for a,b in {'data.train_batch_size=4':'data.train_batch_size=$B','data.max_response_length=384':'data.max_response_length=1024','lora_rank=16':'lora_rank=32','lora_alpha=32':'lora_alpha=64','ppo_mini_batch_size=2':'ppo_mini_batch_size=1','ppo_micro_batch_size_per_gpu=2':'ppo_micro_batch_size_per_gpu=1','gpu_memory_utilization=0.60':'gpu_memory_utilization=0.85','rollout.n=4':'rollout.n=$N','rollout.response_length=384':'rollout.response_length=1024','agent.max_steps=3':'agent.max_steps=12','trainer.total_training_steps=5':'trainer.total_training_steps=$STEPS','trainer.project_name=s1_spike':'trainer.project_name=s5','trainer.experiment_name=qwen3_4b_grpo_lora':'trainer.experiment_name=$ROW'}.items():
    p=p.replace(a,b)
p=p.replace('cd /root/Agent-R1','set -euo pipefail\nROW=$1; B=$2; N=$3; STEPS=$4\nexport PYTHONHASHSEED=20260906\ncd /root/Agent-R1')
p=p.rstrip()+''' \\
    actor_rollout_ref.actor.strategy=fsdp2 \\
    actor_rollout_ref.actor.fsdp_config.offload_policy=True \\
    actor_rollout_ref.rollout.free_cache_engine=True \\
    +actor_rollout_ref.rollout.seed=20260906 \\
    actor_rollout_ref.actor.data_loader_seed=20260906 \\
    data.seed=20260906 \\
    trainer.val_before_train=False \\
    trainer.resume_mode=disable \\
    trainer.default_local_dir=/root/autodl-tmp/s5/checkpoints/$ROW
'''
Path('/root/autodl-tmp/s5/run_a.sh').write_text(p)
