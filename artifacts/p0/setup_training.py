"""Apply recorded spike-only hooks to the pinned upstream trainer."""
from pathlib import Path
import json
import pandas as pd
ROOT=Path('/root/autodl-tmp/p0')
flow=Path('/root/Agent-R1/agent_r1/agent_flow/agent_flow.py')
text=flow.read_text(); old='await agent_flow.run(sampling_params, **kwargs)'; new='await agent_flow.run(sampling_params, _p0_trajectory=trajectory, **kwargs)'
if new not in text:
    assert text.count(old)==1
    flow.write_text(text.replace(old,new))
trainer=Path('/root/Agent-R1/agent_r1/trainer/ppo/ray_trainer.py')
text=trainer.read_text()
anchor='    def _update_actor(self, batch: DataProto) -> DataProto:\n'
hook='''        # P0: real pre-update advantage metric and initialization checkpoint.
        import json as _p0_json
        from pathlib import Path as _P0Path
        _p0_mask = batch.batch["response_mask"].bool()
        _p0_adv = batch.batch["advantages"][_p0_mask].float()
        _p0_row = {"step": self.global_steps, "advantage_abs_mean": _p0_adv.abs().mean().item(), "advantage_min": _p0_adv.min().item(), "advantage_max": _p0_adv.max().item()}
        with _P0Path("/root/autodl-tmp/p0/advantages.jsonl").open("a") as _p0_file:
            _p0_file.write(_p0_json.dumps(_p0_row) + "\\n")
        if self.global_steps == 1:
            self.global_steps = 0
            self._save_checkpoint()
            self.global_steps = 1
'''
if '# P0: real pre-update' not in text:
    assert text.count(anchor)==1
    trainer.write_text(text.replace(anchor,anchor+hook))
(ROOT/'flow.yaml').write_text('- name: octorl_toy\n  _target_: src.adapters.agent_r1.OctoRLAgentFlow\n  env_type: octorl_toy\n')
prompt=[{'role':'system','content':'You repair Python code using tools. Reply with tool calls in <tool_call>{"name": "...", "arguments": {...}}</tool_call>. Use read_file to inspect and apply_patch to submit a unified diff. Do not use markdown fences. /no_think'},{'role':'user','content':'Fix clamp.py. clamp(value, lower, upper) must return value within the inclusive bounds, lower when value is below lower, upper when above upper; inverted bounds raise ValueError. A wrong function call in the return expression breaks this. Read clamp.py, then apply a unified diff. Only clamp.py is writable. Patching submits your final solution.'}]
rows=[]
for index in range(32):
    rows.append({'data_source':'octorl_toy','prompt':prompt,'agent_name':'octorl_toy','env_kwargs':json.dumps({'seed':20260906+index}),'reward_model':{'style':'rule','ground_truth':'hidden-tests'},'extra_info':{'task_id':f'p0-clamp-{index:03d}','index':index}})
pd.DataFrame(rows).to_parquet(ROOT/'train.parquet')
(ROOT/'reward.py').write_text('def compute_score(**kwargs):\n    return 0.0\n')
p=Path('/root/autodl-tmp/s5/run_a.sh').read_text()
p=p.replace('cd /root/Agent-R1','export PYTHONPATH=/root/autodl-tmp/p0:/root/Agent-R1\ncd /root/Agent-R1')
replacements={'/root/data/gsm8k_tool/train.parquet':str(ROOT/'train.parquet'),'/root/data/gsm8k_tool/test.parquet':str(ROOT/'train.parquet'),'actor_rollout_ref.actor.entropy_coeff=0':'actor_rollout_ref.actor.entropy_coeff=0.001','/root/Agent-R1/recipes/gsm8k/base.yaml':str(ROOT/'flow.yaml'),'default_agent_flow=gsm8k_tool':'default_agent_flow=octorl_toy','agent.max_steps=12':'agent.max_steps=3','custom_reward_function.path=recipes/gsm8k/reward_fn.py':'custom_reward_function.path='+str(ROOT/'reward.py'),'trainer.project_name=s5':'trainer.project_name=p0','trainer.save_freq=-1':'trainer.save_freq=5','/root/autodl-tmp/s5/checkpoints/':'/root/autodl-tmp/p0/checkpoints/'}
for a,b in replacements.items():
    assert a in p,a
    p=p.replace(a,b)
p=p.replace('    actor_rollout_ref.rollout.agent.num_workers=$WORKERS','    actor_rollout_ref.rollout.agent.num_workers=$WORKERS \\\n    actor_rollout_ref.rollout.calculate_log_probs=True \\\n    actor_rollout_ref.actor.kl_loss_coef=0.0')
(ROOT/'run_train.sh').write_text(p)
print('Setup complete; schema must already have been validated before invoking run_train.sh')
