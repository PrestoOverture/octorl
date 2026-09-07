"""Agent-R1 spike adapter; framework imports stay outside the environment core.

The target class is loaded by Agent-R1's Hydra agent flow registry.
"""
from __future__ import annotations
import json
import os
import socket
import time
import uuid
from pathlib import Path
from typing import Any

from agent_r1.agent_flow.agent_env_loop import AgentEnvLoop
from agent_r1.agent_flow.agent_flow import AgentFlowOutput, AgentFlowStep
from agent_r1.env import AgentEnv
from agent_r1.env.base import Action, Observation
from agent_r1.env.tool_format import ToolFormatWrapper
from src.environment.record import append_record, new_record
from toy_env import ToyEnvironment

ROOT=Path(os.environ.get('OCTORL_P0_ROOT','/root/autodl-tmp/p0'))

@AgentEnv.register('octorl_toy')
class ToyAgentEnv(AgentEnv):
    def __init__(self, seed: int=20260906, **kwargs: Any):
        self.core=ToyEnvironment(seed)
        self.format=ToolFormatWrapper.from_name('hermes')
        self.messages: list[dict[str,Any]]=[]
        self.last_test=None
        self.last_events=[]

    @property
    def tool_schemas(self) -> list[dict[str,Any]]:
        return [dict(type='function',function=dict(name='read_file',description='Read clamp.py or test_visible.py. Optional 1-based inclusive line range.',parameters=dict(type='object',properties={'path':{'type':'string'},'range':{'type':'array','items':{'type':'integer'}}},required=['path']))),dict(type='function',function=dict(name='apply_patch',description='Apply a unified diff to clamp.py. Use --- a/clamp.py and +++ b/clamp.py headers.',parameters=dict(type='object',properties={'diff':{'type':'string'}},required=['diff'])))]

    def reset(self, **kwargs: Any) -> Observation:
        self.messages=list(kwargs.get('raw_prompt',[]))
        return Observation(messages=list(self.messages))

    async def step(self, action: Action) -> tuple[Observation,float|None,bool,dict[str,Any]]:
        _,calls=self.format.parse_response(action.text or '')
        self.messages.append({'role':'assistant','content':action.text or ''})
        self.last_events=[]
        for call in calls:
            self.last_events.append(self.core.call(call.name,call.arguments))
        done=not calls or any(c.name=='apply_patch' for c in calls) or self.core.steps>=12
        reward=None
        if done:
            self.last_test=self.core.verify()
            reward=float(self.last_test['passed'])
        if calls:
            self.messages.append({'role':'user','content':'\n'.join(self.format.format_observation(e['stdout']+e['stderr']) for e in self.last_events)})
        return Observation(messages=list(self.messages)),reward,done,{}

class OctoRLAgentFlow(AgentEnvLoop):
    async def run(self,sampling_params:dict[str,Any],**kwargs:Any) -> AgentFlowOutput:
        meta=kwargs.pop('_p0_trajectory')
        version=int(meta['step'])
        seed=20260906+version*10000+int(meta['sample_index'])*100+int(meta['rollout_n'])
        params={**sampling_params,'seed':seed}
        env=self._create_env(**kwargs); obs=env.reset(**kwargs)
        revision=json.loads((ROOT/'version_manifest.partial.json').read_text())['model_revision']
        record=new_record(trajectory_id=uuid.uuid4().hex,task_id=str(kwargs['extra_info']['task_id']),difficulty={'source':'mutation','type':'wrong_return','count':1,'hint':'L0','span':'single-function'},injector_seed=env.core.seed,sampling_seed=seed,repo_ref='p0/toy/clamp-v1',manifest_ref='p0-toy-train-v1',container_id=socket.gethostname()+':'+str(env.core.path),policy_version=f'p0-toy-step-{version-1}',base_model_revision=revision,harness_root=ROOT/'harness',adapter_id=f'p0-toy-step-{version-1}')
        record['policy_version_at_update']=f'p0-toy-step-{version}'
        steps=[]; metrics={'generate_sequences':0.,'tool_calls':0.}
        try:
            for index in range(self.max_steps):
                start=time.time()
                prompt=await self._obs_to_prompt(obs,tools=env.tool_schemas)
                if len(prompt)>self.prompt_length: raise RuntimeError('toy prompt exceeds configured length')
                gen_start=time.time()
                output=await self.server_manager.generate(request_id=record['trajectory_id'],prompt_ids=prompt,sampling_params=params)
                gen_end=time.time(); metrics['generate_sequences']+=gen_end-gen_start
                ids=output.token_ids[:self.response_length]
                answer=self.tokenizer.decode(ids,skip_special_tokens=self.skip_special_tokens)
                tool_start=time.time()
                obs,reward,done,info=await env.step(Action(text=answer,token_ids=ids))
                # Turn-cap termination receives the same binary verifier reward.
                if not done and index==self.max_steps-1:
                    env.last_test=env.core.verify(); reward=float(env.last_test['passed']); done=True
                end=time.time(); metrics['tool_calls']+=end-tool_start
                logs=output.log_probs[:len(ids)] if output.log_probs is not None else None
                if logs is None or len(logs)!=len(ids): raise RuntimeError('behavior logprobs missing')
                record['behavior_logprobs'].extend(logs)
                tokens={'input':len(prompt),'output':len(ids),'tool':sum(len(self.tokenizer.encode(e['stdout']+e['stderr'])) for e in env.last_events)}
                test=env.last_test
                test_record=None
                if test:
                    import re
                    passed=re.search(r'(\d+) passed',test['stdout']); failed=re.search(r'(\d+) failed',test['stdout'])
                    test_record={'exit_code':test['exit_code'],'passed':int(passed[1]) if passed else 0,'failed':int(failed[1]) if failed else 0,'stdout':test['stdout'],'stderr':test['stderr']}
                record['turns'].append(dict(index=index,start=start,end=end,assistant=answer,tool_calls=[{k:e[k] for k in ('name','arguments','stdout','stderr','exit_code')} for e in env.last_events],run_tests=test_record,token_counts=tokens))
                for k,v in tokens.items(): record['token_counts'][k]+=v
                record['wall_clock_segments'].extend([dict(name=f'generation:{index}',start=gen_start,end=gen_end),dict(name=f'tools_and_verify:{index}',start=tool_start,end=end)])
                step=AgentFlowStep(prompt_ids=prompt,response_ids=ids,response_logprobs=logs,reward_score=reward)
                steps.append(await self._postprocess(step,**kwargs))
                if done:
                    record['reward']=int(reward or 0); break
            append_record(ROOT/'trajectories.jsonl',record)
            return AgentFlowOutput(steps=steps,metrics=metrics)
        finally: env.core.close()
