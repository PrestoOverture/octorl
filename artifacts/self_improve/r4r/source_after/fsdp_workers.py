# Copyright 2025 Agent-R1 Teams
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Thin FSDP worker wrappers that swap in Agent-R1 local actor/critic implementations.
"""

from verl.single_controller.base.decorator import Dispatch, register
from verl.utils.config import omega_conf_to_dataclass
from verl.workers.fsdp_workers import AsyncActorRolloutRefWorker as VerlAsyncActorRolloutRefWorker
from verl.workers.fsdp_workers import CriticWorker as VerlCriticWorker


class AsyncActorRolloutRefWorker(VerlAsyncActorRolloutRefWorker):
    def _build_model_optimizer(self, *args, **kwargs):
        result = super()._build_model_optimizer(*args, **kwargs)
        if result[2] is not None and self.config.get("r4r_scheduler_log", False):
            optim = kwargs.get("optim_config", args[2] if len(args) > 2 else None)
            total = optim.get("total_training_steps", 0)
            warmup = int(optim.get("lr_warmup_steps", -1))
            if warmup < 0:
                warmup = int(optim.get("lr_warmup_steps_ratio", 0.0) * total)
            kind = optim.get("lr_scheduler_type", "constant")
            print(f"R4R_LR_SCHEDULER total_steps={total} warmup={warmup} type={kind}", flush=True)
        return result
    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def init_model(self):
        super().init_model()

        from agent_r1.workers.actor import DataParallelPPOActor

        if self._is_actor:
            actor_cfg = omega_conf_to_dataclass(self.config.actor)
            self.actor = DataParallelPPOActor(
                config=actor_cfg, actor_module=self.actor_module_fsdp, actor_optimizer=self.actor_optimizer
            )

        if self._is_ref:
            self.ref_policy = DataParallelPPOActor(config=self.config.ref, actor_module=self.ref_module_fsdp)


class CriticWorker(VerlCriticWorker):
    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def init_model(self):
        super().init_model()

        from agent_r1.workers.critic import DataParallelPPOCritic

        self.critic = DataParallelPPOCritic(
            config=self.config, critic_module=self.critic_module, critic_optimizer=self.critic_optimizer
        )
