---
base_model: Qwen/Qwen3-4B
library_name: peft
license: apache-2.0
tags:
  - lora
  - grpo
  - reinforcement-learning
  - agentic-rl
  - tool-use
  - curriculum-learning
---

# OctoRL: Qwen3-4B GRPO LoRA adapters (fixed vs. failure-driven curriculum)

Six LoRA adapters from the preregistered **R4r** experiment of
[OctoRL](https://github.com/PrestoOverture/octorl). The experiment asks whether practising more on the fault types
an agent currently fails at makes GRPO post-training better than a fixed task distribution under an equal budget.

These are **research artifacts for reproducing a comparison**. They are not general-purpose improved models. They were
trained and evaluated only on OctoRL's synthetic tool-recovery task: a short multi-turn task in which the agent repairs
a service configuration with three tools.

## Adapters

| Folder | Arm | Training seed | Fault-recovery success on test2 (%) |
|---|---|---:|---:|
| (base Qwen3-4B, for reference) | — | — | 61.25 |
| `r4r_fixed_42_u80` | fixed distribution | 42 | 61.09 |
| `r4r_fixed_137_u80` | fixed distribution | 137 | 60.63 |
| `r4r_fixed_2718_u80` | fixed distribution | 2718 | 66.09 |
| `r4r_failure_driven_42_u80` | failure-driven | 42 | 63.59 |
| `r4r_failure_driven_137_u80` | failure-driven | 137 | 64.38 |
| `r4r_failure_driven_2718_u80` | failure-driven | 2718 | 69.06 |

test2 is a sealed held-out test of 160 fault instances × 4 rollouts, drawn from service families never seen in training.
Every model used identical per-(instance, rollout) sampling seeds.

## What the preregistered analysis concluded

- **Failure-driven vs. fixed: `Q2_failure_driven_better`.** The difference is +2.50 / +3.75 / +2.97 pp by seed, pooled
  **+3.07 pp, 95% CI [+1.72, +4.53]**. The gain is concentrated in the missing-dependency fault type.
- **Fixed vs. base: `no_evidence_of_improvement`.** By seed −0.16 / −0.63 / +4.84 pp. GRPO with a fixed distribution did
  not reliably beat the base model.
- CIs are instance-cluster bootstraps. They cover test-instance and evaluation-sampling uncertainty only. Training-seed
  variance is **not** estimated (n = 3).
- Normal (no-fault) guardrail: every model passed, i.e. stayed at or above base − 5 pp (base 33.8%, adapters 37.5–43.8%).

Full results are in `artifacts/self_improve/r4r/RESULTS.md`, and the frozen preregistrations are in
`artifacts/self_improve/contracts/`. Both are in the GitHub repository.

## Training

- Base: `Qwen/Qwen3-4B` at revision `1cfa9a7208912126459214e8b04321603b3df60c`.
- Stack: veRL 0.7.0 + Agent-R1, vLLM 0.10.2 rollouts, single RTX 4090.
- GRPO with a sign advantage (A = 2r − 1) and binary verifier reward; 4 groups × 4 rollouts per update; temperature 1.5.
- LoRA r = 16, α = 32, dropout 0, on q/k/v/o/gate/up/down projections. LR 2e-5, cosine with a 100-step horizon.
- Both arms of a seed fork from the same 20-update checkpoint and train to update 80: 60 updates, 240 task groups, 960 rollouts.
  The fixed arm samples the pool distribution. The failure-driven arm re-weights each 10-update stage toward the fault types that failed most in the previous 20 updates.

## Loading

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

base_id, rev = "Qwen/Qwen3-4B", "1cfa9a7208912126459214e8b04321603b3df60c"
tokenizer = AutoTokenizer.from_pretrained(base_id, revision=rev)
base = AutoModelForCausalLM.from_pretrained(base_id, revision=rev, torch_dtype="auto")
model = PeftModel.from_pretrained(base, "{HF_USER}/octorl-qwen3-4b-grpo-lora",
                                  subfolder="r4r_failure_driven_42_u80")
```

For vLLM, download one folder with `huggingface_hub.snapshot_download(..., allow_patterns="r4r_failure_driven_42_u80/*")`
and pass the local path to `LoRARequest`. The reported evaluation used `scripts/self_improve/r3b_eval.py`
from the GitHub repository, with evaluation seed 510000.

## Provenance and one config change

The `adapter_model.safetensors` files are byte-identical to the archived training outputs. Their sha256 values are in
`release_manifest.json` and in the repository's `artifacts/self_improve/r5/checkpoint_manifest.json`.

`adapter_config.json` differs from the training output in two fields only:

- `target_modules`: the training stack serialised the module regex as a list of single characters, which PEFT
  cannot load. It is restored to the regex string. We checked with PEFT 0.20.0 that the restored config injects
  exactly the 504 saved tensors with matching shapes.
- `base_model_name_or_path` / `revision`: the training-time local path is replaced by the public model id and revision.

## Limitations

- Three training seeds, one synthetic task, one 4B model, 80 updates. The effect is modest and concentrated in one fault type.
- The result is evidence about one selection rule under one budget. It is not evidence of general self-improvement.
- The task's tool protocol and prompts are defined in the GitHub repository. Without that environment these adapters have no
  meaningful use.
