---
base_model: Qwen/Qwen3-4B
library_name: peft
license: apache-2.0
tags:
  - lora
---

# OctoRL: adapter archive (private)

This repository is a private backup of the 14 OctoRL LoRA adapters that are not part of the public release
(`PrestoOverture/octorl-qwen3-4b-grpo-lora`). It holds:

- `r3c_{42,137}_u{20,50,80}`: fixed-distribution GRPO runs (R3c). The U20 checkpoints are the shared fork points for seeds 42 and 137.
- `r4_{fixed,failure_driven}_{42,137,2718}_u80` and `r4_warmup_2718_u20`: the **superseded R4 run**. Its staged
  training did not realise the preregistered learning-rate schedule (cumulative LR about 10% of plan). These are kept only as a record of that deviation.
- `r4r_warmup_2718_u20`: the R4r fork point for seed 2718.

The weights are byte-identical to the archived training outputs; see `release_manifest.json`. `adapter_config.json`
carries the same two-field repair as the public release.
