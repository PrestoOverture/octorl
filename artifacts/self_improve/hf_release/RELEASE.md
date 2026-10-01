# Hugging Face release record (2026-10-01)

| Repo | Visibility | HF commit | Files |
|---|---|---|---|
| [PrestoOverture/octorl-qwen3-4b-grpo-lora](https://huggingface.co/PrestoOverture/octorl-qwen3-4b-grpo-lora) | **public** since 2026-10-01, made public together with the GitHub repo and checked anonymously (page 200, weight download sha256 matches) | `ffaa150e` | 6 R4r U80 adapters + card + manifest |
| PrestoOverture/octorl-qwen3-4b-lora-archive | private (anonymous access returns 401, checked 2026-10-01) | `34b2aadf` | 14 other adapters + card + manifest |

Staged by `scripts/self_improve/hf_release_stage.py`. Content commit: `f98b1e6`.
Uploaded with `huggingface_hub.HfApi.upload_folder`.

## Verification

All checks below were run with the HF API after upload.

- **File trees match the staging folders exactly.** The public repo has 14 files and the archive repo has 30, plus `.gitattributes` in each. Nothing is missing and nothing is extra.
- **Weights are byte-identical across all three records.** For each of the 20 `adapter_model.safetensors`, the LFS sha256 on HF equals the sha256 of the local staged file and the value in `release_manifest.json`. The staging script had already checked those against `artifacts/self_improve/r5/checkpoint_manifest.json`.
- **Small files are byte-identical.** All 24 small files (configs, cards, manifests) were downloaded back, and their sha256 matches the local copies.
- **The model card's loading code works.** It was run as written: `PeftModel.from_pretrained(base, repo, subfolder=...)` with PEFT 0.20.0 and a meta-device Qwen3-4B at revision `1cfa9a7`. All 504 LoRA modules were injected. Loading a second adapter by subfolder reported 0 unexpected and 0 missing keys.
- **The config repair is necessary and correct.** The original char-list `target_modules` fails in PEFT, as expected. Each of the 20 repaired configs injects exactly the 504 saved tensors, with matching shapes.
