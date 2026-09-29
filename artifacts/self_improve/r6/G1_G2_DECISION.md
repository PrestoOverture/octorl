# R6 G1/G2 decision record

Decision by Claude Code, 2026-09-30, under prereg `r6-model-chooser-v1`
(sha256 `5c7e879ef98128e82d9e8b56dbbf64640b16800a6393ed7233eb9d6122ace144`, incl. A1).
Numbers were recomputed independently from `replay_real/g1/completions.jsonl` with a separate
parser and aggregation. They match `g1/g1_metrics.json` and `g2/final14/g2_metrics.json`.

## Run

- Real Qwen3-4B on an RTX 4090 with vLLM 0.10.2, `prompt_template_v1` (template sha256 `4945c88848dc…`).
- 36 R4r windows, 2304 queries.
- Six adapters, each hash-checked against `r5/checkpoint_manifest.json` before the first query.
  The 2718 warm-up adapter was checked against `2e97026d…`.
- The smoke test was repeated. Both repeats produced byte-identical completions (16/16).
- Smoke attempt 1 crashed on a LoRA request id outside int32. Only the id derivation was fixed; this is not a spec change.

## G1: PASS

| Criterion | Value | Threshold |
|---|---:|---:|
| Sample-level validity, all 7 choosers | 1.000 (2304/2304) | ≥ 0.95 |
| Unnormalised outputs | 0 | reported |
| Fallback stages | 0 | reported |
| Base input sensitivity, A1 formula | 0.1677 | ≥ 0.10 |

## G2: FAIL. Q4 is not run.

| Quantity | Value |
|---|---:|
| TV_identity, U80 vs base, pooled | 0.003125 (per seed 42: 0.0016, 137: 0.0026, 2718: 0.0052) |
| TV_identity, U20 vs base, pooled | 0.005035 |
| TV_noise, base vs base with noise seeds | 0.009896 |
| Gate | pooled U80 ≥ 0.10 AND ≥ 2 × TV_noise, both false |

TV_identity is about 30 times below the threshold and below sampling noise. This is not a borderline miss.
Training on the repair task left the model's curriculum choices unchanged.

Per `procedure.G2_futility.on_failure`:

- **Q4 = `Q4_mechanism_not_exercised` (pre-training).** No chooser_self or chooser_frozen branch is trained.
- **Q3 is not run automatically.** Running Q3 alone requires a fresh user authorisation.

## Descriptive: what the base chooser does

- 67% of all 2304 completions are exactly `{"constraint_violation": 40, "missing_dependency": 40, "stale_version": 20}`.
- Only 16 distinct completions occur.
- Mean base q = (0.389, 0.410, 0.201).
- On 12 of the 18 failure_driven windows the allocation is 16/14/10 (CV/MD/SV). The fixed arm gets roughly 15/12–13/12–13; the fd-v1 rule got 8–12/17–22/9–13.
- The one window where the chooser clearly follows the stats is seed 137 stage 5: q_MD = 0.60, counts 12/19/9.

A chooser_frozen arm would therefore be close to the fixed arm plus about 1–2 missing_dependency rows per stage.

## Interpretation, bounded

The chooser was never trained to choose. RL on the tool-recovery task (LoRA r16, at most 80 updates) did not transfer into a
different curriculum preference on this prompt, so the recursion path "improved learner → better teacher" is not
exercised in this setup. This is a measurement about this model, adapter scale, prompt and cell space. It says nothing
about RSI in general. A test of the mechanism would need a chooser that is itself optimised for learner gain
(a SEAL-style outer loop); that is a new question with its own contract and budget.

## Cost

Instance billed from about 2026-09-29 23:45 +0800. GPU work was about 5 minutes; the rest is boot, deployment, code fixes and idle.
Until shutdown, about ¥2 (G1_offline line ¥3.0; R6 ceiling ¥70).
