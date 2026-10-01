# R6 report: model-chosen curriculum and the RSI-related question

Status: **closed 2026-09-30**. The pre-training futility gate G2 failed, and the user chose to stop without Q3.
Prereg: `artifacts/self_improve/contracts/r6_model_chooser_prereg.yaml`
(sha256 `5c7e879ef98128e82d9e8b56dbbf64640b16800a6393ed7233eb9d6122ace144`, including amendment A1).

## Question

R4r established `Q2_failure_driven_better`: a fixed rule that re-weights practice toward failing fault types beats
a fixed distribution. That result sits on rung 3 of the project's claims ladder (Agentic RL → failure-driven loop → mechanism adds value → RSI-related evidence). Rung 4, RSI-related
evidence, requires that *the improved system is better at driving the next round of learning*. The fd-v1 rule
cannot improve, so R6 hands the adaptive part of the mix to the model:

- `p = 0.4·π0 + 0.6·q`, where `q` is the output of a chooser that reads the same window statistics the rule reads.
- **Q3**: chooser_frozen (base model) versus fd_rule. Does model choosing beat the rule?
- **Q4**: chooser_self (the learner's own current adapter) versus chooser_frozen. Does training also make the model a
  better curriculum designer for itself? This comparison holds the learner start, the information, the budget and the prompt fixed.

## Gates before any training

| Gate | What it checks | Outcome |
|---|---|---|
| G0 (CPU) | The R6 selection path reproduces all 36 R4r stage selections byte-identically when given the rule's q or π0 | PASS |
| G1 (offline replay) | Chooser outputs are valid (≥ 0.95), and the base chooser responds to the stats (A1 input sensitivity ≥ 0.10) | PASS: validity 1.000 for all 7 choosers, sensitivity 0.168 |
| G2 (futility) | On identical inputs, the trained R4r U80 adapter chooses differently from base (pooled TV ≥ 0.10 and ≥ 2 × TV_noise) | **FAIL**: TV_identity 0.0031, TV_noise 0.0099 |

The replay used real Qwen3-4B on an RTX 4090 with vLLM 0.10.2 and 36 R4r windows, 2304 queries in all.
Six adapters were hash-checked on the remote: the R3c U20 adapters for seeds 42 and 137, the R4r warm-up U20 adapter for 2718, and the R4r failure_driven U80 adapters.
The smoke test was repeated and gave byte-identical completions. The numbers were recomputed independently from the raw completions.
Full numbers are in [G1_G2_DECISION.md](G1_G2_DECISION.md).

## Verdict

- **Q4 = `Q4_mechanism_not_exercised` (pre-training).** The trained adapter's curriculum choices differ from
  base by less than sampling noise, and about 30 times less than the preregistered threshold. A Q4 training run would have compared two
  near-identical choosers and measured only training-trajectory divergence.
- **Q3: not run.** The prereg required a fresh authorisation after a G2 failure, and the user declined (option A). The base
  chooser is nearly constant: 67% of its completions are exactly 40/40/20, and it mostly allocates 16/14/10
  (CV/MD/SV). A chooser_frozen arm would have been about the fixed arm plus 1–2 missing_dependency rows per stage.
  Q3 would largely have re-measured the known R4r Q2 contrast in reverse.

## What this does and does not show

It shows that, for this model (Qwen3-4B), this adapter scale (LoRA r16, ≤ 80 GRPO updates on tool recovery), this
prompt and this cell space (3 fault types), task training does not transfer into different curriculum preferences.
The "improved learner → better teacher" path is not exercised, so this setup cannot produce RSI-related evidence.

It does not show that model-chosen curricula cannot help, or anything about RSI in general. The chooser was never
optimised to choose. Testing the mechanism needs a chooser that is itself trained for learner gain, such as a
SEAL-style outer loop. That is a new question with its own scope and budget, and the project scope listed a full RSI system as
out of scope for this round.

## Cost

| Item | CNY |
|---|---:|
| R6 total (instance about 2026-09-29 23:45 → shutdown about 00:45, including boot, deployment, code fixes and idle; GPU generation about 5 min) | ≈ 2.2 |
| R6 authorisation (hard ceiling) | 70.0 |
| Training cost avoided by G2 (forecast) | ≈ 50 |

The cumulative new-direction spend is about ¥143.6. That is ¥141.4 through R4r/R5, plus R6. The billed total is
the AutoDL ledger's figure.

## Side results

- The R4r 2718 warm-up adapter's remote path, left `null` in the R5 manifest, is
  `/root/autodl-tmp/octorl_r4r/seed_2718/warmup/checkpoints/global_step_20/actor/lora_adapter`, with sha256 `2e97026d…`
  matching the local copy.
- The remote sha256 of the three R4r failure_driven U80 adapters and the two R3c U20 adapters match the manifest.
  The R4r fixed-arm U80 remote hashes were not re-checked.
- test3 (seeds 320000–320278; 160 fault + 40 normal instances; no fingerprint overlap with train, dev, the R3 test or test2) is sealed
  and **still unevaluated**. It remains available as a fresh held-out sample.
- Pre-launch power on test3, for reference in future designs: see `prelaunch_power.md`. The pooled 3-seed figures do not include
  training-seed variance and are an optimistic upper bound.

## Artifacts

| Path | Content |
|---|---|
| `g0_regression.json` | 36-stage byte-identity check |
| `data/test3*` | Sealed test3 |
| `prelaunch_power.{json,md}` | Power, single-seed and pooled |
| `replay_dryrun/` | Mock replay (R6-A) |
| `replay_real/{smoke,g1,g2}/` | Real replay: raw completions, metrics, logs, the failed smoke attempt 1, and the G2 serialization attempts |
| `G1_G2_DECISION.md` | Gate decision record |

Reproduce the gate numbers offline (no GPU):
`uv run scripts/self_improve/r6_offline_replay.py --phase g2 --completions artifacts/self_improve/r6/replay_real/g1/completions.jsonl --output-dir <dir>`
reproduces `replay_real/g2/final14/g2_metrics.json` byte-for-byte.
