# OctoRL — preregistered agentic-RL experiments on failure-driven curricula

Does practising more on what an agent currently fails at make RL post-training better, when the budget is
held equal? And can a model that has been trained this way design its own curriculum, as a first RSI-related test?
OctoRL answers both questions with preregistered, budget-matched experiments. The model is Qwen3-4B, trained with
GRPO + LoRA on a single RTX 4090, and every result is judged on sealed held-out tests. Negative and null results are
reported as they came out.

## Headline results

Primary endpoint: **test2**, 160 held-out fault instances × 4 rollouts from service families never seen in
training, plus 40 normal instances as a guardrail. Fault-recovery success rate (%):

| Model | seed 42 | seed 137 | seed 2718 |
|---|---:|---:|---:|
| Base Qwen3-4B | 61.25 | 61.25 | 61.25 |
| Fixed-distribution GRPO | 61.09 | 60.63 | 66.09 |
| Failure-driven GRPO | 63.59 | 64.38 | 69.06 |

| Question | Per-seed difference (pp) | Pooled, 95% CI (pp) | Preregistered verdict |
|---|---|---|---|
| **Q2**: failure-driven − fixed | +2.50 / +3.75 / +2.97 | **+3.07 [+1.72, +4.53]** | `Q2_failure_driven_better` |
| **Q1**: fixed − base | −0.16 / −0.63 / +4.84 | +1.35 [+0.10, +2.81] | `no_evidence_of_improvement` (not all seeds > 0) |
| **Q4 (R6)**: does training make the model a better curriculum designer? | — | chooser TV 0.003 vs. threshold 0.10 | `Q4_mechanism_not_exercised` (stopped before training) |

- The Q2 gain is concentrated in one fault type, missing-dependency (+9.5 pp on 52 instances). This is the type the
  failure-driven selector up-weighted (42.5–55% of training rows, against 31% under the fixed distribution).
- CIs are instance-cluster bootstraps. They cover test-instance and evaluation-sampling uncertainty only; the
  training-seed variance is **not** estimated (n = 3).
- R6 is a bounded negative measurement. On identical inputs, the trained checkpoint chooses practice mixes that are
  indistinguishable from the base model's (the difference is below sampling noise). Task training did not transfer
  into curriculum design, so the recursive mechanism was never exercised. This says nothing about RSI in general.
- Total GPU spend for the whole study was about ¥144 (about US$20).

## Study design

**Environment** (`src/tasks/tool_recovery/`). A multi-turn tool-use task in which the agent repairs a service
configuration. It has three tools (`read_config`, `query_info`, `submit_fix`) and at most 5 tool calls per episode.
There are three fault types (constraint violation, missing dependency, stale version) plus normal instances, and ten
service families, split 6 train / 2 dev / 2 test. Reward is binary and computed by a verifier. Every instance is
seed-regenerable, and content fingerprints guarantee zero overlap between the train, dev and test sets. The environment
is a pure-Python state machine; the Agent-R1 adapter is `src/adapters/tool_recovery_agent_r1.py`.

**Training.** Qwen3-4B (revision `1cfa9a7`) with veRL 0.7.0, Agent-R1 and vLLM 0.10.2. GRPO uses a sign advantage
(A = 2r − 1), 4 groups × 4 rollouts per update, and temperature 1.5. LoRA is r = 16 / α = 32, and the learning rate is
2e-5 on a cosine schedule with a 100-step horizon.

**Comparison.** Each seed (42, 137, 2718) forks both arms from a shared 20-update checkpoint. Each arm then trains a
further 60 updates in 6 stages, which is 240 task groups / 960 rollouts per arm; the budget is matched in raw groups.
The fixed arm samples the pool distribution π0. The failure-driven arm re-weights each stage toward failing fault types
with `p_c = 0.4·π0_c + 0.6·π0_c·f_c / Σ_k π0_k·f_k`, where `f_c` is the smoothed failure rate over the previous 20
updates and every type is capped at 60%.

**Decision rules**, frozen before any test data existed. A comparison is "better" only if all three per-seed
differences are > 0, the pooled 95% CI lower bound is > 0, and the normal-instance pass rate stays within 5 pp of base.
Evaluation uses identical per-(instance, rollout) sampling seeds for every model.

## How the study protected itself

- **Preregistration with hash locks.** Each phase has a frozen contract in `artifacts/self_improve/contracts/`.
  Amendments are dated and state which data had been seen when they were made. The run queues refuse to start if the
  contract's hash changes.
- **Audit-caught protocol deviation → full re-run.** An offline audit found that veRL 0.7.0 silently overwrote the
  learning-rate horizon in staged training (the logged config is printed *before* the overwrite). The original R4 run
  therefore trained at about 10% of the planned cumulative learning rate. R4 was re-run as **R4r** with code-enforced gates:
  every logged LR matched a reference curve within 1e-12, and task selection was byte-identical to the frozen selector.
  R4r was evaluated on a freshly generated test set, because R4's test had already been opened. R4's verdicts are kept only
  as a record of the deviation.
- **Futility gates before spending.** R6 checked, before any training, whether the mechanism it wanted to test was
  actually present. It was not, and the check cost about ¥2 instead of about ¥50 of training that would have measured noise.
- **Synthetic controls for every statistic.** These include A/A runs giving CI [0, 0], null coverage of 0.95, and an
  inversion test that caught a spec error in which the R6 input-sensitivity formula ran in the wrong direction.

## Reproduce the numbers (offline, no GPU)

The per-rollout evaluation outputs (`artifacts/self_improve/r4r/test_eval/`) are in the repository. Both commands below
reproduce the frozen analysis and gate files byte-for-byte; this was verified on 2026-09-30.

```bash
uv sync --extra dev
for endpoint in test2 test_r3; do
  extra=(); [ "$endpoint" = test_r3 ] && extra=(--secondary)
  args=()
  for m in base fixed_42 fixed_137 fixed_2718 failure_driven_42 failure_driven_137 failure_driven_2718; do
    args+=(--model "$m=artifacts/self_improve/r4r/test_eval/$endpoint/$m.json")
  done
  uv run scripts/self_improve/r4r_analysis.py "${args[@]}" "${extra[@]}" \
    --out-json /tmp/r4r_$endpoint.json --out-md /tmp/r4r_$endpoint.md
  cmp /tmp/r4r_$endpoint.json artifacts/self_improve/r4r/r4r_analysis_$endpoint.json && echo "$endpoint reproduced"
done

# R6 futility gate from the saved chooser completions
uv run scripts/self_improve/r6_offline_replay.py --phase g2 \
  --completions artifacts/self_improve/r6/replay_real/g1/completions.jsonl --output-dir /tmp/r6_g2
cmp /tmp/r6_g2/g2_metrics.json artifacts/self_improve/r6/replay_real/g2/final14/g2_metrics.json && echo "G2 reproduced"
```

Adapter checks run on CPU only: `uv run scripts/self_improve/r5_load_adapter.py --check-only`. The report verifier is
`uv run scripts/self_improve/r5_verify_report.py`. LoRA weights are not in Git; all 20 adapters are indexed with
sha256 values in `artifacts/self_improve/r5/checkpoint_manifest.json`.

## Where things are

| Path | Contents |
|---|---|
| `artifacts/self_improve/contracts/` | Frozen preregistrations (R3, R4, R4r, R6) |
| `artifacts/self_improve/r4r/RESULTS.md` | Primary Q1/Q2 results and diagnostics |
| `artifacts/self_improve/r5/R5_report.md` | Provenance, training diagnostics, costs, limitations, adapter manifest |
| `artifacts/self_improve/r6/R6_report.md` | Model-as-curriculum-designer extension and its futility gate |
| `src/tasks/tool_recovery/` | The environment: generator, state machine, verifier, records |
| `src/curriculum/` | Failure-driven selector (`failure_driven.py`) and model chooser (`model_chooser.py`) |
| `scripts/self_improve/` | Stage runners, gates, evaluation, analysis (`r3*` … `r6*`) |

Some older artifacts link to `docs/*.md`. Those are the author's private design notes and are not published.

## Limitations

- n = 3 training seeds. Seed-to-seed variance is visible: on Q1, seed 2718 is +4.8 pp while the other two seeds are about 0.
- One synthetic task family, one 4B model, one short training budget (80 updates). The effect is modest and
  concentrated in one fault type.
- The failure-driven result is evidence about *this selection rule under this budget*. It is not evidence of general self-improvement.
- An exploratory gap is unexplained: the R3c U80 checkpoint scores 64.7% versus 60.6% for the fixed arm of the same
  seed. The learning rate is ruled out; the candidates are row order, per-stage restarts and seed noise.

---

# Part II — repository-repair RL environment (earlier work)

This was the project's first phase. The training study above uses a separate, shorter task. This environment and its
artifacts (`artifacts/p1/`, `artifacts/p2/`) are kept as a standalone deliverable.

A repository-level code-repair environment for RL research on a single 24 GB GPU:
a seeded bug injector, a pooled Docker sandbox, five frozen tools, and a
property-based verifier with an admission gate that refuses tasks whose defect no
test detects.

The environment is a deliverable in its own right and does not depend on any RL
result. It is framework-neutral: the core imports no training framework, and each
integration lives behind its own adapter.

## Install

```bash
uv sync                        # core: libcst, jsonschema
uv sync --extra dev            # + pytest, hypothesis, numpy, scipy
uv pip install verifiers       # optional: the Verifiers adapter
```

Docker is required — the sandbox executes model-written Python. Build the image
and record its digest:

```bash
docker build -t octorl-sandbox:p1 docker/
docker images --no-trunc --format '{{.ID}}' octorl-sandbox:p1 > artifacts/p1/docker-image.id
```

## Quickstart

```python
from pathlib import Path
from src.environment.harness import ScriptedPolicy, admit, package, run_trajectory
from src.environment.sandbox import SandboxPool
from src.injector.core import Difficulty

image = Path("artifacts/p1/docker-image.id").read_text().strip()
difficulty = Difficulty("mutation", "condition-inversion", 1, "L0", "single-function")
instance = package(Path("repos/train/record_index"), difficulty, seed := 11, Path("/tmp/task"))

with SandboxPool(image, size=1) as pool:
    assert admit(instance, pool)["admitted"]          # the defect is test-detected
    with pool.lease(instance.root) as sandbox:
        record = run_trajectory(
            instance, sandbox, ScriptedPolicy([{"name": "run_tests", "arguments": {}}]),
            sampling_seed=seed, trajectory_id="demo", policy_version="scripted",
            base_model_revision="none", manifest_ref="demo")
print(record["reward"], record["cheat_flags"])
```

`ScriptedPolicy` replays a fixed tool-call list. Any callable with the signature
`(prompt, history) -> {"name", "arguments"} | None` is a policy.

## Through the Verifiers interface

```python
from src.adapters.verifiers_env import load_environment

env = load_environment(pool, [{"repo": "repos/train/record_index",
                               "difficulty": {"source": "mutation", "type": "condition-inversion",
                                              "count": 1, "hint": "L0", "span": "single-function"},
                               "injector_seed": 11}])
results = env.evaluate(client, model="Qwen/Qwen3-4B", rollouts_per_example=8)
```

`OctoRLEnv` subclasses `verifiers.StatefulToolEnv`; the sandbox handle is injected
per rollout and stripped from the advertised tool schema. An Agent-R1 adapter lives
alongside it at `src/adapters/agent_r1.py`.

## Baseline evaluation

```bash
python scripts/baseline_eval.py                                   # admitted census only
python scripts/baseline_eval.py --base-url http://localhost:8000/v1 \
                                --model Qwen/Qwen3-4B --rollouts 8
```

Admission always runs first, so a baseline is never computed over instances whose
defect no test detects.

## Environment

**Task** — `d = (source, type, count, hint, span)` selects a bug; the injector is
seed-deterministic, so `(repo_ref, injector_seed, difficulty)` regenerates an
instance byte-for-byte. Instances are single-use: you cannot estimate an instance's
difficulty before deciding to train on it.

**Tools (frozen, 5)** — `list_files` · `search_code` · `read_file` · `apply_patch`
· `run_tests`. The step limit is 12 for every difficulty tier; varying it by tier
would confound training comparisons.

**Reward** — 1 iff all hidden tests pass, no cheat flag fired, and no pre-existing
visible test regressed; else 0. Hidden tests are a property-based (hypothesis)
strengthened superset of the visible suite.

**Grading integrity** — grading runs through `scripts/grading_runner.py`, which
emits structured execution events to a mount the agent cannot write. The host
compares them against test ids collected from the reference tree *before* the
agent could touch it, and requires every expected test to report setup, call and
teardown. `tests/`, `hidden_tests/` and `conftest.py` are read-only bind mounts.
This closes exit-before-collection, partial-execution and forged-summary attacks;
it is not a boundary against arbitrary in-process Python — the event writer runs
inside pytest. See `artifacts/p1/findings.md`.

**Admission** — an instance is usable only if the clean tree passes, the buggy tree
produces a genuine test failure, the known repair passes, and (multi-edit) each
advertised defect independently matters. `freeze_admitted_manifest` is the only
manifest-authoring entry point, and it writes a census of what was rejected and
why, so a later change in test coverage cannot masquerade as a change in results.

## Current limitations

| | |
|---|---|
| `count` | `count ≥ 2` works for `mutation` (363 cells) and `feat-add` (6 cells, `record_index` only). **`commit-rollback` multi-edit is unavailable** — a second same-type entry needs a second authored buggy→fixed commit pair, which the repos' two-commit history does not provide |
| `span` | **cross-file unsupported** on 7 of 8 repos — they are single-module packages (F3) |
| Repo pool | `parcel_ledger` yields no admitted `mutation` instances (its `__init__.py` is untested), but 14 admitted `commit-rollback`/`feat-add` instances against the tested `billing_rules.py` |
| Throughput gate | p95 3.76 s per 12-step trajectory, measured on macOS Docker Desktop. The intended GPU host runs no container runtime, so the gate is unresolved there (F4) |

Current admitted population: **143 requested → 136 admitted** at `c1-L0-single-function`
(`commit-rollback` 50/50, `feat-add` 50/50, `mutation` 36/43). Full census, with
per-cell rejection reasons, in `artifacts/p1/admission_report.json`.

All are tracked with reproduction steps in `artifacts/p1/findings.md`.

## Tests

```bash
python -m pytest tests/ -q      # Docker-backed tests skip without a daemon
```

## Layout

```
src/environment/   sandbox, 5 tools, verifier, hidden tests, harness, trajectory schema
src/injector/      LibCST bug injection, difficulty buckets
src/adapters/      verifiers_env.py, agent_r1.py — the only framework-aware code
harness/           the agent loop, context builder, pre-submit check, prompt (content-hashed)
repos/             6 training + 2 held-out micro-repositories
attacks/           27 seeded cheat patterns, each asserted to reward 0 end-to-end
```

## License

See `LICENSE`. Micro-repository provenance and license records are in each repo's
`injector_sources.json`.
