# OctoRL — a difficulty-parameterized repository-repair RL environment

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
docs/              PRD, architecture, roadmap, tech stack, progress
```

## License

See `LICENSE`. Micro-repository provenance and license records are in each repo's
`injector_sources.json`.

## Self-improve training study: dev-tool fault recovery

This separate study uses a short, structured tool-recovery environment under
`src/tasks/tool_recovery/`, with the Agent-R1 adapter in
`src/adapters/tool_recovery_agent_r1.py`. It is distinct from the repository-repair
environment described above. R4's pre-registered results are
`no_evidence_of_improvement` (fixed training versus base) and
`no_evidence_of_difference` (failure-driven versus fixed).

Study entry points are `scripts/self_improve/r4_analysis.py` (offline analysis),
`r5_diagnostics.py` (training logs), and `r5_load_adapter.py` (adapter checking or
loading). See [the R5 report](artifacts/self_improve/r5/R5_report.md) for exact
results, exploratory-result labels, provenance, diagnostics, costs and limitations.
Local adapter weights are excluded from Git; acquire them using `r5_pull.py` with
authorized key-only access to `autodl-r4`.

```sh
# Offline analysis; keep the frozen R4 outputs unchanged.
python scripts/self_improve/r4_analysis.py \
  --model base=artifacts/self_improve/r4/test_eval/base.json \
  --model fixed_42=artifacts/self_improve/r4/test_eval/fixed_42.json \
  --model fixed_137=artifacts/self_improve/r4/test_eval/fixed_137.json \
  --model fixed_2718=artifacts/self_improve/r4/test_eval/fixed_2718.json \
  --model failure_driven_42=artifacts/self_improve/r4/test_eval/failure_driven_42.json \
  --model failure_driven_137=artifacts/self_improve/r4/test_eval/failure_driven_137.json \
  --model failure_driven_2718=artifacts/self_improve/r4/test_eval/failure_driven_2718.json \
  --out-json artifacts/self_improve/r5/reanalysis.json \
  --out-md artifacts/self_improve/r5/reanalysis.md
# CPU-only adapter checks, no base weights needed:
python scripts/self_improve/r5_load_adapter.py --check-only
python scripts/self_improve/r5_verify_report.py
# Optional future full load; requires local base weights, torch, transformers and peft.
# This full-load command was not run for R5.
python scripts/self_improve/r5_load_adapter.py --adapter artifacts/self_improve/r5/checkpoints/r4_fixed_137_u80 --base /absolute/path/to/Qwen3-4B --seed 42
```
