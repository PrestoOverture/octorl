# S3 — no-SFT code-repair baseline

This directory contains the entire standalone S3 experiment. No training code is used.
The goal contract is preserved in `goal_contract.md`.

## Reproduction

1. Install `libcst==1.9.0` and `pytest==9.1.1` for fixture generation/validation.
2. Run `python build_tasks.py`, `python measure.py selftest`, and `python measure.py validate`.
3. On 878机, use the supplied Qwen3-4B weights and the verified vLLM 0.10.2 environment. Run `bash run_remote.sh` inside a tmux session. The output directory must not already exist: completed results are never silently overwritten.
4. Copy results back and run `python audit.py`. This reconstructs the counts from raw turns, verifies all seeds and group sizes, and independently replays all final implementations against the original tests.

The remote launcher assumes this directory is `/root/autodl-tmp/s3`. It adds `/root/miniconda3/bin` to PATH and has a five-hour process timeout. Its hourly-price estimate is 2.18 CNY; actual AutoDL billing must be confirmed separately.

## Protocol fixed before the final measurement

- Ten hand-authored, deterministic tasks: two each of condition inversion, off-by-one, wrong return value, variable misuse, and missing exception handling.
- Each task has one 50–150-line implementation file and one visible test file. The clean implementation and inverse patch are stored outside the model's workspace in `solutions/`. LibCST changes exactly one node inside the named function.
- Every task is validated as buggy-fails / exact-revert-passes, locally and on the inference host.
- Eight fresh temporary-directory rollouts per task; temperature 1.0; top-p 1.0; top-k disabled; native Qwen3 chat template, default thinking enabled; at most twelve model turns per rollout. Multiple tool calls in a single turn are allowed by the native template.
- Maximum context: 16,384 tokens. Maximum output per turn: 2,048 tokens. Context exhaustion ends a trajectory and grades the current patch; output exhaustion receives a continuation cue and consumes a step. These events are reported by the completion audit.
- Explicit base seed: **20260905**. Trajectory seed: `20260905 + task_index*100 + rollout_index`. Sampling seed for each turn: `trajectory_seed*100 + zero_based_turn`. Task generation itself is deterministic and does not draw random numbers.
- Calls execute in private temporary repositories. The tool layer restricts patches to the implementation file. Pytest runs with plugin autoload disabled, a thirty-second timeout, and process-group cleanup on timeout. Original test integrity is checked before final grading. Hidden tests equal visible tests, as authorized for this spike.
- Batches contain eight trajectories. vLLM inference uses the model weights directly without LoRA adapters, optimizer, trainer, or fine-tuning.

## Metric conventions

The contract's “pass@1” is the fraction of task groups with **at least one success in eight rollouts**. This is conventionally called pass@8. Artifacts name it `contract_pass_at_1`; reports show the contract label and the conventional interpretation. `per_rollout_success` separately reports total successes divided by 80. Neither number is a separate greedy-decoding evaluation.

Format validity is measured **per attempting assistant turn**: a turn with native tool-call tags, JSON function/name/arguments patterns, or an explicit invocation of a known tool counts as an attempt, including malformed calls. Thinking blocks (including unfinished thinking-only turns) are excluded. Plain prose without those patterns is excluded. A turn is valid only if all calls in it are correctly delimited native JSON, name one of the five tools, and satisfy the exact argument schema. Multiple calls count once per turn; valid object counts are also recorded. Execution failure of a correctly structured call (for example an invalid diff hunk) is distinct from format failure. `measure.CONVENTION` and the run metadata contain the complete rule. No malformed JSON is repaired silently.

Mixed-group ratio is the number of tasks with `0 < K < 8` divided by ten. The gate applies the contract's strict greater-than / less-than 85% branches exactly; exactly 85%, no call attempts, and positive success with no mixed groups reach the catch-all branch.

The self-test has exact known-answer zero-success, all-success, and mixed-group controls. No statistical estimator or Monte Carlo confidence interval is introduced. Ten selected easy tasks characterize this spike, not repository-level generalization.

## Startup correction

The first startup failed before any rollout. The shared model directory's `config.json` had a prior one-field edit reducing `max_position_embeddings` from 40,960 (in `config.json.orig`) to 3,072. vLLM rejected the requested 16,384 context. The runner checks that exact diff and applies `hf_overrides={"max_position_embeddings":40960}` for its own process. It does not alter shared model files or weights. The original error, launcher status, metadata, and pre-fix script are retained in `failed_startup_01/`. The successful vLLM initialization is the regression check for this configuration seam.

## Artifact map

- `build_tasks.py`, `toy_repos/`, `solutions/`, `task_manifest.json`: tasks, injection, and known corrections.
- `measure.py`, `run_remote.sh`: inference loop, five tools, parser, grader, aggregation, and bounded launcher.
- `task_validation.json`, `remote_validation.log`, `selftest.json`: fixture validation and known-answer controls.
- `run_20260905/`: raw prompts, turns, trajectories, final sources/patches and pytest output, prompt/tool schemas, model/runtime metadata, and metrics.
- `inference.log`, `exit_status.txt`, `job_*_unix.txt`: process evidence and timing.
- `model_sha256.txt`: model shard and configuration fingerprints.
- `audit.py`, `completion_audit.json`, `reward_replay.json`: independent completeness and reward replay evidence.
- `gate_decision.md`, `accounting.json`: final handoff and runtime/cost accounting.

All created experiment files are under `artifacts/s3/`. Protected source modules, tests, and `docs/` are unchanged. Local environment setup installed LibCST; remote setup installed pytest, LibCST, and tmux.

## Excluded pilot and tool regression

An early partial run is preserved in `excluded_pilot_01/` and excluded from all final counts. A deterministic test showed the initial patch tool rejected valid zero-context unified diffs. The corrected tool supports them and normalizes an omitted final newline, while still rejecting malformed hunk counts. An unfinished thinking-only turn is also explicitly excluded from format attempts. All final seeds restart unchanged. See `tool_regression.md`; the final run uses one fixed runner.
