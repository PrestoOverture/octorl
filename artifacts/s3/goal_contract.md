### Goal Contract: S3 — No-SFT Baseline Measurement

### Goal

Measure the base Qwen3-4B Instruct model (no fine-tuning) on easy code-repair tasks and record the SFT gate decision per PRD §4.9. Three numbers are needed:

1. **Tool-call format validity rate** — of the model turns that attempt a tool call, what fraction parse as a valid call to one of the 5 tools with correct argument structure
2. **Pass\@1** — fraction of tasks where at least 1 of 8 rollouts achieves reward = 1 (all tests pass after the model's patch)
3. **Mixed-group ratio** — fraction of tasks where 0 < K < 8 (K = successes among 8 rollouts)

These three numbers decide whether SFT is skipped entirely, run in format-only mode, or run in full.

### Constraints

- **Model**: Qwen3-4B Instruct, **no fine-tuning**. Weights at `/root/autodl-tmp/models/Qwen3-4B` on 878机. Load via **vLLM 0.10.2** (already installed in the conda env).
- **Inference only** — no training pipeline, no Agent-R1 trainer. A standalone script using vLLM for generation and a tool-calling loop.
- **Tasks**: Build 10–15 minimal Python repos, each a single `.py` file (50–150 lines) with a corresponding `test_*.py`. Each repo has exactly one injected bug: single-function scope, count=1, one of {condition inversion, off-by-one, wrong return value, variable misuse, missing exception handling}. The bug must cause at least one test to fail, and reverting it must make all tests pass.
- **Hint level L0** (precise): the issue description names the file and function containing the bug, e.g. *"The function&#x20;**`calculate_discount`**&#x20;in&#x20;**`billing.py`**&#x20;returns incorrect results when the discount percentage exceeds 50%."*
- **Tools**: Implement the 5 frozen tools — `list_files(path)`, `search_code(pattern)`, `read_file(path, range)`, `apply_patch(diff)`, `run_tests(subset?)`. They operate on a per-rollout copy of the buggy repo. `run_tests` runs pytest with a 30s timeout. No Docker needed — process isolation via temp directories is sufficient for this spike.
- **Agent loop**: System prompt describes the task and available tools. 12-step limit. Each step = one model turn + optional tool execution. The model should be prompted to call tools using Qwen3's native tool-calling format. Temperature = 1.0, G = 8 rollouts per task.
- **Reward**: all tests in `test_*.py` pass after applying the model's patch → 1, else → 0. For this spike, hidden tests = visible tests (the property-based superset comes in P1).
- **Isolation**: Each rollout starts from a fresh copy of the buggy repo (copy to a temp dir, clean up after).
- **Machine**: 878机 (AutoDL 4090D 24GB). SSH: `ssh -p 25778 root@connect.bjb2.seetacloud.com`. Conda env at `/root/miniconda3`. All commands need `export PATH=/root/miniconda3/bin:$PATH`. Verified stack: torch 2.8.0 / vLLM 0.10.2 / veRL 0.7.0 / transformers 4.57.6 / flash-attn 2.8.3.
- **Do not modify**: `src/sampling/`, `src/diagnostics/staleness.py`, `scripts/estimator_study.py`, `scripts/fork_gate_power.py`, `scripts/fork_gate_a_rules.py`, `scripts/null_gate_power.py`, `tests/test_estimators.py`, `tests/test_staleness.py`, `docs/`.
- **Budget**: ≤ 16 CNY GPU (remaining P-1 allowance). Estimate: \~2–4 hours of 4090 time.
- **Output location**: `artifacts/s3/` — all raw logs, the measurement script, toy repos, aggregate metrics, and the gate decision.

### Success Conditions

- [ ] ≥ 10 easy tasks built, each with a verified bug (reverting the bug makes tests pass; the buggy version fails at least one test). If fewer than 10, the format-validity CI is too wide; if the bug/revert doesn't flip test outcomes, the task is broken and doesn't count.
- [ ] G = 8 rollouts completed per task at temperature 1.0 (≥ 80 total rollouts). Fewer rollouts makes the mixed-group ratio unreliable — a task with true p = 0.2 has P(all-0 or all-1 in 8 trials) = 0.19, so 8 rollouts per task is the minimum to distinguish.
- [ ] Format validity rate reported as `valid_calls / total_call_attempts` (e.g. "342/380 = 90.0%"). A "call attempt" = any model turn whose output matches or is intended as a tool-call pattern, including malformed ones. Plain-text reasoning turns that do not attempt a tool call are excluded from the denominator. If the denominator or the classification rule is ambiguous, document the counting convention chosen.
- [ ] Pass\@1 reported per task and as an aggregate. Aggregate-only hides whether success is concentrated in one task, which would make the gate decision unreliable.
- [ ] Mixed-group ratio reported: number and fraction of tasks with 0 < K < 8. If not reported, GRPO training viability cannot be assessed.
- [ ] SFT gate decision recorded, applying the PRD §4.9 rule:
  - format > 85% AND pass\@1 > 0 AND at least 1 task has a mixed group → **"skip SFT"** (S4 is skipped; proceed to S5)
  - format < 85% but pass\@1 > 0 → **"format-only SFT"** (S4 runs with 100–200 format trajectories)
  - format > 85% but pass\@1 = 0 across all tasks → **"reduce difficulty"** (rebuild easier tasks, re-run S3)
  - none of the above → **"full trajectory SFT"** (S4 runs with full teacher trajectories)
  Without an explicit decision the S4/S5 sequencing is ambiguous.
- [ ] Final handoff lists: changed/created files, the three aggregate numbers with per-task breakdown, the gate decision, GPU hours used, and any deviations from this contract.