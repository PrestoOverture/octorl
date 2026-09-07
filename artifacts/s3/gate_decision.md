# S3 no-SFT baseline — skip SFT

**Decision: skip SFT.** S4 is skipped; proceed to S5. No S5 measurement was run as part of this contract.

Measured the supplied base Qwen3-4B weights using vLLM 0.10.2, native tool calling, default thinking, temperature 1.0, eight seeded rollouts per task, and a twelve-turn limit. No fine-tuning or training pipeline was used.

## Aggregate measurements

- **Tool-call format validity:** 886/890 = **99.55%** of attempting assistant turns.
- **Contract-defined “pass@1”:** 6/10 = **60.0%** of tasks had at least one successful rollout among eight. Conventionally this is pass@8, not greedy pass@1.
- **Mixed-group ratio:** 6/10 = **60.0%** had 0 < K < 8.
- **Additional per-rollout success rate:** 10/80 = **12.5%** at temperature 1.0.

## Per-task results

| Task | Successful rollouts K/8 | Contract “pass@1” (K > 0) | Per-rollout success | Mixed group | Valid / attempted turns |
|---|---:|---:|---:|---|---:|
| 01_billing | 0/8 | 0 | 0.0% | no | 93/94 (98.9%) |
| 02_access | 1/8 | 1 | 12.5% | yes | 82/82 (100.0%) |
| 03_pages | 1/8 | 1 | 12.5% | yes | 93/94 (98.9%) |
| 04_ranges | 0/8 | 0 | 0.0% | no | 95/95 (100.0%) |
| 05_temperature | 3/8 | 1 | 37.5% | yes | 75/76 (98.7%) |
| 06_inventory | 2/8 | 1 | 25.0% | yes | 88/88 (100.0%) |
| 07_shipping | 2/8 | 1 | 25.0% | yes | 79/79 (100.0%) |
| 08_scores | 0/8 | 0 | 0.0% | no | 95/95 (100.0%) |
| 09_parsing | 1/8 | 1 | 12.5% | yes | 90/91 (98.9%) |
| 10_lookup | 0/8 | 0 | 0.0% | no | 96/96 (100.0%) |

## Verification

All ten 50–150-line fixtures passed buggy-fails / exact-revert-passes checks locally and remotely. All 80 rollouts completed. The independent audit verified task membership, eight unique rollout seeds per task, temperature, turn limits, script/manifest hashes, raw-response classifications, and aggregate arithmetic. All 80 final sources were independently replayed against the original test files, with all recorded rewards reproduced. No test-integrity violations occurred. Exact known-answer zero, all-one, and mixed-group controls passed.

Completion reasons: {'step_limit': 69, 'final_response': 11}. Output-token-limited turns: 14; context-limited trajectories: 0. Calls executed by tool: {'search_code': 41, 'read_file': 88, 'apply_patch': 702, 'run_tests': 42, 'list_files': 13}. Patch execution outcomes: {'rejected': 687, 'applied': 15}. These execution outcomes are separate from call-schema validity.

## Counting convention and limitations

One assistant turn is one call attempt if it contains a native call tag, JSON name/arguments/function-call pattern, or explicit invocation of a known tool, including malformed patterns. Plain prose without these markers and thinking blocks (including unfinished thinking-only turns) are excluded. A multi-call turn is valid only if every call has native delimiters, parses as JSON, and has the correct tool name and exact typed argument structure. No JSON is repaired. Correctly structured calls with invalid diff contents remain format-valid. The full reproducible convention is in `README.md`, `measure.py`, and the run metadata.

The selected ten easy tasks and visible-only grader support this S3 gate; they do not establish generalization or hidden-test robustness. The supplied contract's pass@1 terminology is retained alongside its conventional pass@8 meaning and the separately reported rollout success rate. No greedy-temperature evaluation was substituted for the requested temperature-1.0 experiment.

## Time and budget

The completed measurement took **2422.6 seconds (0.673 GPU allocation hours)**, including model loading and tool execution. Observed host allocation from first connection through measurement completion, including setup and the failed startup, was **1.164 hours**. At the provisional **2.18 CNY/hour**, estimated costs are **1.47 CNY** for the measurement and **2.54 CNY** including observed setup, against the 16 CNY ceiling. Actual AutoDL billing is not verified; prior uptime and subsequent idle time are excluded. The launcher enforced a five-hour runtime cap. No optimizer steps, training loss, or new checkpoints apply to this inference-only run.

## Deviations and setup correction

No requested task or rollout count was reduced. Implementation choices not fixed by the contract were: batch size eight; 16,384-token context; 2,048 output tokens per turn; default native thinking; top-p 1.0; top-k disabled; and the counting convention above. Any truncation is reported explicitly.

The first startup failed before producing data because a prior local model config edit had reduced the context limit from 40,960 to 3,072. The runner asserts that exact saved-original diff and restores the original value through a process-local vLLM override; shared model files and weights are unchanged. The failure and pre-fix runner are preserved in `failed_startup_01/`. A partial pilot was also stopped and excluded after a deterministic tool test revealed rejection of valid zero-context diffs. The tool was corrected and regression-tested; all seeds were restarted unchanged with one fixed runner. Unfinished thinking-only turns are explicitly excluded from call attempts. The excluded data and pre-fix runner remain in `excluded_pilot_01/`, and the diagnosis is in `tool_regression.md`. Required missing setup packages were installed (local LibCST; remote pytest, LibCST, tmux).

## Created files

Everything is under `artifacts/s3/`: `build_tasks.py`, `measure.py`, `run_remote.sh`, `audit.py`, `report.py`, `README.md`, `goal_contract.md`; `toy_repos/`, `solutions/`, `task_manifest.json`; validation and self-test evidence; raw `run_20260905/` prompts, responses, tool results, final sources/patches, metrics and metadata; `inference.log`, exit/timing files, model hashes and setup logs; `completion_audit.json`, `reward_replay.json`, `accounting.json`, and this decision record. Protected source modules, existing tests, `docs/`, `CLAUDE.md`, and `AGENTS.md` were not edited.
