"""Write the final handoff only after the independent audit passes."""
from collections import Counter
from datetime import datetime
import json
from measure import ROOT, dump


def report():
    run=ROOT/'run_20260905'
    metrics=json.loads((run/'metrics.json').read_text())
    metadata=json.loads((run/'run_metadata.json').read_text())
    audit=json.loads((ROOT/'completion_audit.json').read_text())
    assert audit['passed'] and audit['all_final_rewards_replayed']
    turns=[json.loads(x) for x in (run/'turns.jsonl').read_text().splitlines()]
    trajectories=[json.loads(x) for x in (run/'trajectories.jsonl').read_text().splitlines()]
    first_contact=datetime.fromisoformat('2026-09-05T21:35:36+08:00').timestamp()
    accounting=dict(first_observed_host_time_unix=first_contact,measurement_start_unix=metadata['started_unix'],
        measurement_end_unix=metadata['finished_unix'],measurement_hours=metadata['wall_seconds']/3600,
        allocated_host_hours_through_measurement_end=(metadata['finished_unix']-first_contact)/3600,
        hourly_rate_cny=metadata['hourly_rate_cny'],rate_is_estimate=metadata['rate_is_estimate'],
        estimated_measurement_cost_cny=metadata['estimated_cost_cny'],
        estimated_total_cost_through_measurement_end_cny=(metadata['finished_unix']-first_contact)/3600*metadata['hourly_rate_cny'],
        budget_cny=16,notes='Includes observed setup time and failed startup; excludes prior machine uptime and subsequent idle billing. Actual AutoDL invoice not available.')
    dump(ROOT/'accounting.json',accounting)
    operations=Counter()
    patch_outcomes=Counter()
    for turn in turns:
        for observation in turn['tool_observations']:
            name=observation['call']['name']
            operations[name]+=1
            if name=='apply_patch':
                result=observation['result']
                patch_outcomes['applied' if result.get('returncode')==0 else 'rejected']+=1
    rows=[]
    for task in metrics['per_task']:
        validity=f"{task['valid_calls']}/{task['total_call_attempts']} ({task['format_validity']:.1%})" if task['format_validity'] is not None else 'N/A (no attempts)'
        rows.append(f"| {task['task_id']} | {task['successes']}/8 | {task['contract_pass_at_1']} | {task['per_rollout_success']:.1%} | {'yes' if task['mixed'] else 'no'} | {validity} |")
    stops=Counter(r['stop_reason'] for r in trajectories)
    next_step={'skip SFT':'S4 is skipped; proceed to S5. No S5 measurement was run as part of this contract.',
               'format-only SFT':'S4 calls for 100–200 format/tool-protocol trajectories.',
               'reduce difficulty':'The contract calls for easier tasks and another S3 measurement before advancing.',
               'full trajectory SFT':'The contract selects full teacher-trajectory SFT in S4.'}[metrics['gate_decision']]
    text=f'''# S3 no-SFT baseline — {metrics['gate_decision']}

**Decision: {metrics['gate_decision']}.** {next_step}

Measured the supplied base Qwen3-4B weights using vLLM 0.10.2, native tool calling, default thinking, temperature 1.0, eight seeded rollouts per task, and a twelve-turn limit. No fine-tuning or training pipeline was used.

## Aggregate measurements

- **Tool-call format validity:** {metrics['valid_calls']}/{metrics['total_call_attempts']} = **{metrics['format_validity']:.2%}** of attempting assistant turns.
- **Contract-defined “pass@1”:** {metrics['solved_tasks']}/{metrics['tasks']} = **{metrics['contract_pass_at_1']:.1%}** of tasks had at least one successful rollout among eight. Conventionally this is pass@8, not greedy pass@1.
- **Mixed-group ratio:** {metrics['mixed_tasks']}/{metrics['tasks']} = **{metrics['mixed_group_ratio']:.1%}** had 0 < K < 8.
- **Additional per-rollout success rate:** {metrics['successful_rollouts']}/{metrics['rollouts']} = **{metrics['per_rollout_success']:.1%}** at temperature 1.0.

## Per-task results

| Task | Successful rollouts K/8 | Contract “pass@1” (K > 0) | Per-rollout success | Mixed group | Valid / attempted turns |
|---|---:|---:|---:|---|---:|
{chr(10).join(rows)}

## Verification

All ten 50–150-line fixtures passed buggy-fails / exact-revert-passes checks locally and remotely. All 80 rollouts completed. The independent audit verified task membership, eight unique rollout seeds per task, temperature, turn limits, script/manifest hashes, raw-response classifications, and aggregate arithmetic. All 80 final sources were independently replayed against the original test files, with all recorded rewards reproduced. No test-integrity violations occurred. Exact known-answer zero, all-one, and mixed-group controls passed.

Completion reasons: {dict(stops)}. Output-token-limited turns: {audit['length_limited_turns']}; context-limited trajectories: {audit['context_limit_trajectories']}. Calls executed by tool: {dict(operations)}. Patch execution outcomes: {dict(patch_outcomes)}. These execution outcomes are separate from call-schema validity.

## Counting convention and limitations

One assistant turn is one call attempt if it contains a native call tag, JSON name/arguments/function-call pattern, or explicit invocation of a known tool, including malformed patterns. Plain prose without these markers and thinking blocks (including unfinished thinking-only turns) are excluded. A multi-call turn is valid only if every call has native delimiters, parses as JSON, and has the correct tool name and exact typed argument structure. No JSON is repaired. Correctly structured calls with invalid diff contents remain format-valid. The full reproducible convention is in `README.md`, `measure.py`, and the run metadata.

The selected ten easy tasks and visible-only grader support this S3 gate; they do not establish generalization or hidden-test robustness. The supplied contract's pass@1 terminology is retained alongside its conventional pass@8 meaning and the separately reported rollout success rate. No greedy-temperature evaluation was substituted for the requested temperature-1.0 experiment.

## Time and budget

The completed measurement took **{metadata['wall_seconds']:.1f} seconds ({accounting['measurement_hours']:.3f} GPU allocation hours)**, including model loading and tool execution. Observed host allocation from first connection through measurement completion, including setup and the failed startup, was **{accounting['allocated_host_hours_through_measurement_end']:.3f} hours**. At the provisional **{accounting['hourly_rate_cny']:.2f} CNY/hour**, estimated costs are **{accounting['estimated_measurement_cost_cny']:.2f} CNY** for the measurement and **{accounting['estimated_total_cost_through_measurement_end_cny']:.2f} CNY** including observed setup, against the 16 CNY ceiling. Actual AutoDL billing is not verified; prior uptime and subsequent idle time are excluded. The launcher enforced a five-hour runtime cap. No optimizer steps, training loss, or new checkpoints apply to this inference-only run.

## Deviations and setup correction

No requested task or rollout count was reduced. Implementation choices not fixed by the contract were: batch size eight; 16,384-token context; 2,048 output tokens per turn; default native thinking; top-p 1.0; top-k disabled; and the counting convention above. Any truncation is reported explicitly.

The first startup failed before producing data because a prior local model config edit had reduced the context limit from 40,960 to 3,072. The runner asserts that exact saved-original diff and restores the original value through a process-local vLLM override; shared model files and weights are unchanged. The failure and pre-fix runner are preserved in `failed_startup_01/`. A partial pilot was also stopped and excluded after a deterministic tool test revealed rejection of valid zero-context diffs. The tool was corrected and regression-tested; all seeds were restarted unchanged with one fixed runner. Unfinished thinking-only turns are explicitly excluded from call attempts. The excluded data and pre-fix runner remain in `excluded_pilot_01/`, and the diagnosis is in `tool_regression.md`. Required missing setup packages were installed (local LibCST; remote pytest, LibCST, tmux).

## Created files

Everything is under `artifacts/s3/`: `build_tasks.py`, `measure.py`, `run_remote.sh`, `audit.py`, `report.py`, `README.md`, `goal_contract.md`; `toy_repos/`, `solutions/`, `task_manifest.json`; validation and self-test evidence; raw `run_20260905/` prompts, responses, tool results, final sources/patches, metrics and metadata; `inference.log`, exit/timing files, model hashes and setup logs; `completion_audit.json`, `reward_replay.json`, `accounting.json`, and this decision record. Protected source modules, existing tests, `docs/`, `CLAUDE.md`, and `AGENTS.md` were not edited.
'''
    (ROOT/'gate_decision.md').write_text(text)

if __name__=='__main__': report()
