"""Independent completeness audit and local replay of every final patch."""
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import tempfile
from measure import ROOT, BASE_SEED, aggregate, digest, dump, parse_turn, tests


def audit(run_name='run_20260905'):
    run=ROOT/run_name
    manifest=json.loads((ROOT/'task_manifest.json').read_text())
    metadata=json.loads((run/'run_metadata.json').read_text())
    records=[json.loads(line) for line in (run/'trajectories.jsonl').read_text().splitlines()]
    turns=[json.loads(line) for line in (run/'turns.jsonl').read_text().splitlines()]
    prompts=[json.loads(line) for line in (run/'prompts.jsonl').read_text().splitlines()]
    validation=json.loads((ROOT/'task_validation.json').read_text())
    assert metadata['status']=='complete'
    assert metadata['versions']['vllm']=='0.10.2' and metadata['inference_only'] is True
    assert metadata['temperature']==1 and metadata['group_size']==8 and metadata['max_steps']==12
    assert metadata['script_sha256']==digest(ROOT/'measure.py')
    assert metadata['manifest_sha256']==digest(ROOT/'task_manifest.json')
    assert len(manifest)>=10 and len(validation)==len(manifest)
    assert all(row['valid'] and row['buggy']['returncode']==1 and row['correct']['returncode']==0 and row['exact_revert'] for row in validation)
    assert len(records)==8*len(manifest) and len({r['trajectory_id'] for r in records})==len(records)
    assert len({r['seed'] for r in records})==len(records)
    assert len(turns)==len(prompts)==sum(r['steps'] for r in records)
    expected_prompts={(p['trajectory_id'],p['step']) for p in prompts}
    assert expected_prompts=={(t['trajectory_id'],t['step']) for t in turns}
    for index,task in enumerate(manifest):
        group=[r for r in records if r['task_id']==task['task_id']]
        assert {r['rollout_index'] for r in group}==set(range(8))
        assert len((ROOT/'toy_repos'/task['task_id']/task['module']).read_text().splitlines())==task['source_lines']
        assert 50<=task['source_lines']<=150 and task['count']==1 and task['hint']=='L0' and task['span']=='single_function'
        for row in group:
            assert row['temperature']==1.0 and row['seed']==BASE_SEED+index*100+row['rollout_index']
            own=[t for t in turns if t['trajectory_id']==row['trajectory_id']]
            assert [t['step'] for t in own]==list(range(1,row['steps']+1)) and 1<=row['steps']<=12
            assert row['tests_intact']
            for turn in own:
                assert turn['parse']==parse_turn(turn['raw_output'])
                assert turn['seed']==row['seed']*100+turn['step']-1
            assert row['attempt_turns']==sum(t['parse']['attempted'] for t in own)
            assert row['valid_turns']==sum(t['parse']['valid'] for t in own)
    assert aggregate(records)==json.loads((run/'metrics.json').read_text())
    def replay(row):
        task=next(t for t in manifest if t['task_id']==row['task_id'])
        with tempfile.TemporaryDirectory(prefix='s3-audit-') as directory:
            repo=Path(directory)
            shutil.copytree(ROOT/'toy_repos'/row['task_id'],repo,dirs_exist_ok=True)
            (repo/task['module']).write_text(row['final_source'])
            result=tests(repo)
            observed=int(result['returncode']==0)
            return dict(trajectory_id=row['trajectory_id'],recorded_reward=row['reward'],replayed_reward=observed,
                        agrees=observed==row['reward'],test_result=result)
    with ThreadPoolExecutor(max_workers=4) as pool:
        replayed=list(pool.map(replay,records))
    dump(ROOT/'reward_replay.json',replayed)
    assert all(row['agrees'] for row in replayed)
    summary=dict(passed=True,verified_at=datetime.now(timezone.utc).isoformat(),tasks=len(manifest),
        trajectories=len(records),turns=len(turns),all_final_rewards_replayed=True,
        same_model_script_hash=True,raw_turn_counts_recomputed=True,all_seeds_verified=True,
        all_five_bug_types=sorted({t['bug_type'] for t in manifest}),
        context_limit_trajectories=sum(r['stop_reason']=='context_limit' for r in records),
        length_limited_turns=sum(t['finish_reason']=='length' for t in turns),
        test_integrity_violations=sum(not r['tests_intact'] for r in records))
    dump(ROOT/'completion_audit.json',summary)
    print(json.dumps(summary,indent=2))

if __name__=='__main__':
    audit()
