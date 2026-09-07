"""S3 inference-only experiment, replayable raw logs, and deterministic aggregation.

Run validate, selftest, run, or summarize. No training imports or parameter updates.
"""
from __future__ import annotations
import argparse
from concurrent.futures import ThreadPoolExecutor
import difflib
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import re
import shutil
import signal
import subprocess
import sys
import tempfile
import time

ROOT = Path(__file__).resolve().parent
MODEL = '/root/autodl-tmp/models/Qwen3-4B'
BASE_SEED = 20260905
TOOLS = []

def tool(name, description, properties, required):
    TOOLS.append({'type': 'function', 'function': {'name': name, 'description': description,
        'parameters': {'type': 'object', 'properties': properties, 'required': required,
                       'additionalProperties': False}}})

tool('list_files', 'List repository files under a relative directory.', {'path': {'type':'string'}}, ['path'])
tool('search_code', 'Search repository Python files with a Python regular expression.', {'pattern': {'type':'string'}}, ['pattern'])
tool('read_file', 'Read a repository file. range is [first_line, last_line], one-based inclusive, or null for all lines.',
     {'path': {'type':'string'}, 'range': {'anyOf':[{'type':'array','items':{'type':'integer'},'minItems':2,'maxItems':2},{'type':'null'}]}}, ['path', 'range'])
tool('apply_patch', 'Apply a standard unified diff to the implementation .py file. Include --- a/file.py, +++ b/file.py, and valid @@ line-count hunks. Test files cannot be changed.', {'diff':{'type':'string'}}, ['diff'])
tool('run_tests', 'Run pytest (30s limit). Omit subset to run all tests, or supply a test filename or filename::test_name.', {'subset':{'type':'string'}}, [])
SYSTEM = '''You are repairing a small Python repository. Fix the issue by inspecting the code and using the tools to apply a patch. Preserve unrelated behavior. The tests describe the expected behavior. You may edit only the implementation Python file; never edit tests or the test framework. Tools operate in your private repository copy. Use the native <tool_call> JSON format shown below. You have at most 12 assistant turns. Each turn may call tools; tool results follow. When finished, give a short final response. Prefer inspecting the named function, applying a focused unified diff, and running tests.'''
CONVENTION = '''A call attempt is one assistant turn with a native tool_call tag (including malformed/unclosed tags), a JSON name/arguments or tool_calls/function_call pattern, or an explicit invocation of a known tool (e.g. read_file(...)). Completed <think> reasoning blocks are excluded. Plain prose without those patterns is excluded. A turn is valid iff it contains one or more native <tool_call> JSON objects, all with known names and exact typed argument schemas, and no unmatched call tags. Multiple calls in one turn count once in both numerator and denominator; object counts are also reported. Tool execution errors, including a malformed unified diff, do not make a correctly typed tool call format-invalid. Unknown or extra argument keys are invalid. No silent repair or JSON coercion is performed.'''


def dump(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n')


def append(path, value):
    with path.open('a') as stream:
        stream.write(json.dumps(value, ensure_ascii=False) + '\n')
        stream.flush()


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def command(argv, cwd, timeout=30, input_text=None):
    started = time.monotonic()
    env = dict(os.environ, PYTEST_DISABLE_PLUGIN_AUTOLOAD='1', PYTHONHASHSEED=str(BASE_SEED), PYTHONDONTWRITEBYTECODE='1')
    proc = subprocess.Popen(argv, cwd=cwd, env=env, stdin=subprocess.PIPE,
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
                            start_new_session=True)
    timed_out = False
    try:
        stdout, stderr = proc.communicate(input_text, timeout=timeout)
    except subprocess.TimeoutExpired:
        timed_out = True
        os.killpg(proc.pid, signal.SIGKILL)
        stdout, stderr = proc.communicate()
    return dict(argv=argv, returncode=proc.returncode, stdout=stdout, stderr=stderr,
                timed_out=timed_out, wall_seconds=time.monotonic()-started)


def tests(repo, subset=None):
    targets = [p.name for p in sorted(repo.glob('test_*.py'))]
    if subset is not None:
        if subset.split('::')[0] not in targets or any(c.isspace() for c in subset):
            raise ValueError('subset must name a visible test file or a test node inside it')
        targets = [subset]
    if not targets:
        raise ValueError('no test files')
    return command([sys.executable, '-m', 'pytest', '-q', '-p', 'no:cacheprovider', *targets], repo)


def validate_args(name, args):
    if not isinstance(args, dict):
        return False
    signature = next((t['function']['parameters'] for t in TOOLS if t['function']['name'] == name), None)
    if signature is None or set(args) - set(signature['properties']) or set(signature['required']) - set(args):
        return False
    for key, value in args.items():
        if key == 'range':
            if value is not None and not (isinstance(value, list) and len(value) == 2 and
                    all(type(n) is int for n in value) and 1 <= value[0] <= value[1]):
                return False
        elif not isinstance(value, str):
            return False
    return True


def parse_turn(raw):
    # Qwen's template starts an open thinking block; a complete response closes it.
    content = raw.rsplit('</think>', 1)[-1] if '</think>' in raw else raw
    attempted = bool(re.search(r'<\s*/?\s*tool_call\b|"(?:name|arguments|tool_calls|function_call)"\s*:|\b(?:list_files|search_code|read_file|apply_patch|run_tests)\s*\(', content))
    blocks = re.findall(r'<tool_call>\s*(.*?)\s*</tool_call>', content, flags=re.S)
    calls, error = [], None
    if attempted:
        try:
            if not blocks or content.count('<tool_call>') != len(blocks) or content.count('</tool_call>') != len(blocks):
                raise ValueError('missing, malformed, or unmatched native tool_call tags')
            for block in blocks:
                obj = json.loads(block)
                if not isinstance(obj, dict) or set(obj) != {'name','arguments'} or not validate_args(obj.get('name'), obj.get('arguments')):
                    raise ValueError('unknown tool or incorrect argument structure')
                calls.append(obj)
        except (ValueError, TypeError) as exc:
            error = str(exc)
    return dict(attempted=attempted, valid=attempted and error is None, calls=calls, error=error)


def resolve(repo, value):
    path = (repo / value).resolve()
    if not path.is_relative_to(repo.resolve()):
        raise ValueError('path must stay within the repository')
    return path


def execute(repo, module, name, args):
    try:
        if name == 'list_files':
            directory = resolve(repo, args['path'])
            if not directory.is_dir():
                raise ValueError('path must be a directory')
            return {'files':[str(p.relative_to(repo)) for p in sorted(directory.rglob('*')) if p.is_file() and '__pycache__' not in p.parts]}
        if name == 'search_code':
            pattern = re.compile(args['pattern'])
            return {'matches':[f'{p.name}:{i}: {line}' for p in sorted(repo.glob('*.py'))
                               for i, line in enumerate(p.read_text().splitlines(),1) if pattern.search(line)]}
        if name == 'read_file':
            lines = resolve(repo, args['path']).read_text().splitlines()
            start, end = args['range'] or [1,len(lines)]
            return {'content':'\n'.join(f'{i}: {line}' for i,line in enumerate(lines,1) if start <= i <= end)}
        if name == 'apply_patch':
            patch = args['diff']
            # Whitelist both sides; disallow rename/delete/create/binary patches.
            headers = re.findall(r'^(?:---|\+\+\+) ([^\t\n]+)', patch, flags=re.M)
            if headers != [f'a/{module}',f'b/{module}'] or re.search(r'^(?:rename |copy |GIT binary|Binary files|new file|deleted file)', patch, flags=re.M):
                raise ValueError('diff must modify only the existing implementation file using a/ and b/ headers')
            check = command(['git','apply','--check','--whitespace=nowarn','-'], repo, input_text=patch)
            if check['returncode']:
                return check
            return command(['git','apply','--whitespace=nowarn','-'], repo, input_text=patch)
        if name == 'run_tests':
            return tests(repo, args.get('subset'))
        raise ValueError('unknown tool')
    except (ValueError, OSError, re.error) as exc:
        return {'error':str(exc)}


def validate_tasks():
    results=[]
    for task in json.loads((ROOT/'task_manifest.json').read_text()):
        with tempfile.TemporaryDirectory(prefix='s3-validate-') as directory:
            repo=Path(directory)
            shutil.copytree(ROOT/'toy_repos'/task['task_id'],repo,dirs_exist_ok=True)
            before=tests(repo)
            applied=execute(repo,task['module'],'apply_patch',{'diff':(ROOT/'solutions'/f"{Path(task['module']).stem}.patch").read_text()})
            after=tests(repo)
            exact_revert=(repo/task['module']).read_bytes() == (ROOT/'solutions'/task['module']).read_bytes()
            result=dict(task_id=task['task_id'],buggy=before,revert=applied,correct=after,exact_revert=exact_revert,
                        valid=before['returncode']==1 and after['returncode']==0 and exact_revert,
                        buggy_sha256=digest(ROOT/'toy_repos'/task['task_id']/task['module']),
                        test_sha256=digest(ROOT/'toy_repos'/task['task_id']/f"test_{task['module']}"))
            results.append(result)
            print(task['task_id'],result['valid'],flush=True)
    dump(ROOT/'task_validation.json',results)
    assert len(results)>=10 and all(r['valid'] for r in results)


def aggregate(records):
    tasks=[]
    for task_id in sorted({r['task_id'] for r in records}):
        group=[r for r in records if r['task_id']==task_id]
        k=sum(r['reward'] for r in group)
        attempted=sum(r['attempt_turns'] for r in group)
        valid=sum(r['valid_turns'] for r in group)
        tasks.append(dict(task_id=task_id,rollouts=len(group),successes=k,
                          contract_pass_at_1=int(k>0),per_rollout_success=k/len(group),mixed=int(0<k<len(group)),
                          valid_calls=valid,total_call_attempts=attempted,
                          format_validity=valid/attempted if attempted else None))
    attempts=sum(t['total_call_attempts'] for t in tasks)
    valid=sum(t['valid_calls'] for t in tasks)
    successes=sum(t['successes'] for t in tasks)
    solved=sum(t['contract_pass_at_1'] for t in tasks)
    mixed=sum(t['mixed'] for t in tasks)
    rate=valid/attempts if attempts else None
    if rate is not None and rate>.85 and solved>0 and mixed>0:
        decision='skip SFT'
    elif rate is not None and rate<.85 and solved>0:
        decision='format-only SFT'
    elif rate is not None and rate>.85 and solved==0:
        decision='reduce difficulty'
    else:
        decision='full trajectory SFT'
    return dict(valid_calls=valid,total_call_attempts=attempts,format_validity=rate,
                tasks=len(tasks),rollouts=len(records),solved_tasks=solved,contract_pass_at_1=solved/len(tasks),
                per_rollout_success=successes/len(records),successful_rollouts=successes,
                mixed_tasks=mixed,mixed_group_ratio=mixed/len(tasks),gate_decision=decision,per_task=tasks)


def selftest():
    valid='<tool_call>{"name":"read_file","arguments":{"path":"x.py","range":null}}</tool_call>'
    assert parse_turn(valid)['valid']
    for raw in ['<tool_call>{bad}</tool_call>', '<tool_call>{"name":"oops","arguments":{}}</tool_call>',
                '<tool_call>{"name":"read_file","arguments":{"path":"x.py"}}</tool_call>',
                'read_file("x.py")', '<tool_call>{"name":"run_tests","arguments":{}}']:
        parsed=parse_turn(raw)
        assert parsed['attempted'] and not parsed['valid'], raw
    assert not parse_turn('I will inspect the source next.')['attempted']
    assert not parse_turn('<think>I could use read_file("x.py")</think>All done.')['attempted']
    assert parse_turn(valid+valid)['valid']
    assert not validate_args('read_file',{'path':'x','range':[True,3]})
    # Known-answer synthetic controls: deterministic all-zero, all-one, and mixed groups.
    records=[dict(task_id=str(i//8),reward=0,attempt_turns=10,valid_turns=9) for i in range(80)]
    zero=aggregate(records)
    assert zero['contract_pass_at_1']==zero['mixed_group_ratio']==0 and zero['format_validity']==.9
    assert zero['gate_decision']=='reduce difficulty'
    for i,record in enumerate(records):
        record['reward']=int(i%8==0)
    mixed=aggregate(records)
    assert mixed['contract_pass_at_1']==mixed['mixed_group_ratio']==1 and mixed['per_rollout_success']==.125
    assert mixed['gate_decision']=='skip SFT'
    for record in records:
        record['valid_turns']=8
    assert aggregate(records)['gate_decision']=='format-only SFT'
    for record in records:
        record['reward']=1
        record['valid_turns']=9
    assert aggregate(records)['mixed_group_ratio']==0 and aggregate(records)['gate_decision']=='full trajectory SFT'
    with tempfile.TemporaryDirectory() as tmp:
        repo=Path(tmp)
        (repo/'x.py').write_text('x = 1\n')
        assert 'error' in execute(repo,'x.py','read_file',{'path':'../outside','range':None})
        assert 'error' in execute(repo,'x.py','apply_patch',{'diff':'--- a/test_x.py\n+++ b/test_x.py\n'})
    dump(ROOT/'selftest.json',dict(passed=True,seed=BASE_SEED,controls='Exact known-answer zero, all-one, mixed groups; parser, gate branches, path/test protections. No Monte Carlo error.'))
    print('selftest passed')


def run(args):
    # Heavy imports occur only in the inference command.
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
    import torch
    output=ROOT/args.output
    output.mkdir(parents=True,exist_ok=False)
    started=time.time()
    manifest=json.loads((ROOT/'task_manifest.json').read_text())
    validation=json.loads((ROOT/'task_validation.json').read_text())
    assert len(manifest)>=10 and all(t['valid'] for t in validation)
    metadata=dict(started_unix=started,model=MODEL,temperature=1.0,top_p=1.0,top_k=-1,
        group_size=8,max_steps=12,max_model_len=16384,max_tokens_per_turn=2048,
        seed=BASE_SEED,seed_rule='trajectory=20260905 + task_index*100 + rollout_index; turn=trajectory_seed*100 + zero_based_turn',
        thinking='native template default (enabled)',batch_size=args.batch_size,inference_only=True,
        format_counting_convention=CONVENTION,
        metric_naming='Contract pass@1 is fraction of task groups with K>0, conventionally pass@8. Also report successes/rollouts.',
        versions={p:importlib.metadata.version(p) for p in ['torch','vllm','transformers','pytest']},
        model_config_sha256=digest(Path(MODEL)/'config.json'),
        script_sha256=digest(Path(__file__)),manifest_sha256=digest(ROOT/'task_manifest.json'),
        hourly_rate_cny=args.hourly_rate_cny,rate_is_estimate=True,budget_cny=16,
        gpu=command(['nvidia-smi'],ROOT),command_line=sys.argv)
    dump(output/'run_metadata.json',metadata)
    tokenizer=AutoTokenizer.from_pretrained(MODEL,local_files_only=True)
    (output/'chat_template.jinja').write_text(tokenizer.chat_template)
    dump(output/'tools.json',TOOLS)
    (output/'system_prompt.txt').write_text(SYSTEM+'\n')
    llm=LLM(model=MODEL,tokenizer=MODEL,dtype='bfloat16',seed=BASE_SEED,
        max_model_len=16384,max_num_seqs=args.batch_size,gpu_memory_utilization=.9,
        enforce_eager=True,enable_prefix_caching=False)
    metadata['model_load_seconds']=time.time()-started
    dump(output/'run_metadata.json',metadata)
    specs=[(i,task,r) for i,task in enumerate(manifest) for r in range(8)]
    all_records=[]
    # A hard watchdog in the launcher additionally limits the whole tmux process.
    for offset in range(0,len(specs),args.batch_size):
        if time.time()-started > args.max_hours*3600:
            raise RuntimeError('budget wall-time limit reached before completion')
        states=[]
        for index,task,rollout in specs[offset:offset+args.batch_size]:
            temporary=tempfile.TemporaryDirectory(prefix='s3-rollout-')
            repo=Path(temporary.name)
            shutil.copytree(ROOT/'toy_repos'/task['task_id'],repo,dirs_exist_ok=True)
            ident=f"{task['task_id']}_r{rollout}"
            states.append(dict(id=ident,task=task,rollout=rollout,seed=BASE_SEED+index*100+rollout,
                temporary=temporary,repo=repo,messages=[{'role':'system','content':SYSTEM},{'role':'user','content':task['issue']}],
                turns=[],done=False,started=time.time(),stop_reason='step_limit'))
        try:
            for step in range(12):
                active=[s for s in states if not s['done']]
                if not active:
                    break
                prompts=[]
                runnable=[]
                for state in active:
                    prompt=tokenizer.apply_chat_template(state['messages'],tools=TOOLS,tokenize=False,add_generation_prompt=True)
                    prompt_ids=tokenizer.encode(prompt,add_special_tokens=False)
                    if len(prompt_ids)>16384-2048:
                        state['done']=True
                        state['stop_reason']='context_limit'
                        continue
                    prompts.append({'prompt_token_ids':prompt_ids})
                    runnable.append(state)
                    append(output/'prompts.jsonl',dict(trajectory_id=state['id'],step=step+1,prompt=prompt,prompt_tokens=len(prompt_ids)))
                if not runnable:
                    continue
                before=time.monotonic()
                params=[SamplingParams(temperature=1.0,top_p=1.0,top_k=-1,max_tokens=2048,seed=s['seed']*100+step) for s in runnable]
                generated=llm.generate(prompts,params,use_tqdm=False)
                batch_seconds=time.monotonic()-before
                def handle(pair):
                    state,response=pair
                    completion=response.outputs[0]
                    raw=completion.text
                    parsed=parse_turn(raw)
                    state['messages'].append({'role':'assistant','content':raw})
                    observations=[]
                    if parsed['valid']:
                        for call in parsed['calls']:
                            result=execute(state['repo'],state['task']['module'],call['name'],call['arguments'])
                            observations.append(dict(call=call,result=result))
                            state['messages'].append({'role':'tool','content':json.dumps(result,ensure_ascii=False)})
                    elif parsed['attempted']:
                        state['messages'].append({'role':'tool','content':json.dumps({'error':parsed['error'],'hint':'Use native <tool_call> JSON with the documented argument schema.'})})
                    elif completion.finish_reason=='length':
                        state['messages'].append({'role':'user','content':'Your turn reached the output limit. Please make a concise tool call or finish.'})
                    else:
                        state['done']=True
                        state['stop_reason']='final_response'
                    row=dict(trajectory_id=state['id'],task_id=state['task']['task_id'],step=step+1,
                        seed=state['seed']*100+step,raw_output=raw,finish_reason=completion.finish_reason,
                        generated_tokens=len(completion.token_ids),prompt_tokens=len(response.prompt_token_ids),
                        generation_batch_wall_seconds=batch_seconds,parse=parsed,tool_observations=observations)
                    state['turns'].append(row)
                    return row
                with ThreadPoolExecutor(max_workers=args.batch_size) as pool:
                    rows=list(pool.map(handle,zip(runnable,generated)))
                for row in rows:
                    append(output/'turns.jsonl',row)
                print(json.dumps(dict(batch=offset//args.batch_size,step=step+1,active=len(runnable),generation_seconds=round(batch_seconds,2))),flush=True)
            def finish(state):
                # Test bytes must match fixtures. Restore original test files before final grading.
                fixture=ROOT/'toy_repos'/state['task']['task_id']
                intact=all((state['repo']/p.name).exists() and digest(state['repo']/p.name)==digest(p) for p in fixture.glob('test_*.py'))
                for p in fixture.glob('test_*.py'):
                    shutil.copy2(p,state['repo']/p.name)
                final=tests(state['repo'])
                source=(state['repo']/state['task']['module']).read_text()
                initial=(fixture/state['task']['module']).read_text()
                row=dict(trajectory_id=state['id'],task_id=state['task']['task_id'],rollout_index=state['rollout'],
                    seed=state['seed'],temperature=1.0,reward=int(final['returncode']==0 and intact),tests_intact=intact,
                    final_tests=final,stop_reason=state['stop_reason'],steps=len(state['turns']),
                    attempt_turns=sum(t['parse']['attempted'] for t in state['turns']),
                    valid_turns=sum(t['parse']['valid'] for t in state['turns']),
                    valid_call_objects=sum(len(t['parse']['calls']) for t in state['turns'] if t['parse']['valid']),
                    wall_seconds=time.time()-state['started'],
                    generated_tokens=sum(t['generated_tokens'] for t in state['turns']),
                    final_source=source,final_patch=''.join(difflib.unified_diff(initial.splitlines(True),source.splitlines(True),fromfile=state['task']['module'],tofile=state['task']['module'])))
                return row
            with ThreadPoolExecutor(max_workers=args.batch_size) as pool:
                records=list(pool.map(finish,states))
            for row in records:
                append(output/'trajectories.jsonl',row)
                all_records.append(row)
                print(json.dumps({k:row[k] for k in ['trajectory_id','reward','steps','stop_reason']}),flush=True)
        finally:
            for state in states:
                state['temporary'].cleanup()
    elapsed=time.time()-started
    metadata.update(finished_unix=time.time(),wall_seconds=elapsed,gpu_reservation_hours=elapsed/3600,
                    estimated_cost_cny=elapsed/3600*args.hourly_rate_cny,
                    torch_peak_allocated_gb=torch.cuda.max_memory_allocated()/1e9,
                    end_gpu=command(['nvidia-smi'],ROOT),status='complete')
    dump(output/'run_metadata.json',metadata)
    summary=aggregate(all_records)
    assert len(all_records)==len(manifest)*8 and all(t['rollouts']==8 for t in summary['per_task'])
    dump(output/'metrics.json',summary)
    print(json.dumps(summary),flush=True)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('action',choices=['validate','selftest','run','summarize'])
    parser.add_argument('--output',default='run_20260905')
    parser.add_argument('--batch-size',type=int,default=8)
    parser.add_argument('--hourly-rate-cny',type=float,default=2.18)
    parser.add_argument('--max-hours',type=float,default=5.0)
    args=parser.parse_args()
    if args.action=='validate': validate_tasks()
    elif args.action=='selftest': selftest()
    elif args.action=='run': run(args)
    else:
        records=[json.loads(line) for line in (ROOT/args.output/'trajectories.jsonl').read_text().splitlines()]
        dump(ROOT/args.output/'metrics.json',aggregate(records))

if __name__=='__main__': main()
