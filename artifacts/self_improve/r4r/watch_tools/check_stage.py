"""Claude Code's own per-stage check: python3 check_stage.py SEED ARM STAGE  (or: 2718 warmup 0)."""
import csv, json, math, subprocess, sys
from pathlib import Path

WS = Path('/Users/kwang/projects/octorl')
SCR = Path(sys.argv[4]) if len(sys.argv) > 4 else Path(__file__).parent / 'stages'  # pass a scratch dir to keep pulls out of the repo
seed, arm, stage = sys.argv[1], sys.argv[2], int(sys.argv[3])
R = '/root/autodl-tmp/octorl_r4r'
if arm == 'warmup':
    rdir, start, end = f'{R}/seed_2718/warmup', 0, 20
else:
    start = 20 + 10 * (stage - 1); end = start + 10
    rdir = f'{R}/seed_{seed}/{arm}/stage_{stage}'
out = SCR / f'{seed}_{arm}_{stage}'; out.mkdir(parents=True, exist_ok=True)
# fetch every attempt's metrics/timing/exit + the attributed records
subprocess.run(['rsync', '-rt', '--include=*/', '--include=metrics_target_*.jsonl', '--include=timing.json',
                '--include=exit.txt', '--include=INFRA_FAILED', '--include=attributed.jsonl', '--include=selection.json',
                '--exclude=*', '-e', 'ssh -o BatchMode=yes', f'autodl-r4:{rdir}/', str(out) + '/'], check=True)
ref = {json.loads(l)['step']: json.loads(l)['data']['actor/lr'] for l in
       open(WS / 'artifacts/self_improve/r5/raw/octorl_r3c/seed_137/metrics_target_100.jsonl')}
attempts = sorted(p for p in out.glob('attempt_*')) if arm != 'warmup' else [out]
ok = [a for a in attempts if (a / f'metrics_target_{end}.jsonl').exists() and (a / f'metrics_target_{end}.jsonl').stat().st_size]
a = ok[-1]
rows = {json.loads(l)['step']: json.loads(l)['data'] for l in open(a / f'metrics_target_{end}.jsonl')}
assert sorted(rows) == list(range(start + 1, end + 1)), sorted(rows)
g = lambda k: [float(rows[s].get(k, float('nan'))) for s in sorted(rows)]
dlr = max(abs(rows[s]['actor/lr'] - ref[s]) for s in rows)
reward, mixed = g('r3b/reward_mean'), g('r3b/mixed_group_fraction')
res = dict(branch=f'{seed}_{arm}', stage=stage, updates=f'U{start+1}-U{end}', attempt=a.name,
           attempts=[p.name for p in attempts], infra_failed=[p.name for p in attempts if (p / 'INFRA_FAILED').exists()],
           max_abs_dlr=dlr, lr_first=rows[start + 1]['actor/lr'], lr_last=rows[end]['actor/lr'],
           grad_max=max(g('actor/grad_norm')), kl_mean=sum(g('actor/kl_loss')) / len(rows),
           entropy_mean=sum(g('actor/entropy')) / len(rows), reward_mean=sum(reward) / len(rows),
           eff_groups=sum(mixed) / len(rows),
           finite=all(math.isfinite(v) for k in ('actor/grad_norm', 'actor/kl_loss', 'actor/pg_loss') for v in g(k)))
t = json.loads((a / 'timing.json').read_text()) if (a / 'timing.json').exists() else {}
res['cost_cny'] = round(t.get('cost_cny', float('nan')), 3); res['exit'] = t.get('exit_code')
# context: R4 same stage and R3c same updates (seed 137 / 42 only)
if arm != 'warmup':
    r4csv = WS / f'artifacts/self_improve/r5/diagnostics/r4_{arm}_{seed}.csv'
    if r4csv.exists():
        rr = [r for r in csv.DictReader(open(r4csv)) if start < int(float(r['global_step'])) <= end]
        res['r4_same_stage'] = dict(lr_sum=sum(float(r['lr']) for r in rr), reward=sum(float(r['reward_mean']) for r in rr) / len(rr),
                                    kl=sum(float(r['kl']) for r in rr) / len(rr))
    r3 = WS / f'artifacts/self_improve/r5/diagnostics/r3c_{seed}.csv'
    if r3.exists():
        rr = [r for r in csv.DictReader(open(r3)) if start < int(float(r['global_step'])) <= end]
        res['r3c_same_updates'] = dict(reward=sum(float(r['reward_mean']) for r in rr) / len(rr), kl=sum(float(r['kl']) for r in rr) / len(rr))
    res['lr_sum'] = sum(rows[s]['actor/lr'] for s in rows)
print(json.dumps(res, indent=1))
