"""R4 estimands for arbitrary instance counts; preregistered R4r power design."""
import json
from pathlib import Path
import numpy as np
try:
    from scripts.self_improve import r4_analysis as r4
except ModuleNotFoundError:
    import r4_analysis as r4
try:
    from scripts.self_improve.r4_analysis import read_eval, calibrate_coupling, clustered_ci
except ModuleNotFoundError:
    from r4_analysis import read_eval, calibrate_coupling, clustered_ci


def analyze(models, *, seed=20260928, resamples=10000):
    """Primary R4r inference; arbitrary n, frozen instance bootstrap defaults."""
    return r4.analyze(models, seed=seed, resamples=resamples)


def synthetic_power(base_dev, *, couplings, n=160, shifts=(0., .03, .05, .10),
                    reps=1000, resamples=10000, seed=20260928):
    source = r4._base_fault_rates(base_dev)
    if n <= 0:
        raise ValueError('n must be positive')
    rates = (source if n == len(source) else
             np.random.Generator(np.random.PCG64(seed)).choice(source, size=n, replace=True))
    result = {'reps': reps, 'resamples': resamples, 'seed': seed, 'conditions': {}}
    # Exactly the same instance-cluster percentile bootstrap as clustered_ci;
    # reuse its deterministic index matrix across replications.
    indices = np.random.Generator(np.random.PCG64(seed)).integers(0, n, size=(resamples, n))
    for ci, (name, q) in enumerate({'independent': 1.0, **couplings}.items()):
        if not 0 <= q <= 1:
            raise ValueError('q must be in [0,1]')
        table = {}
        for shift in shifts:
            rng = np.random.Generator(np.random.PCG64(seed + 100*ci + int(shift*100)))
            covered = detected = 0
            widths = []
            for _ in range(reps):
                shared = rng.random((n, 4))
                independent = rng.random((n, 4)) < q
                a = np.where(independent, rng.random((n, 4)), shared) < rates[:, None]
                b = np.where(independent, rng.random((n, 4)), shared) < np.clip(rates+shift, 0, 1)[:, None]
                d = (b.astype(int)-a.astype(int)).mean(axis=1)
                low, high = np.quantile(d[indices].mean(axis=1), (.025, .975))
                covered += low <= 0 <= high
                detected += low > 0
                widths.append(high-low)
            table[f'{shift:.2f}'] = {'coverage_of_zero': float(covered/reps),
                'detection_rate': float(detected/reps), 'mean_ci_width': float(np.mean(widths))}
        result['conditions'][name] = {'q': q, 'shifts': table}
        result.setdefault('mde_at_80pct_power', {})[name] = next(
            (float(s) for s in table if float(s) > 0 and table[s]['detection_rate'] >= .8),
            '>0.10 within tested shifts')
    return result


def calibrations(workspace):
    root = workspace/'artifacts/self_improve/r3c/results'
    base = read_eval(root/'base_dev_eval.json')
    pairs = {'coupled_q_hi': ('seed_42/u20_dev.json', 'seed_137/u20_dev.json'),
             'coupled_q_lo': ('seed_42/u50_dev.json', 'seed_42/u80_dev.json')}
    return base, {name: calibrate_coupling(base, read_eval(root/a), read_eval(root/b)) for name, (a,b) in pairs.items()}


def regression(workspace):
    root = workspace/'artifacts/self_improve/r4'
    models = {name: read_eval(root/'test_eval'/f'{name}.json') for name in
              ['base']+[f'{arm}_{s}' for arm in ('fixed','failure_driven') for s in (42,137,2718)]}
    result = analyze(models, seed=20260927)
    base, cal = calibrations(workspace)
    result['synthetic_power'] = synthetic_power(base, couplings={k:v['q'] for k,v in cal.items()},
                                               n=40, shifts=(0.,.05,.10), seed=20260927)
    result['synthetic_power']['calibrations'] = cal
    expected = json.loads((root/'r4_analysis.json').read_text())
    if result != expected:
        raise RuntimeError('R4 analysis regression: deep equality failed; stop')
    return {'deep_equal_all_fields': True, 'n': 40, 'seed': 20260927}


def main():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', action='append', metavar='NAME=FILE')
    parser.add_argument('--out-json', type=Path)
    parser.add_argument('--out-md', type=Path)
    parser.add_argument('--secondary', action='store_true', help='Original test: seed 20260927')
    args = parser.parse_args()
    workspace = Path(__file__).resolve().parents[2]
    out = workspace/'artifacts/self_improve/r4r'
    if args.model:
        if not args.out_json or not args.out_md:
            parser.error('--model requires --out-json and --out-md')
        models = {}
        for item in args.model:
            name, separator, path = item.partition('=')
            if not separator or name in models:
                parser.error('unique NAME=FILE model mappings required')
            models[name] = read_eval(path)
        result = analyze(models, seed=20260927 if args.secondary else 20260928)
        args.out_json.write_text(json.dumps(result, indent=2, sort_keys=True)+'\n')
        args.out_md.write_text(r4.markdown_table(result))
        return
    if args.secondary or args.out_json or args.out_md:
        parser.error('analysis output options require --model')
    control = regression(workspace)
    (out/'analysis_regression.json').write_text(json.dumps(control, indent=2)+'\n')
    base, cal = calibrations(workspace)
    result = synthetic_power(base, couplings={k:v['q'] for k,v in cal.items()})
    result.update(n=160, calibrations=cal, source='base dev per-instance fault rates',
                  instance_rate_sampling='replacement, PCG64(20260928)',
                  aa_ci=list(clustered_ci(np.zeros(160), seed=20260928)))
    # Inherited R4 synthetic control: the independent Bernoulli null must cover 0 in [0.93, 0.97].
    # Coupled nulls share uniforms, so most d_i are exactly 0 and the percentile CI over-covers by
    # construction (R4 n=40 q_lo: 0.996); for them only the Type I rate at shift 0 is an acceptance check.
    # Decision record: artifacts/self_improve/r4r/prelaunch_decisions.md (D1).
    result['null_coverage_pass'] = {k: (.93 <= v['shifts']['0.00']['coverage_of_zero'] <= .97 if k == 'independent'
                                        else v['shifts']['0.00']['detection_rate'] <= .05)
                                    for k,v in result['conditions'].items()}
    result['null_acceptance_rule'] = ('independent: coverage of 0 in [0.93, 0.97]; '
                                      'coupled: coverage reported, Type I (shift 0 detection) <= 0.05')
    (out/'prelaunch_power.json').write_text(json.dumps(result, indent=2, sort_keys=True)+'\n')
    lines = ['# R4r pre-launch power', '', '160 instances; 4 rollouts; 1000 replications; 10000 instance resamples; seed 20260928.',
             'MDE is the first tested shift reaching 80% detection; shifts are clipped at probability 1.', '',
             '| Condition | q | Null coverage | +0 detection | +.03 | +.05 | +.10 | MDE |', '|---|---:|---:|---:|---:|---:|---:|---|']
    for k,v in result['conditions'].items():
        t=v['shifts']
        lines.append(f"| {k} | {v['q']:.9f} | {t['0.00']['coverage_of_zero']:.3f} | "+' | '.join(f"{t[s]['detection_rate']:.3f}" for s in ('0.00','0.03','0.05','0.10'))+f" | {result['mde_at_80pct_power'][k]} |")
    lines += ['', 'A/A CI: [0, 0]. R4 n=40 regression: all fields deep-equal.', '',
              'Null acceptance ('+result['null_acceptance_rule']+'): '+json.dumps(result['null_coverage_pass']),
              'See prelaunch_decisions.md D1.', r4.CAVEAT]
    (out/'prelaunch_power.md').write_text('\n'.join(lines)+'\n')
    print('\n'.join(lines), flush=True)
    if not all(result['null_coverage_pass'].values()):
        raise RuntimeError('Frozen simulation fails null coverage acceptance; report without tuning seeds or model')


if __name__ == '__main__':
    main()
