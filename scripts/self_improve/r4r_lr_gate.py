"""Fail closed on incomplete metrics, scheduler construction evidence, or LR drift."""
import argparse
import json
import math
import re
from pathlib import Path

WORKSPACE = Path(__file__).resolve().parents[2]
try:
    from scripts.self_improve.r4r_gates import runtime_input
except ModuleNotFoundError:
    from r4r_gates import runtime_input
REFERENCE = runtime_input(WORKSPACE, 'artifacts/self_improve/r5/raw/octorl_r3c/seed_137/metrics_target_100.jsonl')


def read_lrs(path):
    rows = [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
    result = {}
    for row in rows:
        step = int(row['step'])
        if step in result:
            raise ValueError(f'duplicate step {step}')
        lr = float(row['data']['actor/lr'])
        if not math.isfinite(lr):
            raise ValueError(f'nonfinite LR at {step}')
        result[step] = lr
    return result


def gate(metrics, log, start, end, reference=REFERENCE):
    ref = read_lrs(reference)
    if set(ref) != set(range(1, 81)):
        raise ValueError('reference must contain all 80 steps')
    if not 1 <= start <= end <= 80:
        raise ValueError('invalid LR gate interval')
    actual = read_lrs(metrics)
    if set(actual) != set(range(start, end + 1)):
        raise ValueError('missing or unexpected LR steps')
    markers = re.findall(r'R4R_LR_SCHEDULER total_steps=(\d+) warmup=(\d+) type=(\w+)', Path(log).read_text())
    if not markers or any(marker != ('100', '5', 'cosine') for marker in markers):
        raise ValueError('missing or incorrect R4R_LR_SCHEDULER line')
    for step, lr in actual.items():
        if abs(lr - ref[step]) > 1e-12:
            raise ValueError(f'LR mismatch at U{step}: {lr} != {ref[step]}')
    return {'steps': len(actual), 'max_abs_error': max(abs(lr-ref[s]) for s, lr in actual.items())}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('metrics', 'log'):
        p.add_argument('--'+name, type=Path, required=True)
    p.add_argument('--start', type=int, required=True)
    p.add_argument('--end', type=int, required=True)
    p.add_argument('--reference', type=Path, default=REFERENCE)
    a = p.parse_args()
    print(json.dumps(gate(a.metrics, a.log, a.start, a.end, a.reference)))


if __name__ == '__main__':
    main()
