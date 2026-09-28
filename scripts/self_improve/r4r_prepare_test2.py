"""Generate and seal test2 using the unchanged r2.0 instance generator."""
import json
from collections import Counter
from pathlib import Path
import pandas as pd
from src.tasks.tool_recovery.generator import generate_instance, FAMILY_BY_NAME, PROTOCOL_VERSION
try:
    from scripts.self_improve.r3_prepare import _row
except ModuleNotFoundError:
    from r3_prepare import _row
try:
    from scripts.self_improve.r4r_gates import sha256, prereg_gate
except ModuleNotFoundError:
    from r4r_gates import sha256, prereg_gate

COUNTS = {'constraint_violation': 60, 'missing_dependency': 52, 'stale_version': 48, 'normal': 40}
FAMILIES = ['deployment_service', 'notification_service']


def generate(workspace):
    root = workspace/'artifacts/self_improve/r4r'
    prereg_gate(workspace, root)
    target = root/'data'
    if (target/'test2_seal.json').exists():
        raise RuntimeError('test2 is already sealed; refusing to regenerate')
    prior = {split: json.loads((workspace/f'artifacts/self_improve/r3/data/{split}_manifest.json').read_text()) for split in ('train', 'dev', 'test')}
    assert PROTOCOL_VERSION == prior['test']['generator_version']
    used = {i['fingerprint'] for m in prior.values() for i in m['instances']}
    instances, rejected, duplicates = [], [], []
    seed = 300000
    for fault, count in COUNTS.items():
        accepted = 0
        # Bounded search; never silently reduce the target counts.
        while accepted < count and seed < 400000:
            instance = generate_instance(seed, [FAMILY_BY_NAME[n] for n in FAMILIES], fault)
            seed += 1
            if instance is None:
                rejected.append(seed-1)
            elif instance.fingerprint in used:
                duplicates.append(seed-1)
            else:
                used.add(instance.fingerprint)
                instances.append(instance.to_dict())
                accepted += 1
        if accepted != count:
            raise RuntimeError(f'cannot reach {fault}={count}; generated {accepted}')
    manifest = dict(split='test2', requested_count=200, generated_count=200, start_seed=300000,
                    generator_version=PROTOCOL_VERSION, instances=instances,
                    rejected_seeds=rejected, duplicate_seeds=duplicates)
    target.mkdir(parents=True, exist_ok=True)
    mp, pp = target/'test2_manifest.json', target/'test2.parquet'
    mp.write_text(json.dumps(manifest, indent=2, sort_keys=True)+'\n')
    pd.DataFrame([_row(i, n) for n, i in enumerate(instances)]).to_parquet(pp, index=False)
    fingerprints = {i['fingerprint'] for i in instances}
    seal = dict(manifest_sha256=sha256(mp), parquet_sha256=sha256(pp), generator_version=PROTOCOL_VERSION,
                start_seed=300000, counts=dict(Counter(i['fault_type'] for i in instances)), families=FAMILIES,
                unique_fingerprints=len(fingerprints), model_evaluated=False,
                dedup_evidence={s: dict(manifest_sha256=sha256(workspace/f'artifacts/self_improve/r3/data/{s}_manifest.json'),
                                      overlap_count=len(fingerprints & {i['fingerprint'] for i in m['instances']})) for s, m in prior.items()})
    (target/'test2_seal.json').write_text(json.dumps(seal, indent=2, sort_keys=True)+'\n')
    return seal


if __name__ == '__main__':
    print(json.dumps(generate(Path(__file__).resolve().parents[2]), indent=2))
