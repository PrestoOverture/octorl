"""R4r preregistration, selection, and shared warm-up identity gates."""
from collections import Counter
import hashlib
import json
from pathlib import Path
import pandas as pd

EXPECTED = '46d844d862eb55048c3be346ecf4ebc56f8f1133596e54bd4221891a256f78a4'
PREREG = 'artifacts/self_improve/contracts/r4r_lr_fixed_rerun_prereg.yaml'
RECORDS = 'artifacts/self_improve/r4/remote_records'


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(8*1024*1024), b''):
            h.update(chunk)
    return h.hexdigest()


def prereg_gate(workspace, root):
    actual = sha256(runtime_input(workspace, PREREG))
    if actual != EXPECTED:
        raise ValueError('prereg hash differs from frozen value')
    record = Path(root) / 'prereg_sha256.txt'
    if record.exists():
        if record.read_text() != EXPECTED+'\n':
            raise ValueError('prereg launch hash changed')
    else:
        record.parent.mkdir(parents=True, exist_ok=True)
        record.write_text(EXPECTED+'\n')


def selection_gate(stage_dir, reference):
    stage_dir, reference = Path(stage_dir), Path(reference)
    if (stage_dir/'selection.json').read_bytes() != (reference/'selection.json').read_bytes():
        raise ValueError('selection.json bytes differ')
    # Compare the task-order byte sequence, not parquet container metadata.
    expected = json.loads((reference/'selection.json').read_text())['tasks']
    actual = [row['task_id'] for row in pd.read_parquet(stage_dir/'train.parquet')['extra_info']]
    if json.dumps(actual, ensure_ascii=True).encode() != json.dumps(expected, ensure_ascii=True).encode():
        raise ValueError('train.parquet task order differs')
    if (reference/'train.parquet').exists():
        old = [row['task_id'] for row in pd.read_parquet(reference/'train.parquet')['extra_info']]
        if old != actual:
            raise ValueError('R4 parquet task order differs')


def warmup_gate(actual, reference):
    def counts(path):
        rows = [json.loads(line) for line in Path(path).read_text().splitlines()]
        if len(rows) != 320 or {r['global_step'] for r in rows} != set(range(1, 21)):
            raise ValueError('warm-up must contain U1-U20 and 320 rollouts')
        return Counter((r['global_step'], r['task_id']) for r in rows)
    if counts(actual) != counts(reference):
        raise ValueError('warm-up per-step task multiset differs')


def runtime_input(workspace, relative):
    """Use the original local evidence or the byte-verified deployment bundle."""
    source = Path(workspace)/relative
    if source.exists():
        return source
    bundled = Path(workspace)/'scripts/self_improve/r4r_inputs'/relative
    if not bundled.exists():
        raise ValueError(f'missing frozen runtime input: {relative}')
    return bundled


def preflight_gate(root):
    path = Path(root)/'preflight_pass.json'
    if not path.is_file():
        raise ValueError('GPU preflight has not passed')
    record = json.loads(path.read_text())
    if record.get('prereg_sha256') != EXPECTED or record.get('resume_steps') != [21,22] or record.get('warmup_steps') != [1,2] or record.get('optimizer_load') is not True or record.get('lr_scheduler_load') is not True or record.get('lr_gates') != ['PASS','PASS']:
        raise ValueError('GPU preflight evidence is incomplete')
