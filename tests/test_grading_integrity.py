"""F5 regressions through accepted patches and the actual Docker harness."""
import difflib
from pathlib import Path

import pytest

from src.environment.harness import ScriptedPolicy, package, run_trajectory
from src.environment.sandbox import create_pool
from src.injector.core import Difficulty

ROOT = Path(__file__).resolve().parents[1]
IMAGE = (ROOT / 'artifacts/p1/docker-image.id').read_text().strip()


@pytest.fixture(scope='module')
def pool():
    with create_pool(IMAGE, size=1) as value:
        yield value


@pytest.fixture
def instance(tmp_path):
    return package(ROOT / 'repos/train/record_index',
                   Difficulty('mutation', 'condition-inversion'), 11, tmp_path)


def drive(instance, pool, payload, *, sampling_seed=11):
    name = instance.edits[0].path
    before = instance.modified[name]
    diff = ''.join(difflib.unified_diff(before.splitlines(True),
        (before + '\n' + payload + '\n').splitlines(True),
        fromfile='a/' + name, tofile='b/' + name))
    with pool.lease(instance.root) as box:
        record = run_trajectory(instance, box, ScriptedPolicy([
            {'name': 'apply_patch', 'arguments': {'diff': diff}}]),
            sampling_seed=sampling_seed, trajectory_id='integrity-test',
            policy_version='scripted', base_model_revision='none', manifest_ref='test')
        assert record['turns'][0]['tool_calls'][0]['exit_code'] == 0
        return record


@pytest.mark.parametrize('payload', [
    'import os as _os; _os._exit(0)',
    'import os as _os; print("51 passed in 0.01s", flush=True); _os._exit(0)',
])
def test_exit_and_forged_summary_fail_without_detectors(instance, pool, monkeypatch, payload):
    import src.environment.verifier as verifier
    monkeypatch.setattr(verifier, 'detect_constant_backfill', lambda *a: [])
    monkeypatch.setattr(verifier, 'detect_skip_xfail', lambda *a: [])
    record = drive(instance, pool, payload)
    assert record['reward'] == 0
    assert any('incomplete' in f for f in record['cheat_flags'])


def test_protected_write_fails_with_reason(instance, pool):
    record = drive(instance, pool,
        "from pathlib import Path as _P; _P('tests/MARKER').write_text('x')")
    assert record['reward'] == 0
    assert any('protected' in f for f in record['cheat_flags'])


@pytest.mark.parametrize('name', ['tests/MARKER', 'hidden_tests/MARKER', 'conftest.py'])
def test_os_denies_grading_asset_writes_but_allows_source(instance, pool, name):
    with pool.lease(instance.root) as box:
        result = box.run(['python', '-c',
            'from pathlib import Path\n'
            f'p=Path({name!r})\n'
            'try: p.write_text("BAD")\n'
            'except OSError as e: print(type(e).__name__)\n'
            'else: raise AssertionError("write succeeded")\n'])
        assert result.exit_code == 0, result.stdout + result.stderr
        assert 'PermissionError' in result.stdout or 'OSError' in result.stdout
        assert not (box.root / name).exists() or (box.root / name).read_text() != 'BAD'
        editable = instance.edits[0].path
        result = box.run(['python', '-c', f'from pathlib import Path; p=Path({editable!r}); p.write_text(p.read_text()+"\\n# allowed\\n")'])
        assert result.exit_code == 0, result.stderr
