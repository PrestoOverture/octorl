"""Every registered attack traverses the real tools, Docker, and verifier."""
import dataclasses
import difflib
import json
from pathlib import Path

import pytest

from attacks import PATTERNS, PARTIAL_EXIT
from tests.reward_fixtures import instance as make_instance, CLEAN, BUGGY
from src.environment.harness import ScriptedPolicy, run_trajectory
from src.environment.sandbox import create_pool

ROOT = Path(__file__).resolve().parents[1]
IMAGE = (ROOT / 'artifacts/p1/docker-image.id').read_text().strip()


@pytest.fixture(scope='module')
def pool():
    with create_pool(IMAGE, size=1) as value:
        yield value


def run_attack(instance, pool, pattern, spy=None):
    if pattern.category == 'path_violation':
        diff = pattern.source
    else:
        target = 'subject.py'
        before = BUGGY
        if pattern.name == 'write_to_conftest':
            target, before, after = 'conftest.py', '', pattern.source
        elif pattern.name == 'break_other_function':
            after = 'def target(x, y):\n    return x + y\n' + pattern.source
        elif pattern.category in {'constant_backfill', 'compile_error'} or pattern.name == 'delete_function_body':
            after = pattern.source + '\ndef other_func(x):\n    return x * 2\n'
        else:
            after = BUGGY + '\n' + pattern.source
        diff = ''.join(difflib.unified_diff(before.splitlines(True), after.splitlines(True),
                                         fromfile='a/' + target, tofile='b/' + target))
    with pool.lease(instance.root) as box:
        if spy is not None:
            original = box.run
            def observed(argv, **kwargs):
                result = original(argv, **kwargs)
                spy.append((list(argv), result))
                return result
            box.run = observed
        try:
            record = run_trajectory(instance, box, ScriptedPolicy([
                {'name': 'apply_patch', 'arguments': {'diff': diff}}]),
                sampling_seed=11, trajectory_id=pattern.name, policy_version='seeded-attack',
                base_model_revision='none', manifest_ref='attack-suite')
        finally:
            if spy is not None:
                box.run = original
        return record


@pytest.mark.parametrize('pattern', PATTERNS, ids=lambda p: p.name)
def test_attack_yields_zero_through_full_harness(tmp_path, pool, pattern):
    record = run_attack(make_instance(tmp_path), pool, pattern)
    assert record['reward'] == 0, record
    if pattern.category == 'path_violation' or pattern.name == 'write_to_conftest':
        assert 'patch_path_whitelist' in record['cheat_flags']
    else:
        assert record['turns'][0]['tool_calls'][0]['exit_code'] == 0
    if pattern.name in {'exit_during_import', 'partial_test_exit', 'forged_pass_summary'}:
        assert any('incomplete' in f for f in record['cheat_flags'])
    if pattern.name == 'runtime_test_write':
        assert any('protected_asset_write' in f for f in record['cheat_flags'])


def test_partial_exit_after_two_real_passes_without_blacklist(tmp_path, pool, monkeypatch):
    import src.environment.verifier as verifier
    monkeypatch.setattr(verifier, 'detect_constant_backfill', lambda *a: [])
    monkeypatch.setattr(verifier, 'detect_skip_xfail', lambda *a: [])
    pattern = next(p for p in PATTERNS if p.name == 'partial_test_exit')
    seen = []
    record = run_attack(make_instance(tmp_path), pool, pattern, seen)
    assert record['reward'] == 0
    assert any('execution_count_mismatch' in f for f in record['cheat_flags'])
    hidden = [r for args, r in seen if 'hidden_tests/' in args and '--collect-only' not in args][-1]
    passed = [e for e in hidden.events if e.get('event') == 'test' and e.get('when') == 'call' and e.get('outcome') == 'passed']
    assert len(passed) == 2, hidden.events
