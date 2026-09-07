"""Small real repair instance used by admission and full-harness attack tests."""
from pathlib import Path
from src.environment.harness import TaskInstance
from src.injector.core import Difficulty, Edit

CLEAN = 'def target(x, y):\n    return x + y\n\ndef other_func(x):\n    return x * 2\n'
BUGGY = CLEAN.replace('x + y', 'x - y')
SUITE = '''from subject import target, other_func
def test_a_sanity():
    assert 1 + 1 == 2
def test_b_sanity():
    assert other_func(4) == 8
def test_c_target():
    assert target(2, 4) == 6
def test_d_target_boundary():
    assert target(-2, 4) == 2
'''


def instance(root: Path) -> TaskInstance:
    tree = root / 'tree'
    (tree / 'tests').mkdir(parents=True)
    (tree / 'subject.py').write_text(BUGGY)
    (tree / 'tests/test_visible.py').write_text(SUITE)
    hidden = root / 'hidden'
    hidden.mkdir()
    (hidden / 'test_hidden.py').write_text(SUITE)
    return TaskInstance('toy:11', 'toy', 'toy@sha256:' + '1'*64,
        Difficulty('mutation', 'wrong-return-value'), 11, tree, hidden, 'Repair addition',
        {'subject.py': CLEAN}, {'subject.py': BUGGY},
        [Edit('subject.py', 'target', 2, 'x + y', 'x - y', 'fixture')],
        {'subject.py': ['tests/test_visible.py']})
