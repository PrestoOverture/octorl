"""Seed 20260906: real pytest failure/pass round trips for all source/type pairs."""
import json
import subprocess
import sys
import shutil
from pathlib import Path

import pytest
from src.injector import BUG_TYPES, SOURCES, BUCKETS, Difficulty, FeatureChange, HistoricalChange, inject
from src.injector.catalog import CatalogError, validate_catalog, inject_from_catalog

CASES = {
    'condition-inversion': ('def target(x, y):\n    if x > 0:\n        return x\n    return y\n', 'target(2, 4) == 2'),
    'off-by-one': ('def target(x, y):\n    return list(range(x))\n', 'target(2, 4) == [0, 1]'),
    'variable-misuse': ('def target(x, y):\n    return x\n', 'target(2, 4) == 2'),
    'wrong-return-value': ('def target(x, y):\n    return x + y\n', 'target(2, 4) == 6'),
    'missing-exception-handling': ('def target(x, y):\n    try:\n        return int(x)\n    except ValueError:\n        return y\n', "target('bad', 4) == 4"),
    'api-parameter-error': ('def target(x, y):\n    return sorted(x, reverse=True)\n', 'target([1, 3], 0) == [3, 1]'),
    'boundary-condition-omission': ('def target(x, y):\n    if not x:\n        return y\n    return x[0]\n', 'target([], 4) == 4'),
}

def git(repo, *args):
    return subprocess.check_output(['git', '-C', str(repo), *args], text=True).strip()

def run(repo):
    shutil.rmtree(repo / "__pycache__", ignore_errors=True)
    return subprocess.run([sys.executable, '-m', 'pytest', '-q', '-p', 'no:cacheprovider'], cwd=repo,
        capture_output=True, text=True).returncode

@pytest.mark.parametrize('kind', BUG_TYPES)
@pytest.mark.parametrize('source', SOURCES)
def test_roundtrip(tmp_path, kind, source):
    clean, check = CASES[kind]
    (tmp_path/'code.py').write_text(clean)
    (tmp_path/'test_visible.py').write_text('from code import target\ndef test_behavior():\n    assert '+check+'\n')
    # Avoid Python's already-loaded stdlib code module in subprocess pytest.
    (tmp_path/'code.py').rename(tmp_path/'subject.py')
    (tmp_path/'test_visible.py').write_text('from subject import target\ndef test_behavior():\n    assert '+check+'\n')
    mutation = inject(tmp_path, Difficulty('mutation', kind), 20260906)
    buggy = mutation.modified['subject.py']
    history, features = [], []
    if source == 'commit-rollback':
        git(tmp_path, 'init', '-q')
        git(tmp_path, 'config', 'user.email', 'fixture@example.invalid')
        git(tmp_path, 'config', 'user.name', 'Injector fixture')
        (tmp_path/'subject.py').write_text(buggy)
        git(tmp_path, 'add', '.')
        git(tmp_path, 'commit', '-qm', 'Initial implementation')
        (tmp_path/'subject.py').write_text(clean)
        git(tmp_path, 'add', '.')
        git(tmp_path, 'commit', '-qm', 'Fix existing behavior')
        history = [HistoricalChange(kind, 'subject.py', git(tmp_path, 'rev-parse', 'HEAD'), buggy, clean)]
    elif source == 'feat-add':
        # A new opt-in describe feature is introduced; old behavior regresses in
        # the same implementation. The new feature has its own passing test.
        added = buggy.replace('def target(x, y):\n', "def target(x, y, *, describe=False):\n    if describe:\n        return 'target operation'\n")
        features = [FeatureChange(kind, 'subject.py', clean, added, 'Add describe keyword to report operation name')]
    result = inject(tmp_path, Difficulty(source, kind), 20260906, history=history, features=features)
    assert result.to_dict() == inject(tmp_path, Difficulty(source, kind), 20260906, history=history, features=features).to_dict()
    assert run(tmp_path) == 0
    result.apply(tmp_path)
    assert run(tmp_path) == 1
    if source == 'feat-add':
        namespace = {}
        exec(result.modified['subject.py'], namespace)
        assert namespace['target'](None, None, describe=True) == 'target operation'
    result.revert(tmp_path)
    assert run(tmp_path) == 0

@pytest.mark.parametrize('count,span', [(1,'single-function'),(2,'single-function'),(3,'single-function'),(2,'single-file'),(3,'single-file'),(2,'cross-file')])
def test_count_span_and_hints(tmp_path, count, span):
    text = '# preserved header\ndef first(x):\n    # retain this comment\n    if x == 1:\n        return 1\n    if x == 2:\n        return 2\n    if x == 3:\n        return 3\n    return 0\n'
    text += text[text.index('def first'):].replace('first', 'second')
    text += text[text.index('def first'):text.index('def second')].replace('first', 'third')
    (tmp_path/'a.py').write_text(text)
    (tmp_path/'b.py').write_text(text)
    issues = []
    for hint in ('L0','L1','L2'):
        result = inject(tmp_path, Difficulty('mutation','condition-inversion',count,hint,span),20260906)
        assert len(result.edits) == count
        assert all('# preserved header' in value and '# retain this comment' in value for value in result.modified.values())
        files = {e.path for e in result.edits}
        funcs = {(e.path,e.function) for e in result.edits}
        assert (len(funcs) == 1) if span == 'single-function' else (len(files) == 1 and len(funcs)>1) if span == 'single-file' else len(files)>1
        issues.append(result.issue)
    assert len(set(issues)) == 3


def test_buckets_and_seed():
    assert len(BUCKETS) == 18
    assert len({b.id for b in BUCKETS}) == 18
    assert {BUCKETS[0].sample(s).source for s in range(100)} == set(SOURCES)
    assert {BUCKETS[0].sample(s).type for s in range(100)} == set(BUG_TYPES)
    assert BUCKETS[0].sample(20260906) == BUCKETS[0].sample(20260906)

@pytest.mark.parametrize('source', ['commit-rollback', 'feat-add'])
@pytest.mark.parametrize('count,span', [(2,'single-function'), (3,'single-function'), (2,'single-file'), (3,'single-file'), (2,'cross-file')])
def test_provenance_sources_count_span(tmp_path, source, count, span):
    clean = 'def first(x):\n    if x == 1:\n        return 1\n    if x == 2:\n        return 2\n    if x == 3:\n        return 3\n    return 0\n\ndef second(x):\n    if x == 4:\n        return 4\n    return 0\n\ndef third(x):\n    if x == 5:\n        return 5\n    return 0\n'
    for name in ('a.py','b.py'):
        (tmp_path/name).write_text(clean)
    git(tmp_path,'init','-q')
    git(tmp_path,'config','user.name','History fixture')
    git(tmp_path,'config','user.email','fixture@example.invalid')
    for name in ('a.py','b.py'):
        (tmp_path/name).write_text(clean.replace('x ==', 'x !='))
    git(tmp_path,'add','.')
    git(tmp_path,'commit','-qm','Initial guards')
    for name in ('a.py','b.py'):
        (tmp_path/name).write_text(clean)
    git(tmp_path,'add','.')
    git(tmp_path,'commit','-qm','Correct guards')
    commit = git(tmp_path,'rev-parse','HEAD')
    history, features = [], []
    for name in ('a.py','b.py'):
        for i in range(1,6):
            history.append(HistoricalChange('condition-inversion',name,commit,f'x != {i}',f'x == {i}'))
            # Add a new accepted sentinel, while accidentally inverting equality.
            features.append(FeatureChange('condition-inversion',name,f'x == {i}',f'x != {i} or x == {100+i}',f'Support sentinel {100+i}'))
    result = inject(tmp_path,Difficulty(source,'condition-inversion',count,'L0',span),20260906,history=history,features=features)
    assert len(result.edits) == count
    result.apply(tmp_path)
    assert any((tmp_path/p).read_text() != original for p,original in result.originals.items())
    result.revert(tmp_path)
    assert all((tmp_path/p).read_text() == clean for p in result.originals)


def test_annotation_nodes_do_not_crash():
    """Visitor must skip Annotation nodes without CSTCodegenError."""
    import libcst as cst
    from libcst.metadata import MetadataWrapper, PositionProvider
    source = 'def target(x: int, y: str = "hi") -> bool:\n    if x > 0:\n        return True\n    return False\n'
    module = cst.parse_module(source)
    wrapper = MetadataWrapper(module)
    class Probe(cst.CSTVisitor):
        METADATA_DEPENDENCIES = (PositionProvider,)
        def __init__(self):
            self.visited = 0
        def on_visit(self, node):
            super().on_visit(node)
            if isinstance(node, (cst.BaseExpression, cst.BaseSmallStatement, cst.BaseStatement)):
                module.code_for_node(node)
            self.visited += 1
            return True
    probe = Probe()
    wrapper.visit(probe)
    assert probe.visited > 10


def test_indented_catalog_match(tmp_path):
    """History fragments with class-method indent match after dedent."""
    clean = 'class Svc:\n    def method(self, x: int) -> bool:\n        return x > 0\n'
    buggy = 'class Svc:\n    def method(self, x: int) -> bool:\n        return x >= 0\n'
    (tmp_path / 'svc.py').write_text(clean)
    git(tmp_path, 'init', '-q')
    git(tmp_path, 'config', 'user.name', 'Fixture')
    git(tmp_path, 'config', 'user.email', 'f@example.invalid')
    (tmp_path / 'svc.py').write_text(buggy)
    git(tmp_path, 'add', '.')
    git(tmp_path, 'commit', '-qm', 'Introduce bug')
    (tmp_path / 'svc.py').write_text(clean)
    git(tmp_path, 'add', '.')
    git(tmp_path, 'commit', '-qm', 'Fix bug')
    commit = git(tmp_path, 'rev-parse', 'HEAD')
    history = [HistoricalChange('boundary-condition-omission', 'svc.py', commit,
        '    def method(self, x: int) -> bool:\n        return x >= 0\n',
        '    def method(self, x: int) -> bool:\n        return x > 0\n')]
    result = inject(tmp_path, Difficulty('commit-rollback', 'boundary-condition-omission'), 20260906, history=history)
    assert len(result.edits) == 1
    result.apply(tmp_path)
    assert 'x >= 0' in (tmp_path / 'svc.py').read_text()
    result.revert(tmp_path)
    assert 'x > 0' in (tmp_path / 'svc.py').read_text()


def test_catalog_validation_empty_fragment():
    errors = validate_catalog({'history': [{'type': 'off-by-one', 'path': 'a.py', 'fix_commit': 'abc',
                                            'before': '', 'after': 'x = 1'}]})
    assert any('empty before' in e for e in errors)


def test_catalog_validation_unparseable():
    errors = validate_catalog({'history': [{'type': 'off-by-one', 'path': 'a.py', 'fix_commit': 'abc',
                                            'before': 'x = 1\ny = 2', 'after': 'x = 1'}]})
    assert any('not a parseable' in e for e in errors)


def test_catalog_validation_missing_fix_commit():
    errors = validate_catalog({'history': [{'type': 'off-by-one', 'path': 'a.py', 'fix_commit': '',
                                            'before': 'x = 1', 'after': 'x = 2'}]})
    assert any('missing fix_commit' in e for e in errors)


def test_catalog_validation_feature_identical():
    errors = validate_catalog({'features': [{'type': 'off-by-one', 'path': 'a.py',
                                             'before': 'x = 1', 'after': 'x = 1', 'description': 'test'}]})
    assert any('identical' in e for e in errors)


def test_catalog_validation_feature_missing_description():
    errors = validate_catalog({'features': [{'type': 'off-by-one', 'path': 'a.py',
                                             'before': 'x = 1', 'after': 'x = 2', 'description': ''}]})
    assert any('missing description' in e for e in errors)


def test_catalog_error_raised_for_malformed_type(tmp_path):
    (tmp_path / 'a.py').write_text('def target(x, y):\n    return x + y\n')
    catalog = tmp_path / 'injector_sources.json'
    catalog.write_text(json.dumps({'history': [{'type': 'off-by-one', 'path': 'a.py', 'fix_commit': 'abc',
                                                'before': 'x = 1\ny = 2', 'after': 'x = 3'}]}))
    with pytest.raises(CatalogError, match='malformed'):
        inject_from_catalog(tmp_path, Difficulty('commit-rollback', 'off-by-one'), 1, catalog=catalog)


def test_catalog_valid_entries_pass():
    data = {'history': [{'type': 'condition-inversion', 'path': 'a.py', 'fix_commit': 'abc123',
                         'before': 'x > 0', 'after': 'x < 0'}],
            'features': [{'type': 'variable-misuse', 'path': 'a.py',
                          'before': 'return x', 'after': 'return y', 'description': 'swap vars'}]}
    assert validate_catalog(data) == []
