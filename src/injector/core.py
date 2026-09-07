"""Seeded, formatting-preserving repair-task synthesis.

Rollback candidates must name an actual git fix; feature candidates must supply a
reviewable implementation change. Neither source silently falls back to mutation.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from pathlib import Path
import itertools
import random
import subprocess
import textwrap
from typing import Literal

import libcst as cst
from libcst.metadata import MetadataWrapper, PositionProvider

BUG_TYPES = ('condition-inversion', 'off-by-one', 'variable-misuse', 'wrong-return-value',
             'missing-exception-handling', 'api-parameter-error', 'boundary-condition-omission')
SOURCES = ('mutation', 'commit-rollback', 'feat-add')
SPANS = ('single-function', 'single-file', 'cross-file')

@dataclass(frozen=True)
class Difficulty:
    source: str
    type: str
    count: int = 1
    hint: str = 'L0'
    span: str = 'single-function'

    def __post_init__(self):
        if self.source not in SOURCES or self.type not in BUG_TYPES:
            raise ValueError('Unknown source or bug type')
        if self.count not in (1, 2, 3) or self.hint not in ('L0', 'L1', 'L2') or self.span not in SPANS:
            raise ValueError('Invalid difficulty dimension')
        if self.span != 'single-function' and self.count < 2:
            raise ValueError('Multiple locations require count >= 2')

@dataclass(frozen=True)
class HistoricalChange:
    type: str
    path: str
    fix_commit: str
    before: str
    after: str

@dataclass(frozen=True)
class FeatureChange:
    type: str
    path: str
    before: str
    after: str
    description: str

@dataclass(frozen=True)
class Edit:
    path: str
    function: str
    line: int
    before: str
    after: str
    provenance: str

@dataclass
class InjectionResult:
    difficulty: Difficulty
    seed: int
    originals: dict[str, str]
    modified: dict[str, str]
    edits: list[Edit]
    issue: str

    def apply(self, repo: Path) -> None:
        self._write(repo, self.originals, self.modified)

    def revert(self, repo: Path) -> None:
        self._write(repo, self.modified, self.originals)

    @staticmethod
    def _write(repo: Path, expected: dict[str, str], desired: dict[str, str]) -> None:
        for path, text in expected.items():
            if _path(repo, path).read_text() != text:
                raise ValueError(f'Workspace changed: {path}')
        for path, text in desired.items():
            _path(repo, path).write_text(text)

    def to_dict(self) -> dict:
        return asdict(self)


def _path(repo: Path, path: str) -> Path:
    p = (repo / path).resolve()
    if not p.is_relative_to(repo.resolve()) or p.is_symlink():
        raise ValueError('Path escapes repository')
    if 'tests' in Path(path).parts or Path(path).name.startswith('test_'):
        raise ValueError('Cannot inject tests')
    return p


class _Candidates(cst.CSTVisitor):
    METADATA_DEPENDENCIES = (PositionProvider,)
    def __init__(self, module: cst.Module, path: str, kind: str):
        self.module, self.path, self.kind = module, path, kind
        self.functions: list[str] = []
        self.params: list[list[str]] = []
        self.items: list[tuple[cst.CSTNode, cst.CSTNode, Edit]] = []

    def visit_FunctionDef(self, node: cst.FunctionDef):
        self.functions.append(node.name.value)
        self.params.append([p.name.value for p in (*node.params.posonly_params, *node.params.params, *node.params.kwonly_params)])

    def leave_FunctionDef(self, node: cst.FunctionDef):
        self.functions.pop()
        self.params.pop()

    def add(self, old: cst.CSTNode, new: cst.CSTNode):
        if self.functions:
            self.items.append((old, new, Edit(self.path, '.'.join(self.functions),
                self.get_metadata(PositionProvider, old).start.line,
                self.module.code_for_node(old), self.module.code_for_node(new), 'libcst-operator')))

    def visit_If(self, node: cst.If):
        if self.kind == 'condition-inversion':
            self.add(node.test, cst.UnaryOperation(cst.Not(), node.test.with_changes(lpar=(cst.LeftParen(),), rpar=(cst.RightParen(),))))
        elif self.kind == 'boundary-condition-omission':
            self.add(node.test, cst.Name('False'))

    def visit_Call(self, node: cst.Call):
        if self.kind == 'off-by-one' and isinstance(node.func, cst.Name) and node.func.value == 'range' and node.args:
            i = 0 if len(node.args) == 1 else 1
            old = node.args[i].value
            self.add(old, cst.BinaryOperation(old, cst.Add(), cst.Integer('1')))
        if self.kind == 'api-parameter-error':
            for arg in node.args:
                if arg.keyword and isinstance(arg.value, cst.Name) and arg.value.value in ('True', 'False'):
                    self.add(arg.value, cst.Name('False' if arg.value.value == 'True' else 'True'))
                elif arg.keyword and isinstance(arg.value, cst.Integer):
                    self.add(arg.value, cst.Integer(str(int(arg.value.evaluated_value) + 1)))

    def visit_Return(self, node: cst.Return):
        if self.kind == 'wrong-return-value' and node.value:
            self.add(node.value, cst.Name('None'))
        elif self.kind == 'variable-misuse' and isinstance(node.value, cst.Name) and self.params:
            for name in self.params[-1]:
                if name != node.value.value:
                    self.add(node.value, cst.Name(name))

    def visit_Try(self, node: cst.Try):
        if self.kind == 'missing-exception-handling' and node.handlers:
            # Preserve the try body and comments, but stop catching ordinary errors.
            handler = node.handlers[0]
            self.add(handler, handler.with_changes(type=cst.Name('SystemExit')))


class _Replace(cst.CSTTransformer):
    def __init__(self, replacements: dict[int, cst.CSTNode]):
        self.replacements = replacements
    def on_leave(self, original_node, updated_node):
        return self.replacements.get(id(original_node), updated_node)


def _fits(edits: list[Edit], span: str) -> bool:
    files = {e.path for e in edits}
    funcs = {(e.path, e.function) for e in edits}
    if span == 'single-function':
        return len(funcs) == 1
    if span == 'single-file':
        return len(files) == 1 and len(funcs) > 1
    return len(files) > 1


def _issue(d: Difficulty, edits: list[Edit]) -> str:
    if d.hint == 'L0':
        return 'Repair ' + d.type + ' defects at ' + ', '.join(f'{e.path}:{e.line} ({e.function})' for e in edits) + '.'
    if d.hint == 'L1':
        return 'Existing behavior regressed in ' + ', '.join(sorted({e.path for e in edits})) + '. Inspect boundary cases and restore the documented behavior.'
    return 'Some supported inputs now produce incorrect results or unexpected exceptions. Restore existing behavior while preserving supported features.'


def inject(repo: Path, difficulty: Difficulty, seed: int, *,
           history: list[HistoricalChange] = (), features: list[FeatureChange] = ()) -> InjectionResult:
    if type(seed) is not int:
        raise TypeError('An explicit integer seed is required')
    repo = Path(repo)
    rng = random.Random(seed)
    modules = {}
    candidates = []
    paths = sorted(p for p in repo.rglob('*.py') if not any(x.startswith('.') or x in ('tests', '__pycache__') for x in p.relative_to(repo).parts) and not p.name.startswith('test_'))
    for path in paths:
        rel = path.relative_to(repo).as_posix()
        wrapper = MetadataWrapper(cst.parse_module(path.read_text()))
        modules[rel] = wrapper.module
        finder = _Candidates(wrapper.module, rel, difficulty.type)
        wrapper.visit(finder)
        if difficulty.source == 'mutation':
            candidates.extend(finder.items)
    if difficulty.source != 'mutation':
        entries = history if difficulty.source == 'commit-rollback' else features
        for entry in entries:
            if entry.type != difficulty.type:
                continue
            path = _path(repo, entry.path)
            module = modules[entry.path]
            if difficulty.source == 'commit-rollback':
                assert isinstance(entry, HistoricalChange)
                def git(ref):
                    return subprocess.check_output(['git', '-C', str(repo), 'show', f'{ref}:{entry.path}'], text=True)
                if entry.before not in git(entry.fix_commit + '^') or entry.after not in git(entry.fix_commit):
                    raise ValueError('Rollback fragments do not match actual fix history')
                before, after, provenance = entry.after, entry.before, entry.fix_commit
            else:
                assert isinstance(entry, FeatureChange)
                if not entry.description.strip() or entry.before == entry.after:
                    raise ValueError('Feature change requires a description and changed implementation')
                before, after, provenance = entry.before, entry.after, 'feature: ' + entry.description
            # Match entire functions/statements through the CST; never textual rewrite.
            class Match(cst.CSTVisitor):
                METADATA_DEPENDENCIES = (PositionProvider,)
                def __init__(self):
                    self.functions = []
                def visit_FunctionDef(self, node):
                    self.functions.append(node.name.value)
                def leave_FunctionDef(self, node):
                    self.functions.pop()
                def on_visit(self, node):
                    super().on_visit(node)
                    if not self.functions:
                        return True
                    if not isinstance(node, (cst.BaseExpression, cst.BaseSmallStatement, cst.BaseStatement)):
                        return True
                    if module.code_for_node(node).strip() == textwrap.dedent(before).strip():
                        dedented_after = textwrap.dedent(after)
                        try:
                            if isinstance(node, cst.BaseExpression):
                                replacement = cst.parse_expression(dedented_after)
                            elif isinstance(node, cst.BaseSmallStatement):
                                replacement = cst.parse_statement(dedented_after).body[0]
                            elif isinstance(node, cst.BaseStatement):
                                replacement = cst.parse_statement(dedented_after)
                            else:
                                return True
                        except cst.ParserSyntaxError:
                            return True
                        candidates.append((node, replacement, Edit(entry.path, '.'.join(self.functions),
                            self.get_metadata(PositionProvider, node).start.line, before, after, provenance)))
                        return False
                    return True
            MetadataWrapper(module, unsafe_skip_copy=True).visit(Match())
    rng.shuffle(candidates)
    selected = None
    for group in itertools.combinations(candidates, difficulty.count):
        edits = [x[2] for x in group]
        if _fits(edits, difficulty.span) and len({(e.path, e.line) for e in edits}) == len(edits):
            # Avoid overlapping nested transformations, which would silently lose edits.
            if any(a[2].before in b[2].before for a, b in itertools.permutations(group, 2) if a[2].path == b[2].path and a[2].function == b[2].function):
                continue
            selected = group
            break
    if selected is None:
        raise ValueError(f'No eligible {difficulty.count}-edit {difficulty.span} task for {difficulty.source}/{difficulty.type}')
    originals, modified = {}, {}
    for path in sorted({x[2].path for x in selected}):
        originals[path] = _path(repo, path).read_text()
        modified[path] = modules[path].visit(_Replace({id(old): new for old, new, edit in selected if edit.path == path})).code
        compile(modified[path], path, 'exec')
    edits = [x[2] for x in selected]
    return InjectionResult(difficulty, seed, originals, modified, edits, _issue(difficulty, edits))
