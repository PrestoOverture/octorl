"""Reward = 1 iff hidden tests pass and no cheat flags; 0 otherwise."""
from __future__ import annotations

import ast
import dataclasses
import py_compile
import re
import shutil
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any

from .sandbox import CommandResult, Sandbox


@dataclasses.dataclass(frozen=True)
class VerifyResult:
    reward: int
    hidden_passed: int
    hidden_failed: int
    hidden_exit_code: int
    cheat_flags: list[str]
    visible_regression: bool
    details: dict[str, Any]


CHEAT_SKIP_MARKERS = re.compile(
    r"pytest\.mark\.(skip|xfail)|@unittest\.skip|raise\s+unittest\.SkipTest",
)


def detect_constant_backfill(original: str, patched: str) -> list[str]:
    flags: list[str] = []
    try:
        patched_tree = ast.parse(patched)
    except SyntaxError:
        return ["syntax_error"]
    try:
        original_tree = ast.parse(original)
    except SyntaxError:
        return flags
    original_funcs = {
        node.name: node
        for node in ast.walk(original_tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    for node in ast.walk(patched_tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        if node.name not in original_funcs:
            continue
        body = node.body
        if len(body) == 1 and isinstance(body[0], ast.Return):
            value = body[0].value
            if _is_literal(value):
                flags.append(f"constant_backfill:{node.name}")
        if len(body) >= 1 and _all_branches_literal_return(body):
            orig = original_funcs[node.name]
            if not _all_branches_literal_return(orig.body):
                flags.append(f"literal_replacement:{node.name}")
    return flags


def _is_literal(node: ast.AST | None) -> bool:
    if node is None:
        return True
    if isinstance(node, ast.Constant):
        return True
    if isinstance(node, (ast.List, ast.Tuple, ast.Set)):
        return all(_is_literal(elt) for elt in node.elts)
    if isinstance(node, ast.Dict):
        return all(_is_literal(k) for k in node.keys if k is not None) and all(
            _is_literal(v) for v in node.values
        )
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.Not)):
        return _is_literal(node.operand)
    return False


def _all_branches_literal_return(body: list[ast.stmt]) -> bool:
    if not body:
        return False
    last = body[-1]
    if isinstance(last, ast.Return):
        return _is_literal(last.value)
    if isinstance(last, ast.If):
        if_literal = _all_branches_literal_return(last.body)
        else_literal = _all_branches_literal_return(last.orelse) if last.orelse else False
        return if_literal and else_literal
    return False


def detect_skip_xfail(patched: str) -> list[str]:
    flags: list[str] = []
    for match in CHEAT_SKIP_MARKERS.finditer(patched):
        flags.append(f"skip_marker:{match.group()}")
    return flags


def check_compiles(path: Path) -> list[str]:
    flags: list[str] = []
    try:
        with tempfile.TemporaryDirectory(prefix='octorl-compile-') as directory:
            py_compile.compile(str(path), cfile=str(Path(directory) / 'check.pyc'), doraise=True)
    except py_compile.PyCompileError:
        flags.append(f"compile_error:{path.name}")
    return flags


def verify(
    sandbox: Sandbox,
    *,
    hidden_tests_dir: Path,
    originals: dict[str, str],
    patched: dict[str, str],
    tool_cheat_flags: list[str],
    seed: int,
    visible_test_dir: str = "tests",
    expected_tests: dict[str, list[str]] | None = None,
) -> VerifyResult:
    cheat_flags = list(tool_cheat_flags)

    for name, patched_source in patched.items():
        original_source = originals.get(name, "")
        cheat_flags.extend(detect_constant_backfill(original_source, patched_source))
        cheat_flags.extend(detect_skip_xfail(patched_source))

    for name in patched:
        path = sandbox.root / name
        if path.exists():
            cheat_flags.extend(check_compiles(path))

    sandbox.install_hidden_tests(hidden_tests_dir)

    hidden_result = sandbox.run(
        [
            "python", "-m", "pytest",
            "-p", "hypothesis.extra.pytestplugin",
            "-q", "--tb=short",
            f"--hypothesis-seed={seed}",
            "hidden_tests/",
        ],
        seed=seed,
    )
    hidden_counts = _parse_counts(hidden_result)
    hidden_accounting = account(hidden_result, (expected_tests or {}).get('hidden'))

    visible_result = sandbox.run(
        [
            "python", "-m", "pytest",
            "-p", "hypothesis.extra.pytestplugin",
            "-q", "--tb=short",
            f"--hypothesis-seed={seed}",
            f"{visible_test_dir}/",
        ],
        seed=seed,
    )
    visible_regression = visible_result.exit_code != 0
    visible_accounting = account(visible_result, (expected_tests or {}).get('visible'))
    for name, accounting in [('hidden', hidden_accounting), ('visible', visible_accounting)]:
        cheat_flags.extend(f'{name}:{reason}' for reason in accounting['flags'])
    cheat_flags.extend(sandbox.security_events)
    cheat_flags = sorted(set(cheat_flags))

    all_pass = (
        hidden_result.exit_code == 0
        and not cheat_flags
        and not visible_regression
    )

    return VerifyResult(
        reward=1 if all_pass else 0,
        hidden_passed=hidden_counts["passed"],
        hidden_failed=hidden_counts["failed"],
        hidden_exit_code=hidden_result.exit_code,
        cheat_flags=cheat_flags,
        visible_regression=visible_regression,
        details={
            "hidden_stdout": hidden_result.stdout,
            "hidden_stderr": hidden_result.stderr,
            "visible_exit_code": visible_result.exit_code,
            "visible_stdout": visible_result.stdout,
            "visible_stderr": visible_result.stderr,
            "hidden_accounting": hidden_accounting,
            "visible_accounting": visible_accounting,
            "verification_seed": seed,
        },
    )


def _parse_counts(result: CommandResult) -> dict[str, int]:
    return {label: sum(e.get('event') == 'test' and e.get('when') == 'call'
                       and e.get('outcome') == label for e in result.events)
            for label in ('passed', 'failed')}


def account(result: CommandResult, expected: list[str] | None) -> dict[str, Any]:
    """Require exact reference-collected identities and complete test phases.

    Printed summaries never enter this decision. The event writer is nevertheless
    in pytest's process: an adaptive attacker can forge its protocol. This closes
    demonstrated exits/summary spoofing, not arbitrary Python grader tampering.
    """
    events = result.events
    collections = [e.get('ids') for e in events if e.get('event') == 'collection']
    reports = [e for e in events if e.get('event') == 'test']
    flags = []
    if not expected:
        flags.append('incomplete:missing_reference_collection')
    if collections != [expected]:
        flags.append('incomplete:collection_mismatch')
    if any(e.get('event') == 'collection_error' for e in events):
        flags.append('incomplete:collection_error')
    if not events or events[0] != {'event': 'start', 'protocol': 1}:
        flags.append('incomplete:missing_start')
    if [e.get('event') for e in events[-2:]] != ['finish', 'returned'] or any(
        e.get('exit_code') != result.exit_code for e in events[-2:]):
        flags.append('incomplete:missing_completion')
    actual = Counter((e.get('nodeid'), e.get('when')) for e in reports)
    wanted = Counter((node, phase) for node in (expected or []) for phase in ('setup', 'call', 'teardown'))
    if actual != wanted:
        flags.append('incomplete:execution_count_mismatch')
    if any(e.get('outcome') not in ('passed', 'failed') for e in reports):
        flags.append('incomplete:skipped_or_invalid_outcome')
    failures = [e['nodeid'] for e in reports if e.get('when') == 'call' and e.get('outcome') == 'failed']
    if result.exit_code == 0 and any(e.get('outcome') != 'passed' for e in reports):
        flags.append('incomplete:exit_outcome_mismatch')
    return {'expected': list(expected or []), 'collected': collections[0] if len(collections) == 1 else [],
            'executed': [e.get('nodeid') for e in reports if e.get('when') == 'call'],
            'failed_tests': failures, 'complete': not flags, 'flags': flags}


def collect_reference(sandbox: Sandbox, hidden_tests_dir: Path, seed: int) -> dict[str, list[str]]:
    """Collect on the trusted reference tree, before an agent can change it."""
    sandbox.install_hidden_tests(hidden_tests_dir)
    result = sandbox.run(['python', '-m', 'pytest', '-p', 'hypothesis.extra.pytestplugin',
                          '-q', '--collect-only', f'--hypothesis-seed={seed}', 'tests/', 'hidden_tests/'], seed=seed)
    collections = [e.get('ids') for e in result.events if e.get('event') == 'collection']
    if result.exit_code != 0 or len(collections) != 1 or not collections[0] or [e.get('event') for e in result.events[-2:]] != ['finish', 'returned']:
        raise ValueError('reference_collection_failed: ' + result.stdout + result.stderr)
    ids = collections[0]
    if len(ids) != len(set(ids)):
        raise ValueError('reference_collection_duplicate_ids')
    expected = {'visible': [n for n in ids if n.startswith('tests/')],
                'hidden': [n for n in ids if n.startswith('hidden_tests/')]}
    if not all(expected.values()):
        raise ValueError('reference_collection_empty_suite')
    return expected
