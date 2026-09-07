"""Task packaging and the trajectory execution loop (P1.20).

Packaging turns (repo, difficulty, injector_seed) into a self-contained instance
directory; execution drives a policy through the five tools inside a pooled
sandbox and emits one schema-v1 trajectory record.

The turn policy, prompt construction, and pre-submit choice live in `harness/`
and are called into here, never reimplemented -- `harness_version` is a content
hash of those files, so a bypassed call silently voids the replay contract.
"""
from __future__ import annotations

import ast
import dataclasses
import json
import shutil
import sys
import time
import tempfile
from collections import Counter
from textwrap import dedent

import libcst as cst
from libcst.metadata import MetadataWrapper, PositionProvider
from pathlib import Path
from typing import Any, Callable, Iterable, Protocol

import harness.agent_loop as agent_loop
import harness.presubmit_check as presubmit_check
from harness.context_builder import build_prompt

from src.environment.hidden_tests import generate as generate_hidden_tests, REPO_GENERATORS
from src.environment.record import content_hash, harness_version, new_record, freeze_manifest
from src.environment.sandbox import Sandbox
from src.environment.tools import ToolSession
from src.environment.verifier import verify, collect_reference
from src.injector.catalog import inject_from_catalog
from src.injector.core import Difficulty, Edit

# Never copied into an instance: build noise, git history, and the injector's
# own catalog, which records the fix verbatim and would be readable by the agent.
EXCLUDED = {".git", "__pycache__", ".pytest_cache", ".hypothesis", "hidden_tests", "injector_sources.json"}

HARNESS_ROOT = Path(__file__).resolve().parents[2] / "harness"
ADMISSION_RULE_VERSION = 'test-detected-each-edit-v1'


def token_proxy(text: str) -> int:
    """Byte/4 stand-in. A model-backed policy supplies real tokenizer counts."""
    return len(text.encode()) // 4


class Policy(Protocol):
    def __call__(self, prompt: list[dict[str, str]], history: list[dict[str, str]]) -> dict[str, Any] | None:
        """Return {"name", "arguments"} for the next tool call, or None to stop."""


class ScriptedPolicy:
    """Replays a fixed call list. No model, no network -- the throughput gate
    measures CPU-side sandbox cost, so the policy must contribute none of it."""

    def __init__(self, calls: Iterable[dict[str, Any]]) -> None:
        self.calls = list(calls)
        self.index = 0

    def __call__(self, prompt: list[dict[str, str]], history: list[dict[str, str]]) -> dict[str, Any] | None:
        if self.index >= len(self.calls):
            return None
        call = self.calls[self.index]
        self.index += 1
        return {"name": call["name"], "arguments": dict(call.get("arguments", {}))}


@dataclasses.dataclass(frozen=True)
class TaskInstance:
    task_id: str
    repo_name: str
    repo_ref: str
    difficulty: Difficulty
    injector_seed: int
    root: Path
    hidden_tests_dir: Path
    issue: str
    originals: dict[str, str]
    modified: dict[str, str]
    edits: list[Edit]
    affected_tests: dict[str, list[str]]
    expected_tests: dict[str, list[str]] = dataclasses.field(default_factory=dict, compare=False)

    @property
    def verification_seed(self) -> int:
        # Independent of sampling_seed, stable across processes and regenerations.
        return int(content_hash({'domain': 'verification-v1', 'repo_ref': self.repo_ref,
                                 'seed': self.injector_seed,
                                 'difficulty': dataclasses.asdict(self.difficulty)})[:8], 16)

    def prompt_instance(self) -> dict[str, Any]:
        """The agent-visible view. Never carries originals, edits, or provenance."""
        return {"task_id": self.task_id, "issue": self.issue, "difficulty": dataclasses.asdict(self.difficulty)}


def repo_content_hash(repo: Path) -> str:
    sources = {}
    for path in sorted(repo.rglob("*")):
        rel = path.relative_to(repo)
        if any(part in EXCLUDED - {'injector_sources.json'} for part in rel.parts) or not path.is_file():
            continue
        sources[rel.as_posix()] = path.read_text(errors="replace")
    project = Path(__file__).resolve().parents[2]
    image_file = project / 'artifacts/p1/docker-image.id'
    if image_file.exists():
        sandbox_descriptor = image_file.read_text().strip()
    else:
        import pytest as _pytest, hypothesis as _hyp
        sandbox_descriptor = f"process:python={sys.version_info[:3]}:pytest={_pytest.__version__}:hypothesis={_hyp.__version__}"
    return content_hash({'sources': sources, 'hidden_generator': REPO_GENERATORS.get(repo.name),
        'admission_rule_version': ADMISSION_RULE_VERSION,
        'implementation': {name: (project / name).read_text() for name in (
            'src/environment/harness.py', 'src/environment/verifier.py', 'src/environment/sandbox.py',
            'src/environment/hidden_tests.py', 'scripts/grading_runner.py',
            'src/injector/core.py', 'src/injector/catalog.py')},
        'sandbox_image': sandbox_descriptor})


def repo_ref(repo: Path) -> str:
    """Identifies the repo *and* pins its content, so regeneration detects drift."""
    return f"{repo.name}@sha256:{repo_content_hash(repo)}"


def _test_identifiers(root: Path, names: set[str]) -> tuple[list[str], list[str]]:
    """(tests touching `names`, every visible test module) as pytest ids."""
    selected, modules = [], []
    for path in sorted((root / "tests").glob("test_*.py")):
        rel = path.relative_to(root).as_posix()
        modules.append(rel)
        try:
            tree = ast.parse(path.read_text())
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) or not node.name.startswith("test"):
                continue
            used = {n.id for n in ast.walk(node) if isinstance(n, ast.Name)} | {
                n.attr for n in ast.walk(node) if isinstance(n, ast.Attribute)}
            if used & names:
                selected.append(f"{rel}::{node.name}")
    return selected, modules


def affected_tests_for(root: Path, edits: list[Edit]) -> dict[str, list[str]]:
    """Map each edited source file to the visible tests that exercise it.

    Without this the tool layer falls back to a single smoke module, which makes
    per-step pytest cost -- the dominant term in the throughput gate -- look far
    cheaper than it is."""
    names = {part for edit in edits for part in edit.function.split(".") if part}
    selected, modules = _test_identifiers(root, names)
    return {edit.path: sorted(selected) or modules for edit in edits}


def package(repo: Path, difficulty: Difficulty, injector_seed: int, dest: Path) -> TaskInstance:
    """Build one task instance. Reads `repo` (including .git for commit-rollback
    sources) but never writes to it."""
    repo = Path(repo)
    dest = Path(dest)
    result = inject_from_catalog(repo, difficulty, injector_seed)

    root = dest / "instance"
    if root.exists():
        shutil.rmtree(root)
    root.mkdir(parents=True)
    for path in sorted(repo.rglob("*")):
        rel = path.relative_to(repo)
        if any(part in EXCLUDED for part in rel.parts) or path.is_symlink():
            continue
        if path.is_dir():
            (root / rel).mkdir(parents=True, exist_ok=True)
        elif path.is_file():
            (root / rel).parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, root / rel)
    for name, source in result.modified.items():
        (root / name).write_text(source)

    staging = dest / "hidden"
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)
    hidden_tests_dir = generate_hidden_tests(repo.name, staging)

    affected = affected_tests_for(root, result.edits)
    bucket_id = f"c{difficulty.count}-{difficulty.hint}-{difficulty.span}"
    task_id = f"{repo.name}:{bucket_id}:{injector_seed}"
    (root / "task_metadata.json").write_text(json.dumps(
        {"task_id": task_id, "issue": result.issue, "affected_tests": affected}, indent=2, sort_keys=True) + "\n")

    return TaskInstance(
        task_id=task_id, repo_name=repo.name, repo_ref=repo_ref(repo), difficulty=difficulty,
        injector_seed=injector_seed, root=root, hidden_tests_dir=hidden_tests_dir, issue=result.issue,
        originals=dict(result.originals), modified=dict(result.modified), edits=list(result.edits),
        affected_tests=affected)


def regenerate(spec: dict[str, Any], repos_root: Path, dest: Path) -> TaskInstance:
    """Rebuild an instance from a frozen manifest triple.

    Raises if the repo has drifted since freezing -- the manifest is the
    reproducibility contract, so a silent content change must not pass."""
    name, _, digest = spec["repo_ref"].partition("@")
    candidates = [p for p in sorted(Path(repos_root).glob("*/*")) if p.name == name and p.is_dir()]
    if not candidates:
        raise FileNotFoundError(f"no repository named {name!r} under {repos_root}")
    repo = candidates[0]
    actual = f"sha256:{repo_content_hash(repo)}"
    if actual != digest:
        raise ValueError(f"repository {name} drifted since the manifest was frozen: {digest} != {actual}")
    return package(repo, Difficulty(**spec["difficulty"]), spec["injector_seed"], dest)


def _patched_sources(sandbox: Sandbox, changed: Iterable[str]) -> dict[str, str]:
    out = {}
    for name in sorted(changed):
        path = sandbox.root / name
        if path.is_file():
            out[name] = path.read_text(errors="replace")
    return out


def run_trajectory(
    instance: TaskInstance,
    sandbox: Sandbox,
    policy: Policy,
    *,
    sampling_seed: int,
    trajectory_id: str,
    policy_version: str,
    base_model_revision: str,
    manifest_ref: str,
    harness_root: Path = HARNESS_ROOT,
    adapter_id: str | None = None,
    count_tokens: Callable[[str], int] = token_proxy,
) -> dict[str, Any]:
    """Drive one trajectory to completion and return its schema-v1 record.

    The sandbox must already hold this instance's tree (via `SandboxPool.lease`)."""
    prepare_grading(instance, sandbox)
    origin = time.monotonic()
    record = new_record(
        trajectory_id=trajectory_id, task_id=instance.task_id,
        difficulty=dataclasses.asdict(instance.difficulty), injector_seed=instance.injector_seed,
        sampling_seed=sampling_seed, repo_ref=instance.repo_ref, manifest_ref=manifest_ref,
        container_id=sandbox.container_id, policy_version=policy_version,
        base_model_revision=base_model_revision, harness_root=harness_root, adapter_id=adapter_id)

    session = ToolSession(sandbox, seed=sampling_seed, affected_tests=instance.affected_tests)
    history: list[dict[str, str]] = []
    turns: list[dict[str, Any]] = []
    totals = {"input": 0, "output": 0, "tool": 0}
    finished = False
    presubmit_used = False

    turns_start = time.monotonic() - origin
    while agent_loop.should_continue(session.steps, finished):
        prompt = build_prompt(instance.prompt_instance(), history, sampling_seed)
        call = policy(prompt, history)
        if call is None:
            # The agent stopped; the pre-submit check decides what runs before
            # submission when edits exist.
            if presubmit_used or not session.changed:
                break
            call = presubmit_check.check_request(sorted(session.changed))
            presubmit_used = True

        started = time.monotonic() - origin
        assistant = json.dumps(call, sort_keys=True)
        result = session.call(call["name"], call["arguments"])
        counts = {
            "input": sum(count_tokens(m["content"]) for m in prompt),
            "output": count_tokens(assistant),
            "tool": count_tokens(result.stdout) + count_tokens(result.stderr),
        }
        for key, value in counts.items():
            totals[key] += value
        turns.append({
            "index": len(turns), "start": started, "end": time.monotonic() - origin,
            "assistant": assistant,
            "tool_calls": [{"name": call["name"], "arguments": call["arguments"],
                            "stdout": result.stdout, "stderr": result.stderr, "exit_code": result.exit_code}],
            "run_tests": result.test_result, "token_counts": counts,
        })
        history.append({"role": "assistant", "content": assistant})
        history.append({"role": "tool", "content": result.stdout or result.stderr})

    verify_start = time.monotonic() - origin
    outcome = verify(
        sandbox, hidden_tests_dir=instance.hidden_tests_dir, originals=instance.originals,
        patched=_patched_sources(sandbox, session.changed), tool_cheat_flags=session.cheat_flags,
        seed=instance.verification_seed, expected_tests=instance.expected_tests)

    record["turns"] = turns
    record["token_counts"] = totals
    record["wall_clock_segments"] = [
        {"name": "turns", "start": turns_start, "end": verify_start},
        {"name": "verify", "start": verify_start, "end": time.monotonic() - origin},
    ]
    record["reward"] = outcome.reward
    record["cheat_flags"] = list(outcome.cheat_flags)
    return record


def prepare_grading(instance: TaskInstance, sandbox: Sandbox) -> None:
    """Reference collection is once per instance, never from a model's patch."""
    if instance.expected_tests:
        return
    with tempfile.TemporaryDirectory(prefix='octorl-reference-') as td:
        root = Path(td) / 'reference'
        shutil.copytree(instance.root, root)
        for name, source in instance.originals.items():
            (root / name).write_text(source)
        try:
            sandbox.reset(root)
            expected = collect_reference(sandbox, instance.hidden_tests_dir, instance.verification_seed)
            instance.expected_tests.update(expected)
        finally:
            sandbox.reset(instance.root)


def isolated_defect(instance: TaskInstance, edit: Edit) -> dict[str, str]:
    """Leave exactly one advertised edit on the clean tree using LibCST.

    Match the original location and fragment, not the injector's next random
    draw. Unsupported/ambiguous edits reject admission instead of losing edits.
    """
    sources = dict(instance.originals)
    wrapper = MetadataWrapper(cst.parse_module(sources[edit.path]))
    matches = []

    class Locate(cst.CSTVisitor):
        METADATA_DEPENDENCIES = (PositionProvider,)

        def on_visit(self, node):
            if not isinstance(node, (cst.BaseExpression, cst.BaseSmallStatement, cst.BaseStatement, cst.ExceptHandler)):
                return True
            if self.get_metadata(PositionProvider, node).start.line != edit.line:
                return True
            if wrapper.module.code_for_node(node).strip() != edit.before.strip():
                return True
            if isinstance(node, cst.BaseExpression):
                replacement = cst.parse_expression(edit.after)
            elif isinstance(node, cst.ExceptHandler):
                replacement = cst.parse_statement('try:\n    pass\n' + dedent(edit.after)).handlers[0]
            else:
                replacement = cst.parse_statement(dedent(edit.after))
                if isinstance(node, cst.BaseSmallStatement):
                    if not isinstance(replacement, cst.SimpleStatementLine) or len(replacement.body) != 1:
                        raise ValueError('single_edit_replacement_shape')
                    replacement = replacement.body[0]
            matches.append((node, replacement))
            return False

    wrapper.visit(Locate())
    if len(matches) != 1:
        raise ValueError('single_edit_not_uniquely_reconstructable')
    old, new = matches[0]

    class Replace(cst.CSTTransformer):
        def on_leave(self, original_node, updated_node):
            return new if original_node is old else updated_node

    sources[edit.path] = wrapper.module.visit(Replace()).code
    return sources


def admit(instance: TaskInstance, pool) -> dict[str, Any]:
    """Judge a fixed requested instance, without a model or replacement draws.

    All four conditions use the same full grader and instance-scoped seed.
    Compilation/collection/setup failures are not evidence of a detected defect.
    """
    started = time.monotonic()
    result = {'task_id': instance.task_id, 'repo_ref': instance.repo_ref,
              'repo': instance.repo_name, 'difficulty': dataclasses.asdict(instance.difficulty),
              'injector_seed': instance.injector_seed, 'verification_seed': instance.verification_seed,
              'admission_rule_version': ADMISSION_RULE_VERSION, 'admitted': False,
              'reasons': [], 'checks': {}}
    with tempfile.TemporaryDirectory(prefix='octorl-admission-') as td:
        tree = Path(td) / 'tree'
        shutil.copytree(instance.root, tree)
        for name, text in instance.originals.items():
            (tree / name).write_text(text)
        with pool.lease(tree) as box:
            try:
                expected = collect_reference(box, instance.hidden_tests_dir, instance.verification_seed)
                instance.expected_tests.clear()
                instance.expected_tests.update(expected)
            except ValueError as error:
                result['reasons'].append('clean:reference_collection_failed')
                result['collection_error'] = str(error)
                result['seconds'] = time.monotonic() - started
                return result

            def grade(label, sources, must_pass):
                for name, text in instance.originals.items():
                    (tree / name).write_text(text)
                for name, text in sources.items():
                    (tree / name).write_text(text)
                box.reset(tree)
                outcome = verify(box, hidden_tests_dir=instance.hidden_tests_dir,
                    originals=instance.originals, patched={}, tool_cheat_flags=[],
                    seed=instance.verification_seed, expected_tests=expected)
                result['checks'][label] = dataclasses.asdict(outcome)
                accounting = [outcome.details[key] for key in ('hidden_accounting', 'visible_accounting')]
                if must_pass:
                    good = outcome.reward == 1
                else:
                    good = (outcome.reward == 0 and not outcome.cheat_flags and
                            all(a['complete'] for a in accounting) and
                            any(a['failed_tests'] for a in accounting))
                if not good:
                    result['reasons'].append(label + (':reference_does_not_pass' if must_pass else ':no_genuine_test_failure'))
                return good

            grade('clean', instance.originals, True)
            grade('buggy', instance.modified, False)
            grade('repair', instance.originals, True)
            if len(instance.edits) != instance.difficulty.count:
                result['reasons'].append('advertised_edit_count_mismatch')
            if instance.difficulty.count > 1:
                for i, edit in enumerate(instance.edits):
                    try:
                        sources = isolated_defect(instance, edit)
                    except (ValueError, cst.ParserSyntaxError) as error:
                        result['reasons'].append(f'edit_{i}:reconstruction_failed:{error}')
                    else:
                        grade(f'edit_{i}', sources, False)
    result['admitted'] = not result['reasons']
    result['seconds'] = time.monotonic() - started
    return result


def admission_report(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Descriptive census counts; not a model-dependent or adaptive population."""
    grouped = {}
    accepted = [row for row in rows if row['admitted']]
    for row in rows:
        d = row['difficulty']
        bucket = f"c{d['count']}-{d['hint']}-{d['span']}"
        key = '|'.join((row['repo'], d['type'], d['source'], bucket))
        group = grouped.setdefault(key, {'admitted': 0, 'rejected': 0, 'reasons': {}})
        group['admitted' if row['admitted'] else 'rejected'] += 1
        for reason in row['reasons']:
            group['reasons'][reason] = group['reasons'].get(reason, 0) + 1
    proportions = {}
    for dimension in ('repo', 'type', 'source', 'span', 'bucket'):
        def value(row):
            if dimension == 'repo':
                return row['repo']
            d = row['difficulty']
            return f"c{d['count']}-{d['hint']}-{d['span']}" if dimension == 'bucket' else d[dimension]
        counts = Counter(value(row) for row in accepted)
        proportions[dimension] = {name: {'count': count, 'proportion': count / len(accepted)} for name, count in sorted(counts.items())}
    return {'admission_rule_version': ADMISSION_RULE_VERSION,
            'population': 'exact requested instances; no model outcomes; no retries or substitutions',
            'requested': len(rows), 'admitted': len(accepted), 'rejected': len(rows) - len(accepted),
            'by_cell': grouped, 'admitted_proportions': proportions, 'instances': rows}


def freeze_admitted_manifest(path: Path, instances: list[TaskInstance], pool, *,
                             manifest_id: str, manifest_role: str, report_path: Path) -> dict:
    """The environment's manifest authoring entry point. Never admits by schema alone.

    record.freeze_manifest remains the unchanged low-level JSON writer; callers
    generating research populations must use this admission-gated API.
    """
    rows = [admit(instance, pool) for instance in instances]
    report = admission_report(rows)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + '\n')
    admitted = [i for i, row in zip(instances, rows) if row['admitted']]
    if not admitted:
        raise ValueError('no_admitted_instances')
    manifest = {'schema_version': 1, 'manifest_id': manifest_id, 'manifest_role': manifest_role,
                'instances': [{'task_id': i.task_id, 'repo_ref': i.repo_ref,
                               'injector_seed': i.injector_seed,
                               'difficulty': dataclasses.asdict(i.difficulty)} for i in admitted]}
    freeze_manifest(path, manifest)
    return manifest
