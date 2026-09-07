"""Five fixed tools with exact-context patches and immutable test boundaries."""
from __future__ import annotations

import dataclasses
import json
import re
from pathlib import Path, PurePosixPath
from typing import Any

from .sandbox import CommandResult, Sandbox

PROTECTED = {"tests", "hidden_tests", ".git", ".github", ".gitlab", "__pycache__", ".pytest_cache", ".hypothesis"}


def safe_path(root: Path, name: str) -> Path:
    relative = PurePosixPath(name)
    if relative.is_absolute() or ".." in relative.parts or "\\" in name:
        raise ValueError("path must stay within repository")
    path = root / name
    if path.is_symlink() or any(p.is_symlink() for p in path.parents if p != root.parent):
        raise ValueError("symlink access denied")
    if not path.resolve().is_relative_to(root.resolve()):
        raise ValueError("path outside repository")
    return path


def editable(name: str, allowed: set[str]) -> bool:
    path = PurePosixPath(name)
    return name in allowed and path.suffix == ".py" and not any(part in PROTECTED for part in path.parts) and path.name not in {"conftest.py", "setup.py", "sitecustomize.py", "usercustomize.py"} and not path.name.startswith("test_")


def parse_patch(root: Path, diff: str, allowed: set[str]) -> dict[str, str]:
    """Apply only standard unified diffs, atomically, without fuzzy matching."""
    lines = diff.splitlines(keepends=True)
    changes: dict[str, str] = {}
    i = 0
    while i < len(lines):
        if lines[i].startswith(("diff --git ", "index ")) or not lines[i].strip():
            i += 1
            continue
        if not lines[i].startswith("--- ") or i + 1 >= len(lines) or not lines[i + 1].startswith("+++ "):
            raise ValueError("expected unified ---/+++ file header")
        old = lines[i][4:].strip().split("\t")[0]
        new = lines[i + 1][4:].strip().split("\t")[0]
        old = old[2:] if old.startswith("a/") else old
        new = new[2:] if new.startswith("b/") else new
        if old != new or not editable(new, allowed) or new in changes:
            raise ValueError(f"patch path denied: {new}")
        path = safe_path(root, new)
        original = path.read_text().splitlines(keepends=True)
        output: list[str] = []
        cursor = 0
        i += 2
        hunks = 0
        while i < len(lines) and lines[i].startswith("@@"):
            match = re.match(r"@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@", lines[i])
            if not match:
                raise ValueError("malformed hunk")
            start = int(match[1]) - 1 if int(match[1]) else 0
            old_count = int(match[2] or 1)
            new_count = int(match[4] or 1)
            if start < cursor or start > len(original):
                raise ValueError("overlapping or out-of-range hunk")
            output.extend(original[cursor:start])
            cursor = start
            i += 1
            removed = added = 0
            while i < len(lines) and not lines[i].startswith(("@@", "--- ", "diff --git ")):
                line = lines[i]
                if line.startswith("\\ No newline"):
                    raise ValueError("no-newline markers unsupported; include trailing newline")
                if not line or line[0] not in " +-":
                    break
                marker, value = line[0], line[1:]
                if marker in " -":
                    if cursor >= len(original) or original[cursor] != value:
                        raise ValueError("patch context does not match")
                    cursor += 1
                    removed += 1
                if marker in " +":
                    output.append(value)
                    added += 1
                i += 1
            if (removed, added) != (old_count, new_count):
                raise ValueError("hunk line counts do not match")
            hunks += 1
        if not hunks:
            raise ValueError("patch has no hunks")
        output.extend(original[cursor:])
        changes[new] = "".join(output)
    if not changes:
        raise ValueError("empty patch")
    return changes


@dataclasses.dataclass
class ToolResult:
    stdout: str = ""
    stderr: str = ""
    exit_code: int = 0
    test_result: dict[str, Any] | None = None


class ToolSession:
    MAX_STEPS = 12
    NAMES = {"list_files", "search_code", "read_file", "apply_patch", "run_tests"}

    def __init__(self, sandbox: Sandbox, *, seed: int, affected_tests: dict[str, list[str]] | None = None) -> None:
        if type(seed) is not int:
            raise TypeError("explicit integer seed required")
        self.sandbox = sandbox
        self.root = sandbox.root
        self.seed = seed
        self.steps = 0
        self.allowed = {p.relative_to(self.root).as_posix() for p in self.root.rglob("*.py")}
        self.allowed = {p for p in self.allowed if editable(p, self.allowed)}
        self.changed: set[str] = set()
        self.cheat_flags: list[str] = []
        self.affected_tests = affected_tests or {}
        if not affected_tests and (self.root / "task_metadata.json").exists():
            self.affected_tests = json.loads((self.root / "task_metadata.json").read_text()).get("affected_tests", {})

    def call(self, name: str, arguments: dict[str, Any]) -> ToolResult:
        if self.steps >= self.MAX_STEPS:
            return ToolResult(stderr="12-step limit reached", exit_code=2)
        self.steps += 1
        if name not in self.NAMES:
            return ToolResult(stderr=f"unknown tool: {name}", exit_code=2)
        try:
            return getattr(self, name)(**arguments)
        except (ValueError, OSError, TypeError, re.error) as exc:
            return ToolResult(stderr=str(exc), exit_code=2)

    def list_files(self, path: str = ".") -> ToolResult:
        directory = safe_path(self.root, path)
        names = [p.relative_to(self.root).as_posix() for p in directory.rglob("*") if p.is_file() and not any(x in PROTECTED - {"tests"} for x in p.relative_to(self.root).parts)]
        return ToolResult(stdout="\n".join(sorted(names)))

    def read_file(self, path: str, range: list[int] | None = None) -> ToolResult:
        if any(x in PROTECTED - {"tests"} for x in PurePosixPath(path).parts):
            raise ValueError("protected path")
        lines = safe_path(self.root, path).read_text().splitlines(keepends=True)
        if range is not None:
            if len(range) != 2 or any(type(x) is not int for x in range) or not 1 <= range[0] <= range[1]:
                raise ValueError("range must be inclusive [start, end], starting at 1")
            lines = lines[range[0] - 1:range[1]]
        return ToolResult(stdout="".join(lines))

    def search_code(self, pattern: str) -> ToolResult:
        expression = re.compile(pattern)
        matches = []
        for name in self.list_files().stdout.splitlines():
            if not name.endswith(".py"):
                continue
            for number, line in enumerate(safe_path(self.root, name).read_text().splitlines(), 1):
                if expression.search(line):
                    matches.append(f"{name}:{number}:{line}")
        return ToolResult(stdout="\n".join(matches))

    def apply_patch(self, diff: str) -> ToolResult:
        try:
            changes = parse_patch(self.root, diff, self.allowed)
        except ValueError as exc:
            if "denied" in str(exc):
                self.cheat_flags.append("patch_path_whitelist")
            raise
        for name, source in changes.items():
            safe_path(self.root, name).write_text(source)
        self.changed.update(changes)
        return ToolResult(stdout=f"Updated {', '.join(sorted(changes))}")

    def run_tests(self, subset: list[str] | str | None = None) -> ToolResult:
        if subset is None:
            selected = sorted({test for path in self.changed for test in self.affected_tests.get(path, [])})
            if not selected:
                # No edit yet: run one visible smoke module, never the entire suite.
                selected = [p.relative_to(self.root).as_posix() for p in sorted((self.root / "tests").glob("test_*.py"))][:1]
        else:
            selected = [subset] if isinstance(subset, str) else subset
        if not selected:
            raise ValueError("no affected visible tests available")
        for test in selected:
            filename = test.split("::")[0]
            path = safe_path(self.root, filename)
            if not filename.startswith("tests/") or not path.is_file():
                raise ValueError("only visible tests may be selected")
        result = self.sandbox.run(["python", "-m", "pytest", "-p", "hypothesis.extra.pytestplugin", "-q", "--tb=short", f"--hypothesis-seed={self.seed}", *selected], seed=self.seed)
        counts = {"passed": 0, "failed": 0}
        for count, label in re.findall(r"(\d+) (passed|failed)", result.stdout):
            counts[label] = int(count)
        return ToolResult(result.stdout, result.stderr, result.exit_code, {"exit_code": result.exit_code, **counts, "stdout": result.stdout, "stderr": result.stderr})
