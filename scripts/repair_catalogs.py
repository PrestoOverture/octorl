"""Repair truncated catalog fragments (F2 delta #1, validation half).

Three repos still carry `before`/`after` fragments that are partial statements -- a
`for` header without its body, a function cut mid-way -- so they do not parse as CST
nodes and `validate_entry` rejects them. Codex repaired `slot_planner` this way;
these were left behind.

Each malformed fragment is widened to its complete enclosing function, sliced
**verbatim** from the file text. Verbatim matters: `inject` checks
`entry.before in git(fix_commit^)` as a raw substring, so a dedented method body
would fail the provenance check even though it parses.
"""
from __future__ import annotations

import json
import subprocess
import sys
import textwrap
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import libcst as cst
from libcst.metadata import MetadataWrapper, PositionProvider

from src.injector.catalog import validate_entry

ROOT = Path(__file__).resolve().parents[1]


def git(repo: Path, *args: str) -> str:
    return subprocess.check_output(["git", "-C", str(repo), *args], text=True)


def functions(source: str) -> dict[str, str]:
    """name -> verbatim source slice, for every def (methods included)."""
    wrapper = MetadataWrapper(cst.parse_module(source))
    positions = wrapper.resolve(PositionProvider)
    lines = source.splitlines(keepends=True)
    found: dict[str, str] = {}

    class Collect(cst.CSTVisitor):
        METADATA_DEPENDENCIES = (PositionProvider,)

        def visit_FunctionDef(self, node: cst.FunctionDef) -> bool:
            span = positions[node]
            found[node.name.value] = "".join(lines[span.start.line - 1:span.end.line])
            return True

    wrapper.visit(Collect())
    return found


def owning_function(source: str, fragment: str) -> str:
    """The function whose body contains the fragment's first substantive line."""
    key = next(line for line in fragment.splitlines() if line.strip())
    matches = [name for name, code in functions(source).items() if key in code]
    if len(matches) != 1:
        raise SystemExit(f"fragment maps to {len(matches)} functions, need exactly 1:\n{key!r}")
    return matches[0]


def line_substitution(before: str, after: str) -> tuple[str, str]:
    """The single line that differs between the old truncated pair."""
    old = [line for line in before.splitlines(keepends=True) if line not in after.splitlines(keepends=True)]
    new = [line for line in after.splitlines(keepends=True) if line not in before.splitlines(keepends=True)]
    if len(old) != 1 or len(new) != 1:
        raise SystemExit(f"expected a one-line difference, got {len(old)}/{len(new)}")
    return old[0], new[0]


def main() -> int:
    changed = 0
    for repo in sorted(p for p in (ROOT / "repos").glob("*/*") if p.is_dir()):
        catalog_path = repo / "injector_sources.json"
        catalog = json.loads(catalog_path.read_text())
        broken = [(section, entry) for section in ("history", "features")
                  for entry in catalog[section] if validate_entry(entry, section)]
        if not broken:
            continue
        print(f"\n{repo.name}: {len(broken)} malformed")
        fix_commit = catalog["fix_commit"]
        for section, entry in broken:
            path = entry["path"]
            buggy = git(repo, "show", f"{fix_commit}^:{path}")
            fixed = git(repo, "show", f"{fix_commit}:{path}")
            current = (repo / path).read_text()
            if section == "history":
                name = owning_function(buggy, entry["before"])
                new_before, new_after = functions(buggy)[name], functions(fixed)[name]
                assert entry["before"] in buggy, "original fragment must come from the buggy commit"
                assert new_before in buggy and new_after in fixed, "slices must stay verbatim"
            else:
                name = owning_function(current, entry["before"])
                new_before = functions(current)[name]
                if new_before.startswith(entry["before"]):
                    # The pair was truncated at a common point: the recorded `after`
                    # is the feature variant's prefix, so splice the untouched tail
                    # of the real function back onto it.
                    new_after = entry["after"] + new_before[len(entry["before"]):]
                else:
                    old_line, new_line = line_substitution(entry["before"], entry["after"])
                    if old_line not in new_before:
                        raise SystemExit(f"{repo.name}/{name}: feature line not in current source")
                    new_after = new_before.replace(old_line, new_line)
            if new_before == new_after:
                raise SystemExit(f"{repo.name}/{name}: widening collapsed the pair")
            for fragment in (new_before, new_after):
                cst.parse_module(textwrap.dedent(fragment))  # must be a parseable node
            entry["before"], entry["after"] = new_before, new_after
            error = validate_entry(entry, section)
            if error:
                raise SystemExit(f"still invalid after repair: {error}")
            print(f"  {section:9} {entry['type']:28} -> widened to {name}()")
            changed += 1
        catalog_path.write_text(json.dumps(catalog, indent=2) + "\n")
    print(f"\nrepaired {changed} entries")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
