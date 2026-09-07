"""Author second feat-add catalog entries so `count>=2` becomes satisfiable (F2 delta #1).

`inject` filters catalog entries to the requested bug type, so a `count=2` task
needs two same-type entries; every catalog shipped exactly one per type. Each new
entry pairs a complete current function (`before`, sliced verbatim) with a feature
variant (`after`) that adds a keyword-only option **and** regresses the documented
behaviour -- the shape PRD 4.4 specifies for `feat-add`, not a bare operator
mutation, which would make the source indistinguishable from `mutation`.

Re-runnable: entries already present are skipped. Extend NEW to cover more
(repo, type) cells; each addition must pass `validate_entry` and then admission.
"""
import json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import libcst as cst
from libcst.metadata import MetadataWrapper, PositionProvider
from src.injector.catalog import validate_entry

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT / "repos/train/record_index"
SRC = REPO / "record_index/__init__.py"

def verbatim(name: str) -> str:
    src = SRC.read_text()
    w = MetadataWrapper(cst.parse_module(src)); pos = w.resolve(PositionProvider)
    lines = src.splitlines(keepends=True); out = {}
    class V(cst.CSTVisitor):
        METADATA_DEPENDENCIES = (PositionProvider,)
        def visit_FunctionDef(self, n):
            s = pos[n]; out[n.name.value] = "".join(lines[s.start.line-1:s.end.line]); return True
    w.visit(V())
    return out[name]

NEW = [
 ("condition-inversion", "vocabulary",
  '    def vocabulary(self, prefix: str = "", *, suffix: bool = False) -> tuple[str, ...]:\n'
  '        prefix = prefix.casefold()\n'
  '        if suffix:\n'
  '            return tuple(sorted(word for word in self._postings if word.endswith(prefix)))\n'
  '        return tuple(sorted(word for word in self._postings if not word.startswith(prefix)))\n',
  "Add suffix matching to vocabulary; the prefix branch inverts its condition."),
 ("off-by-one", "excerpt",
  'def excerpt(text: str, query: str, width: int = 8, *, ellipsis: bool = False) -> str:\n'
  '    """Return up to width whitespace words centered near the first match."""\n'
  '    if width <= 0:\n'
  '        raise ValueError("positive excerpt width required")\n'
  '    words = text.split()\n'
  '    wanted = set(tokenize(query))\n'
  '    match = next((index for index, word in enumerate(words) if wanted.intersection(tokenize(word))), 0)\n'
  '    start = max(0, min(match - width // 2, len(words) - width))\n'
  '    body = " ".join(words[start:start + width - 1])\n'
  '    return body + "…" if ellipsis else body\n',
  "Add an ellipsis suffix to excerpts; the word window loses its last word."),
]

catalog = json.loads((REPO/"injector_sources.json").read_text())
path = catalog["features"][0]["path"]
for bug_type, func, after, description in NEW:
    before = verbatim(func)
    entry = {"type": bug_type, "path": path, "before": before, "after": after,
             "description": description}
    err = validate_entry(entry, "features")
    if err: raise SystemExit(f"invalid: {err}")
    if any(e["before"] == before and e["type"] == bug_type for e in catalog["features"]):
        print(f"  {bug_type}/{func}: already present"); continue
    catalog["features"].append(entry)
    print(f"  added features/{bug_type} on {func}()")
(REPO/"injector_sources.json").write_text(json.dumps(catalog, indent=2) + "\n")
