"""Source-balanced injector coverage over the full difficulty grid (F2 delta #2).

Enumerates every (repo x source x bucket x type) cell explicitly -- never samples.
`Bucket.sample` draws a source at random, so a seed-sampled grid silently omits
whole sources; the original 432-cell grid contained no `mutation` at all.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.injector.buckets import BUCKETS
from src.injector.catalog import CatalogError, inject_from_catalog
from src.injector.core import BUG_TYPES, SOURCES, Difficulty

ROOT = Path(__file__).resolve().parents[1]


def classify(error: Exception) -> tuple[str, str]:
    """(status, named reason) -- an unsupported cell must never be a bare error."""
    name = type(error).__name__
    text = str(error)
    if isinstance(error, CatalogError):
        return "catalog_invalid", text
    if name == "CSTCodegenError":
        return "codegen_crash", text
    if "No eligible" in text:
        return "no_eligible_task", text
    return "error", f"{name}: {text}"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--out", default=str(ROOT / "artifacts/p1/injector_coverage.json"))
    args = parser.parse_args()

    repos = sorted(p for p in (ROOT / "repos").glob("*/*") if p.is_dir())
    rows = []
    for repo in repos:
        for source in SOURCES:
            for bucket in BUCKETS:
                for bug_type in BUG_TYPES:
                    difficulty = Difficulty(source, bug_type, bucket.count, bucket.hint, bucket.span)
                    row = {"split": repo.parent.name, "repo": repo.name, "source": source,
                           "type": bug_type, "bucket": bucket.id, "count": bucket.count,
                           "hint": bucket.hint, "span": bucket.span}
                    try:
                        result = inject_from_catalog(repo, difficulty, args.seed)
                    except Exception as error:  # noqa: BLE001 - every cell is classified
                        status, reason = classify(error)
                        row.update({"status": status, "reason": reason})
                    else:
                        row.update({"status": "ok", "edits": len(result.edits),
                                    "files": sorted({e.path for e in result.edits})})
                        assert len(result.edits) == bucket.count, row
                    rows.append(row)

    counts = Counter(r["status"] for r in rows)
    by_source = {s: dict(Counter(r["status"] for r in rows if r["source"] == s)) for s in SOURCES}
    multi = {s: sum(1 for r in rows if r["source"] == s and r["count"] > 1 and r["status"] == "ok")
             for s in SOURCES}
    repos_ok = defaultdict(set)
    for r in rows:
        if r["status"] == "ok":
            repos_ok[(r["source"], r["bucket"])].add(r["repo"])
    unsupported = sorted({(r["source"], r["bucket"]) for r in rows
                          if not repos_ok.get((r["source"], r["bucket"]))})

    report = {
        "seed": args.seed, "enumerated": len(rows),
        "grid": {"repos": len(repos), "sources": len(SOURCES), "buckets": len(BUCKETS),
                 "types": len(BUG_TYPES)},
        "status_counts": dict(counts), "status_by_source": by_source,
        "multi_edit_ok_by_source": multi,
        "repos_generating_by_source_bucket": {f"{s}|{b}": sorted(v) for (s, b), v in
                                              sorted(repos_ok.items())},
        "unsupported_cells": [{"source": s, "bucket": b, "reason":
                               "no repository can satisfy this source at this bucket"}
                              for s, b in unsupported],
        "cells": rows,
    }
    Path(args.out).write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")

    print(f"enumerated {len(rows)} cells "
          f"({len(repos)} repos x {len(SOURCES)} sources x {len(BUCKETS)} buckets x {len(BUG_TYPES)} types)")
    print("status:", dict(counts))
    for source in SOURCES:
        print(f"  {source:16} {by_source[source]}  multi-edit ok: {multi[source]}")
    print(f"unsupported (source, bucket) pairs: {len(unsupported)}")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
