"""Explicit source catalogs for replayable history and feature tasks."""
from __future__ import annotations
import json
import textwrap
from pathlib import Path

import libcst as cst

from .core import Difficulty, FeatureChange, HistoricalChange, InjectionResult, inject


class CatalogError(ValueError):
    """A named catalog entry is structurally invalid."""


def _parseable(fragment: str) -> bool:
    dedented = textwrap.dedent(fragment).strip()
    if not dedented:
        return False
    for parser in (cst.parse_expression, cst.parse_statement):
        try:
            parser(dedented)
            return True
        except cst.ParserSyntaxError:
            continue
    return False


def validate_entry(entry: dict, section: str) -> str | None:
    """Return an error string if the entry is malformed, else None."""
    tag = f'{section} ({entry.get("type", "?")})'
    for field in ('before', 'after'):
        val = entry.get(field, '')
        if not val.strip():
            return f'{tag}: empty {field} fragment'
        if not _parseable(val):
            return f'{tag}: {field} fragment is not a parseable CST node'
    if section == 'history' and not entry.get('fix_commit', '').strip():
        return f'{tag}: missing fix_commit'
    if section == 'features':
        if not entry.get('description', '').strip():
            return f'{tag}: missing description'
        if entry.get('before', '').strip() == entry.get('after', '').strip():
            return f'{tag}: before and after are identical'
    return None


def validate_catalog(data: dict) -> list[str]:
    """Return a list of human-readable error strings for malformed entries."""
    errors = []
    for section in ('history', 'features'):
        for entry in data.get(section, []):
            err = validate_entry(entry, section)
            if err:
                errors.append(err)
    return errors


def inject_from_catalog(repo: Path, difficulty: Difficulty, seed: int,
                        catalog: Path | None = None) -> InjectionResult:
    """Load provenance records, validate, filter malformed, then inject."""
    path = catalog or Path(repo) / 'injector_sources.json'
    data = json.loads(path.read_text()) if path.exists() else {}
    valid_history, valid_features = [], []
    rejected = []
    for entry in data.get('history', []):
        err = validate_entry(entry, 'history')
        if err:
            rejected.append((entry.get('type'), err))
        else:
            valid_history.append(HistoricalChange(**entry))
    for entry in data.get('features', []):
        err = validate_entry(entry, 'features')
        if err:
            rejected.append((entry.get('type'), err))
        else:
            valid_features.append(FeatureChange(**entry))
    section = 'history' if difficulty.source == 'commit-rollback' else 'features'
    type_rejected = [msg for typ, msg in rejected if typ == difficulty.type]
    if type_rejected:
        valid_for_type = (
            [e for e in valid_history if e.type == difficulty.type] if difficulty.source == 'commit-rollback'
            else [e for e in valid_features if e.type == difficulty.type]
        )
        if not valid_for_type:
            raise CatalogError(
                f'All {section} entries for {difficulty.type} are malformed:\n  ' +
                '\n  '.join(type_rejected))
    return inject(Path(repo), difficulty, seed,
                  history=valid_history, features=valid_features)
