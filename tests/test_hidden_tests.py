"""Test that hidden test generators produce valid, parseable test files."""
import ast

import pytest

from src.environment.hidden_tests import REPO_GENERATORS, generate


REPOS = list(REPO_GENERATORS.keys())


@pytest.mark.parametrize("repo", REPOS)
def test_generator_produces_valid_python(repo):
    tree = ast.parse(REPO_GENERATORS[repo])
    funcs = [n.name for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)]
    test_funcs = [f for f in funcs if f.startswith("test_")]
    assert len(test_funcs) >= 3, f"{repo} needs at least 3 test functions"


@pytest.mark.parametrize("repo", REPOS)
def test_generate_writes_files(tmp_path, repo):
    dest = generate(repo, tmp_path)
    assert dest.exists()
    assert (dest / "__init__.py").exists()
    test_files = list(dest.glob("test_*.py"))
    assert len(test_files) == 1


def test_all_repos_covered():
    expected = {"parcel_ledger", "record_index", "slot_planner",
                "route_graph", "stock_reservations",
                "config_parser", "metric_aggregator", "task_scheduler"}
    assert expected == set(REPOS)


def test_unknown_repo_raises():
    with pytest.raises(KeyError):
        generate("nonexistent", None)
