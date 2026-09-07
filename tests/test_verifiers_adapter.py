"""P1.22: the environment is usable through the real Verifiers interface.

Skipped when `verifiers` is not installed — it is an optional extra, since the
package must remain independently valid without a training framework.
"""
from __future__ import annotations

import asyncio
import difflib
from pathlib import Path

import pytest

from src.environment.sandbox import create_pool, docker_available

verifiers = pytest.importorskip("verifiers")
from src.adapters.verifiers_env import OctoRLEnv, load_environment, task_specs  # noqa: E402
from src.injector.buckets import BUCKETS  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
IMAGE = (ROOT / "artifacts/p1/docker-image.id").read_text().strip()
needs_docker = pytest.mark.skipif(not docker_available(), reason="docker daemon unavailable")
SPEC = [{"repo": str(ROOT / "repos/train/record_index"),
         "difficulty": {"source": "mutation", "type": "condition-inversion",
                        "count": 1, "hint": "L0", "span": "single-function"},
         "injector_seed": 11}]


@pytest.fixture(scope="module")
def pool():
    with create_pool(IMAGE, size=1) as value:
        yield value


@pytest.fixture
def env(pool, tmp_path):
    return load_environment(pool, SPEC, workspace=tmp_path)


def test_it_is_the_real_verifiers_interface(env):
    """architecture.md §7 names `Taskset`/`Harness`; neither exists in verifiers."""
    assert isinstance(env, verifiers.StatefulToolEnv)
    assert isinstance(env, verifiers.Environment)
    assert not hasattr(verifiers, "Taskset") and not hasattr(verifiers, "Harness")


def test_the_five_frozen_tools_are_advertised_without_the_sandbox_handle(env):
    names = {t["function"]["name"] for t in env.oai_tools}
    assert names == {"list_files", "search_code", "read_file", "apply_patch", "run_tests"}
    for tool in env.oai_tools:
        properties = set(tool["function"]["parameters"].get("properties", {}))
        assert "session" not in properties, tool["function"]["name"]
        assert "self" not in properties
    assert env.max_turns == 12


def test_dataset_rows_replay_a_manifest_triple(env):
    row = env.dataset[0]
    assert set(row["info"]) >= {"repo", "difficulty", "injector_seed"}
    specs = task_specs([Path("repos/train/record_index")], BUCKETS[:2], "condition-inversion", [11, 12])
    assert len(specs) == 4
    assert all(set(s) == {"repo", "difficulty", "injector_seed"} for s in specs)


@needs_docker
def test_rollout_state_leases_a_sandbox_and_builds_the_prompt(env):
    state = asyncio.run(env.setup_state({"info": dict(env.dataset[0]["info"]), "turn": 0}))
    try:
        assert state["prompt"][0]["role"] == "system"
        assert "octorl_sandbox" in state and "octorl_instance" in state
        args = env.update_tool_args("list_files", {"path": "."}, [], state)
        assert args["session"] is env._sessions[state["octorl_key"]]
        assert "record_index/__init__.py" in env.list_files(".", session=args["session"])
    finally:
        state.pop("octorl_lease").__exit__(None, None, None)


@needs_docker
@pytest.mark.parametrize("repair", [True, False], ids=["repaired", "untouched"])
def test_reward_is_the_verifier_verdict(env, repair):
    state = asyncio.run(env.setup_state({"info": dict(env.dataset[0]["info"]), "turn": 0}))
    instance = state["octorl_instance"]
    session = env._sessions[state["octorl_key"]]
    if repair:
        path = instance.edits[0].path
        diff = "".join(difflib.unified_diff(
            instance.modified[path].splitlines(keepends=True),
            instance.originals[path].splitlines(keepends=True),
            fromfile=f"a/{path}", tofile=f"b/{path}"))
        assert "[error]" not in env.apply_patch(diff, session=session)
    reward = env.hidden_tests_pass(completion=[], state=state)
    assert reward == (1.0 if repair else 0.0)
    assert state["octorl_verdict"]["cheat_flags"] == []
    assert "octorl_lease" not in state, "the sandbox lease must be released by scoring"


@needs_docker
def test_step_cap_is_the_frozen_twelve(env):
    state = asyncio.run(env.setup_state({"info": dict(env.dataset[0]["info"]), "turn": 0}))
    try:
        session = env._sessions[state["octorl_key"]]
        for _ in range(12):
            env.list_files(".", session=session)
        assert session.steps == 12
        assert asyncio.run(env.is_completed([], state)) is True
        assert "[error]" in env.list_files(".", session=session)
    finally:
        state.pop("octorl_lease").__exit__(None, None, None)
