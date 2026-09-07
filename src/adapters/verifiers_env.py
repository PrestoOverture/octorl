"""Formal Verifiers interface for the OctoRL repository-repair environment (P1.22).

The core environment (`src/environment`, `src/injector`, `harness/`) depends on no
framework type; this module is the only place `verifiers` is imported, mirroring
the Agent-R1 adapter next to it (architecture.md §7).

**Interface note (P1.22 re-verification, 2026-09-06).** architecture.md §7 and
tech_stack.md §8 describe a `Taskset`/`Harness` pair and mark the naming
provisional. Those names do not exist in `verifiers` 0.1.5: the public surface is
`Environment` -> `MultiTurnEnv` -> `ToolEnv` -> `StatefulToolEnv`, scored by a
`Rubric`. `StatefulToolEnv` is the correct base because each rollout needs its own
sandbox lease, which its `update_tool_args` hook injects without exposing the
handle to the model.
"""
from __future__ import annotations

import dataclasses
import json
import tempfile
from pathlib import Path
from typing import Any, Callable

from datasets import Dataset
from verifiers import Rubric, StatefulToolEnv

from src.environment.harness import package, prepare_grading
from src.environment.tools import ToolSession
from src.environment.verifier import verify
from src.injector.core import Difficulty

ROOT = Path(__file__).resolve().parents[2]
SYSTEM_PROMPT = (ROOT / "harness/prompt.md").read_text()


def task_specs(repos: list[Path], buckets, bug_type: str, seeds: list[int]) -> list[dict[str, Any]]:
    """Serializable (repo, difficulty, seed) triples — the same shape as a frozen
    manifest instance, so an eval set can be replayed from one."""
    return [{"repo": str(repo), "difficulty": {"source": "mutation", "type": bug_type,
                                               "count": b.count, "hint": b.hint, "span": b.span},
             "injector_seed": seed}
            for repo in repos for b in buckets for seed in seeds]


class OctoRLEnv(StatefulToolEnv):
    """Repository repair with the five frozen tools, graded by hidden tests.

    One sandbox is leased per rollout and released when the rollout completes.
    Reward is the verifier's 0/1, never a partial score.
    """

    def __init__(self, pool, specs: list[dict[str, Any]], *, max_turns: int = ToolSession.MAX_STEPS,
                 workspace: Path | None = None, **kwargs: Any) -> None:
        self.pool = pool
        self.workspace = Path(workspace or tempfile.mkdtemp(prefix="octorl-vf-"))
        self._sessions: dict[str, ToolSession] = {}
        dataset = Dataset.from_list([
            {"question": "", "answer": "", "info": {**spec, "index": i}}
            for i, spec in enumerate(specs)])
        rubric = Rubric(funcs=[self.hidden_tests_pass], weights=[1.0])
        super().__init__(tools=[], max_turns=max_turns, dataset=dataset, rubric=rubric,
                         system_prompt=SYSTEM_PROMPT, **kwargs)
        # `session` is injected by update_tool_args and stripped from the schema,
        # so the model never sees a handle to the sandbox.
        for tool in (self.list_files, self.search_code, self.read_file,
                     self.apply_patch, self.run_tests):
            self.add_tool(tool, args_to_skip=["session"])

    # ── lifecycle ──

    async def setup_state(self, state: dict, **kwargs: Any) -> dict:
        info = state["info"]
        instance = package(Path(info["repo"]), Difficulty(**info["difficulty"]),
                           int(info["injector_seed"]), self.workspace / str(info["index"]))
        lease = self.pool.lease(instance.root)
        sandbox = lease.__enter__()
        prepare_grading(instance, sandbox)
        key = f"{instance.task_id}#{info['index']}"
        self._sessions[key] = ToolSession(sandbox, seed=instance.verification_seed,
                                          affected_tests=instance.affected_tests)
        state.update({"octorl_key": key, "octorl_lease": lease, "octorl_sandbox": sandbox,
                      "octorl_instance": instance,
                      "prompt": [{"role": "system", "content": SYSTEM_PROMPT},
                                 {"role": "user", "content": json.dumps(
                                     instance.prompt_instance(), sort_keys=True)}]})
        return await super().setup_state(state, **kwargs)

    def update_tool_args(self, tool_name: str, tool_args: dict, messages, state: dict,
                         **kwargs: Any) -> dict:
        """Bind the rollout's session without ever exposing it to the model."""
        return {**tool_args, "session": self._sessions[state["octorl_key"]]}

    async def is_completed(self, messages, state: dict, **kwargs: Any) -> bool:
        session = self._sessions.get(state.get("octorl_key", ""))
        if session is not None and session.steps >= ToolSession.MAX_STEPS:
            return True
        return await super().is_completed(messages, state, **kwargs)

    # ── scoring ──

    def hidden_tests_pass(self, completion, state: dict, **kwargs: Any) -> float:
        """Reward 1 iff hidden tests all pass and no cheat flag fired."""
        instance, sandbox = state["octorl_instance"], state["octorl_sandbox"]
        session = self._sessions.pop(state["octorl_key"])
        try:
            patched = {name: (sandbox.root / name).read_text(errors="replace")
                       for name in sorted(session.changed) if (sandbox.root / name).is_file()}
            outcome = verify(sandbox, hidden_tests_dir=instance.hidden_tests_dir,
                             originals=instance.originals, patched=patched,
                             tool_cheat_flags=session.cheat_flags,
                             seed=instance.verification_seed,
                             expected_tests=instance.expected_tests)
            state["octorl_verdict"] = {"reward": outcome.reward, "cheat_flags": outcome.cheat_flags,
                                       "hidden_passed": outcome.hidden_passed,
                                       "hidden_failed": outcome.hidden_failed,
                                       "visible_regression": outcome.visible_regression}
            return float(outcome.reward)
        finally:
            state.pop("octorl_lease").__exit__(None, None, None)

    # ── the five frozen tools ──

    @staticmethod
    def _call(session: ToolSession, name: str, arguments: dict[str, Any]) -> str:
        result = session.call(name, arguments)
        return result.stdout if result.exit_code == 0 else f"[error] {result.stderr}"

    def list_files(self, path: str = ".", session: Any = None) -> str:
        """List repository files. Args: path (directory, default ".")."""
        return self._call(session, "list_files", {"path": path})

    def search_code(self, pattern: str, session: Any = None) -> str:
        """Search source files by regular expression. Args: pattern."""
        return self._call(session, "search_code", {"pattern": pattern})

    def read_file(self, path: str, range: list[int] | None = None, session: Any = None) -> str:
        """Read a file, optionally an inclusive 1-based [start, end] line range."""
        return self._call(session, "read_file", {"path": path, "range": range})

    def apply_patch(self, diff: str, session: Any = None) -> str:
        """Apply a unified diff to source files. Tests are not writable."""
        return self._call(session, "apply_patch", {"diff": diff})

    def run_tests(self, subset: list[str] | None = None, session: Any = None) -> str:
        """Run visible tests; defaults to the tests affected by your edits."""
        return self._call(session, "run_tests", {"subset": subset})


def load_environment(pool, specs: list[dict[str, Any]], **kwargs: Any) -> OctoRLEnv:
    """Entry point matching the `verifiers` environment-loading convention."""
    return OctoRLEnv(pool, specs, **kwargs)
