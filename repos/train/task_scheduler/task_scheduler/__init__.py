"""Priority task queue with dependencies, deadlines, and execution tracking."""
from __future__ import annotations

import heapq
from collections import deque
from collections.abc import Iterable
from dataclasses import dataclass, field
from enum import Enum


class Status(Enum):
    PENDING = "pending"
    READY = "ready"
    RUNNING = "running"
    DONE = "done"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class Task:
    identifier: str
    priority: int
    deadline: int | None = None
    dependencies: frozenset[str] = field(default_factory=frozenset)
    status: Status = Status.PENDING
    started_at: int | None = None
    finished_at: int | None = None
    result: str | None = None


class Scheduler:
    """Lower priority numbers run first; ties break by identifier."""

    def __init__(self, now: int = 0) -> None:
        if now < 0:
            raise ValueError("negative clock")
        self.now = now
        self._tasks: dict[str, Task] = {}

    def add(self, identifier: str, priority: int = 0, deadline: int | None = None,
            dependencies: Iterable[str] = ()) -> Task:
        if not identifier or identifier in self._tasks:
            raise ValueError("new nonempty task identifier required")
        if priority < 0:
            raise ValueError("negative priority")
        if deadline is not None and deadline <= self.now:
            raise ValueError("deadline must be in the future")
        deps = frozenset(dependencies)
        for dep in deps:
            if dep not in self._tasks:
                raise KeyError(f"unknown dependency: {dep}")
            if dep == identifier:
                raise ValueError("self-dependency")
        task = Task(identifier, priority, deadline, deps)
        self._tasks[identifier] = task
        self._refresh(identifier)
        return task

    def _refresh(self, identifier: str) -> None:
        task = self._tasks[identifier]
        if task.status != Status.PENDING:
            return
        if all(self._tasks[dep].status == Status.DONE for dep in task.dependencies):
            task.status = Status.READY

    def get(self, identifier: str) -> Task:
        return self._tasks[identifier]

    def cancel(self, identifier: str) -> Task:
        task = self._tasks[identifier]
        if task.status in (Status.DONE, Status.CANCELLED):
            raise ValueError("already finished")
        task.status = Status.CANCELLED
        for other in self._tasks.values():
            if identifier in other.dependencies and other.status in (Status.PENDING, Status.READY):
                other.status = Status.CANCELLED
        return task

    def next_task(self) -> Task | None:
        ready = [t for t in self._tasks.values() if t.status == Status.READY]
        if not ready:
            return None
        return min(ready, key=lambda t: (t.priority, t.identifier))

    def start(self, identifier: str) -> Task:
        task = self._tasks[identifier]
        if task.status != Status.READY:
            raise ValueError("task is not ready")
        task.status = Status.RUNNING
        task.started_at = self.now
        return task

    def complete(self, identifier: str, result: str = "ok") -> Task:
        task = self._tasks[identifier]
        if task.status != Status.RUNNING:
            raise ValueError("task is not running")
        task.status = Status.DONE
        task.finished_at = self.now
        task.result = result
        for other in self._tasks.values():
            if identifier in other.dependencies:
                self._refresh(other.identifier)
        return task

    def fail(self, identifier: str, reason: str = "error") -> Task:
        task = self._tasks[identifier]
        if task.status != Status.RUNNING:
            raise ValueError("task is not running")
        task.status = Status.FAILED
        task.result = reason
        task.finished_at = self.now
        return task

    def retry(self, identifier: str) -> Task:
        task = self._tasks[identifier]
        if task.status != Status.FAILED:
            raise ValueError("only failed tasks can be retried")
        task.status = Status.PENDING
        task.started_at = None
        task.finished_at = None
        task.result = None
        self._refresh(identifier)
        return task

    def advance(self, now: int) -> list[str]:
        if now < self.now:
            raise ValueError("clock cannot move backward")
        self.now = now
        expired = []
        for task in self._tasks.values():
            if task.deadline is not None and task.deadline <= now:
                if task.status in (Status.PENDING, Status.READY):
                    task.status = Status.CANCELLED
                    expired.append(task.identifier)
        return sorted(expired)

    def by_status(self, status: Status) -> tuple[Task, ...]:
        return tuple(t for t in sorted(self._tasks.values(),
                                       key=lambda t: t.identifier) if t.status == status)

    def all_tasks(self) -> tuple[Task, ...]:
        return tuple(sorted(self._tasks.values(), key=lambda t: (t.priority, t.identifier)))

    def is_blocked(self, identifier: str) -> bool:
        task = self._tasks[identifier]
        return any(self._tasks[dep].status != Status.DONE for dep in task.dependencies)

    def dependents(self, identifier: str) -> tuple[str, ...]:
        if identifier not in self._tasks:
            raise KeyError(identifier)
        return tuple(sorted(t.identifier for t in self._tasks.values()
                            if identifier in t.dependencies))

    def critical_path(self) -> list[str]:
        order = self._topological()
        longest: dict[str, int] = {}
        parent: dict[str, str | None] = {}
        for identifier in order:
            task = self._tasks[identifier]
            longest[identifier] = 0
            parent[identifier] = None
            for dep in task.dependencies:
                candidate = longest[dep] + 1
                if candidate > longest[identifier]:
                    longest[identifier] = candidate
                    parent[identifier] = dep
        if not longest:
            return []
        end = max(longest, key=lambda x: longest[x])
        path = [end]
        while parent[path[-1]] is not None:
            path.append(parent[path[-1]])
        return list(reversed(path))

    def _topological(self) -> list[str]:
        degrees: dict[str, int] = {t.identifier: 0 for t in self._tasks.values()}
        for task in self._tasks.values():
            for dep in task.dependencies:
                degrees[task.identifier] = degrees.get(task.identifier, 0)
        for task in self._tasks.values():
            for dep in task.dependencies:
                degrees[task.identifier] += 0
            degrees[task.identifier] = len([d for d in task.dependencies if d in degrees])
        ready = sorted(i for i, d in degrees.items() if d == 0)
        result = []
        while ready:
            node = ready.pop(0)
            result.append(node)
            for task in self._tasks.values():
                if node in task.dependencies:
                    degrees[task.identifier] -= 1
                    if degrees[task.identifier] == 0:
                        bisect_pos = 0
                        for j, r in enumerate(ready):
                            if r > task.identifier:
                                bisect_pos = j
                                break
                            bisect_pos = j + 1
                        ready.insert(bisect_pos, task.identifier)
        return result

    def utilization(self, window_start: int, window_end: int) -> float:
        if window_end <= window_start:
            raise ValueError("invalid window")
        busy = 0
        for task in self._tasks.values():
            if task.started_at is not None and task.finished_at is not None:
                start = max(task.started_at, window_start)
                end = min(task.finished_at, window_end)
                if start < end:
                    busy += end - start
        return busy / (window_end - window_start)


def parse_task_list(text: str) -> list[tuple[str, int]]:
    """Parse 'identifier:priority' lines."""
    result = []
    for number, line in enumerate(text.splitlines(), 1):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if ":" not in line:
            raise ValueError(f"line {number}: missing ':'")
        identifier, _, priority_str = line.partition(":")
        identifier = identifier.strip()
        try:
            priority = int(priority_str.strip())
        except ValueError as error:
            raise ValueError(f"line {number}: invalid priority") from error
        if not identifier:
            raise ValueError(f"line {number}: empty identifier")
        result.append((identifier, priority))
    return result


def execution_order(tasks: Iterable[Task]) -> list[str]:
    pending = {t.identifier: t for t in tasks}
    done: set[str] = set()
    result = []
    changed = True
    while changed:
        changed = False
        ready = sorted(
            (t for t in pending.values() if t.dependencies.issubset(done)),
            key=lambda t: (t.priority, t.identifier),
        )
        if ready:
            chosen = ready[0]
            result.append(chosen.identifier)
            done.add(chosen.identifier)
            del pending[chosen.identifier]
            changed = True
    if pending:
        raise ValueError("circular dependency detected")
    return result
