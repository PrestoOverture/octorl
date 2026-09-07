"""Tests for task_scheduler."""
import pytest
from task_scheduler import Status, Task, Scheduler, parse_task_list, execution_order


# ── lifecycle: add → start → complete ────────────────────────────

class TestTaskLifecycle:
    def test_add_creates_ready_task(self):
        s = Scheduler()
        t = s.add("a", priority=1)
        assert t.identifier == "a"
        assert t.priority == 1
        assert t.status == Status.READY

    def test_start_sets_running(self):
        s = Scheduler()
        s.add("a")
        t = s.start("a")
        assert t.status == Status.RUNNING
        assert t.started_at == 0

    def test_complete_sets_done_and_finished_at(self):
        s = Scheduler()
        s.add("a")
        s.start("a")
        s.advance(5)
        t = s.complete("a", "success")
        assert t.status == Status.DONE
        assert t.finished_at == 5
        assert t.result == "success"

    def test_full_lifecycle_timestamps(self):
        s = Scheduler()
        s.add("a")
        s.start("a")
        s.advance(10)
        s.complete("a")
        t = s.get("a")
        assert t.status == Status.DONE
        assert t.started_at == 0
        assert t.finished_at == 10

    def test_add_duplicate_raises(self):
        s = Scheduler()
        s.add("a")
        with pytest.raises(ValueError):
            s.add("a")

    def test_add_empty_raises(self):
        s = Scheduler()
        with pytest.raises(ValueError):
            s.add("")

    def test_negative_priority_raises(self):
        s = Scheduler()
        with pytest.raises(ValueError):
            s.add("a", priority=-1)

    def test_start_non_ready_raises(self):
        s = Scheduler()
        s.add("a")
        s.start("a")
        with pytest.raises(ValueError):
            s.start("a")

    def test_complete_non_running_raises(self):
        s = Scheduler()
        s.add("a")
        with pytest.raises(ValueError):
            s.complete("a")

    def test_negative_clock_raises(self):
        with pytest.raises(ValueError):
            Scheduler(now=-1)


# ── dependencies ─────────────────────────────────────────────────

class TestDependencies:
    def test_pending_with_unmet_deps(self):
        s = Scheduler()
        s.add("a")
        s.add("b", dependencies=["a"])
        assert s.get("b").status == Status.PENDING

    def test_ready_when_deps_done(self):
        s = Scheduler()
        s.add("a")
        s.add("b", dependencies=["a"])
        s.start("a")
        s.complete("a")
        assert s.get("b").status == Status.READY

    def test_chain_three_deep(self):
        s = Scheduler()
        s.add("a")
        s.add("b", dependencies=["a"])
        s.add("c", dependencies=["b"])
        assert s.get("b").status == Status.PENDING
        assert s.get("c").status == Status.PENDING
        s.start("a")
        s.complete("a")
        assert s.get("b").status == Status.READY
        assert s.get("c").status == Status.PENDING
        s.start("b")
        s.complete("b")
        assert s.get("c").status == Status.READY

    def test_is_blocked(self):
        s = Scheduler()
        s.add("a")
        s.add("b", dependencies=["a"])
        assert s.is_blocked("b")
        s.start("a")
        s.complete("a")
        assert not s.is_blocked("b")

    def test_unknown_dep_raises(self):
        s = Scheduler()
        with pytest.raises(KeyError):
            s.add("b", dependencies=["nope"])

    def test_dependents(self):
        s = Scheduler()
        s.add("a")
        s.add("b", dependencies=["a"])
        s.add("c", dependencies=["a"])
        assert s.dependents("a") == ("b", "c")

    def test_dependents_unknown_raises(self):
        s = Scheduler()
        with pytest.raises(KeyError):
            s.dependents("z")


# ── cancellation cascades ────────────────────────────────────────

class TestCancellation:
    def test_cancel_ready(self):
        s = Scheduler()
        s.add("a")
        t = s.cancel("a")
        assert t.status == Status.CANCELLED

    def test_cancel_cascades_to_direct_dependents(self):
        s = Scheduler()
        s.add("a")
        s.add("b", dependencies=["a"])
        s.add("c", dependencies=["a"])
        s.cancel("a")
        assert s.get("b").status == Status.CANCELLED
        assert s.get("c").status == Status.CANCELLED

    def test_cancel_does_not_cascade_transitively(self):
        s = Scheduler()
        s.add("a")
        s.add("b", dependencies=["a"])
        s.add("c", dependencies=["b"])
        s.cancel("a")
        assert s.get("b").status == Status.CANCELLED
        assert s.get("c").status == Status.PENDING

    def test_cancel_done_raises(self):
        s = Scheduler()
        s.add("a")
        s.start("a")
        s.complete("a")
        with pytest.raises(ValueError):
            s.cancel("a")

    def test_cancel_already_cancelled_raises(self):
        s = Scheduler()
        s.add("a")
        s.cancel("a")
        with pytest.raises(ValueError):
            s.cancel("a")

    def test_cancel_running(self):
        s = Scheduler()
        s.add("a")
        s.start("a")
        t = s.cancel("a")
        assert t.status == Status.CANCELLED


# ── deadlines ────────────────────────────────────────────────────

class TestDeadlines:
    def test_expires_at_deadline(self):
        s = Scheduler()
        s.add("a", deadline=5)
        expired = s.advance(5)
        assert expired == ["a"]
        assert s.get("a").status == Status.CANCELLED

    def test_safe_before_deadline(self):
        s = Scheduler()
        s.add("a", deadline=5)
        expired = s.advance(4)
        assert expired == []
        assert s.get("a").status == Status.READY

    def test_past_deadline_raises(self):
        s = Scheduler(now=10)
        with pytest.raises(ValueError):
            s.add("a", deadline=5)

    def test_backward_clock_raises(self):
        s = Scheduler(now=5)
        with pytest.raises(ValueError):
            s.advance(3)

    def test_running_not_expired(self):
        s = Scheduler()
        s.add("a", deadline=5)
        s.start("a")
        expired = s.advance(5)
        assert expired == []
        assert s.get("a").status == Status.RUNNING

    def test_deadline_equal_to_now_raises(self):
        s = Scheduler(now=5)
        with pytest.raises(ValueError):
            s.add("a", deadline=5)


# ── retry ────────────────────────────────────────────────────────

class TestRetry:
    def test_retry_resets_to_ready(self):
        s = Scheduler()
        s.add("a")
        s.start("a")
        s.fail("a", "oops")
        t = s.retry("a")
        assert t.status == Status.READY
        assert t.started_at is None
        assert t.finished_at is None
        assert t.result is None

    def test_retry_non_failed_raises(self):
        s = Scheduler()
        s.add("a")
        with pytest.raises(ValueError):
            s.retry("a")

    def test_fail_records_reason(self):
        s = Scheduler()
        s.add("a")
        s.start("a")
        t = s.fail("a", "timeout")
        assert t.result == "timeout"
        assert t.finished_at == 0

    def test_fail_non_running_raises(self):
        s = Scheduler()
        s.add("a")
        with pytest.raises(ValueError):
            s.fail("a")

    def test_retry_then_complete(self):
        s = Scheduler()
        s.add("a")
        s.start("a")
        s.fail("a")
        s.retry("a")
        s.start("a")
        s.advance(7)
        t = s.complete("a")
        assert t.status == Status.DONE
        assert t.finished_at == 7


# ── next_task priority selection ─────────────────────────────────

class TestNextTask:
    def test_picks_lowest_priority(self):
        s = Scheduler()
        s.add("a", priority=5)
        s.add("b", priority=1)
        s.add("c", priority=3)
        assert s.next_task().identifier == "b"

    def test_ties_break_by_id(self):
        s = Scheduler()
        s.add("b", priority=1)
        s.add("a", priority=1)
        assert s.next_task().identifier == "a"

    def test_none_when_empty(self):
        s = Scheduler()
        assert s.next_task() is None

    def test_skips_non_ready(self):
        s = Scheduler()
        s.add("a", priority=0)
        s.start("a")
        s.add("b", priority=1)
        assert s.next_task().identifier == "b"


# ── critical path ────────────────────────────────────────────────

class TestCriticalPath:
    def test_linear_chain(self):
        s = Scheduler()
        s.add("a")
        s.start("a")
        s.complete("a")
        s.add("b", dependencies=["a"])
        s.start("b")
        s.complete("b")
        s.add("c", dependencies=["b"])
        assert s.critical_path() == ["a", "b", "c"]

    def test_empty_scheduler(self):
        s = Scheduler()
        assert s.critical_path() == []

    def test_independent_tasks(self):
        s = Scheduler()
        s.add("x")
        s.add("y")
        path = s.critical_path()
        assert len(path) == 1

    def test_diamond(self):
        s = Scheduler()
        s.add("a")
        s.start("a")
        s.complete("a")
        s.add("b", dependencies=["a"])
        s.add("c", dependencies=["a"])
        s.start("b")
        s.complete("b")
        s.start("c")
        s.complete("c")
        s.add("d", dependencies=["b", "c"])
        path = s.critical_path()
        assert path[0] == "a"
        assert path[-1] == "d"
        assert len(path) == 3


# ── utilization ──────────────────────────────────────────────────

class TestUtilization:
    def test_full(self):
        s = Scheduler()
        s.add("a")
        s.start("a")
        s.advance(10)
        s.complete("a")
        assert s.utilization(0, 10) == 1.0

    def test_zero(self):
        s = Scheduler()
        s.add("a")
        assert s.utilization(0, 10) == 0.0

    def test_partial(self):
        s = Scheduler()
        s.add("a")
        s.start("a")
        s.advance(5)
        s.complete("a")
        assert s.utilization(0, 10) == 0.5

    def test_invalid_window_raises(self):
        s = Scheduler()
        with pytest.raises(ValueError):
            s.utilization(10, 5)

    def test_overlapping_window(self):
        s = Scheduler()
        s.add("a")
        s.start("a")
        s.advance(10)
        s.complete("a")
        assert s.utilization(5, 15) == 0.5


# ── by_status ────────────────────────────────────────────────────

class TestByStatus:
    def test_filters_correctly(self):
        s = Scheduler()
        s.add("a")
        s.add("b")
        s.start("a")
        ready = s.by_status(Status.READY)
        assert [t.identifier for t in ready] == ["b"]

    def test_alphabetical_order(self):
        s = Scheduler()
        s.add("c")
        s.add("a")
        s.add("b")
        ready = s.by_status(Status.READY)
        assert [t.identifier for t in ready] == ["a", "b", "c"]


# ── all_tasks ────────────────────────────────────────────────────

class TestAllTasks:
    def test_priority_then_id_order(self):
        s = Scheduler()
        s.add("b", priority=1)
        s.add("a", priority=1)
        s.add("c", priority=0)
        result = s.all_tasks()
        assert [t.identifier for t in result] == ["c", "a", "b"]


# ── parse_task_list ──────────────────────────────────────────────

class TestParseTaskList:
    def test_valid_lines(self):
        assert parse_task_list("alpha:1\nbeta:2") == [("alpha", 1), ("beta", 2)]

    def test_skips_comments_and_blanks(self):
        text = "# comment\nalpha:1\n\nbeta:2"
        assert parse_task_list(text) == [("alpha", 1), ("beta", 2)]

    def test_missing_colon_raises(self):
        with pytest.raises(ValueError, match="missing ':'"):
            parse_task_list("bad line")

    def test_invalid_priority_raises(self):
        with pytest.raises(ValueError, match="invalid priority"):
            parse_task_list("alpha:abc")

    def test_empty_identifier_raises(self):
        with pytest.raises(ValueError, match="empty identifier"):
            parse_task_list(":5")

    def test_whitespace_handling(self):
        assert parse_task_list("  task : 3  ") == [("task", 3)]


# ── execution_order ──────────────────────────────────────────────

class TestExecutionOrder:
    def test_respects_deps(self):
        tasks = [
            Task("a", 0),
            Task("b", 0, dependencies=frozenset(["a"])),
            Task("c", 0, dependencies=frozenset(["b"])),
        ]
        assert execution_order(tasks) == ["a", "b", "c"]

    def test_respects_priority(self):
        tasks = [Task("a", 2), Task("b", 1)]
        assert execution_order(tasks) == ["b", "a"]

    def test_circular_raises(self):
        tasks = [
            Task("a", 0, dependencies=frozenset(["b"])),
            Task("b", 0, dependencies=frozenset(["a"])),
        ]
        with pytest.raises(ValueError, match="circular"):
            execution_order(tasks)

    def test_empty_input(self):
        assert execution_order([]) == []

    def test_priority_with_deps(self):
        tasks = [
            Task("a", 5),
            Task("b", 0, dependencies=frozenset(["a"])),
        ]
        assert execution_order(tasks) == ["a", "b"]
