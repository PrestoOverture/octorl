"""Tests for metric_aggregator."""
import math

import pytest

from metric_aggregator import (
    AlertRule,
    Metric,
    MetricRegistry,
    WindowedMetric,
    bucket_histogram,
    ewma,
    rate,
)


# ── Metric ──────────────────────────────────────────────────────────

class TestMetricAddAndCount:
    def test_empty(self):
        m = Metric()
        assert m.count() == 0
        assert len(m) == 0

    def test_add_increments(self):
        m = Metric()
        m.add(1.0)
        m.add(2.0)
        assert m.count() == 2

    def test_init_from_iterable(self):
        m = Metric([3, 1, 2])
        assert m.count() == 3
        assert m.values() == (1, 2, 3)

    def test_add_rejects_inf(self):
        m = Metric()
        with pytest.raises(ValueError):
            m.add(float("inf"))

    def test_add_rejects_nan(self):
        m = Metric()
        with pytest.raises(ValueError):
            m.add(float("nan"))


class TestMetricAggregation:
    def test_total(self):
        assert Metric([1.0, 2.0, 3.0]).total() == 6.0

    def test_mean(self):
        assert Metric([1.0, 2.0, 3.0]).mean() == 2.0

    def test_mean_empty_raises(self):
        with pytest.raises(ValueError, match="empty"):
            Metric().mean()

    def test_minimum(self):
        assert Metric([3.0, 1.0, 2.0]).minimum() == 1.0

    def test_maximum(self):
        assert Metric([3.0, 1.0, 2.0]).maximum() == 3.0

    def test_min_max_empty_raises(self):
        m = Metric()
        with pytest.raises(ValueError):
            m.minimum()
        with pytest.raises(ValueError):
            m.maximum()


class TestPercentile:
    def test_p0_p100(self):
        m = Metric([10, 20, 30, 40, 50])
        assert m.percentile(0) == 10
        assert m.percentile(100) == 50

    def test_p50(self):
        m = Metric([10, 20, 30, 40, 50])
        assert m.percentile(50) == 30

    def test_interpolation(self):
        m = Metric([10, 20, 30, 40])
        assert abs(m.percentile(25) - 17.5) < 1e-9

    def test_out_of_range(self):
        m = Metric([1.0])
        with pytest.raises(ValueError, match="between 0 and 100"):
            m.percentile(-1)
        with pytest.raises(ValueError):
            m.percentile(101)

    def test_empty_raises(self):
        with pytest.raises(ValueError, match="empty"):
            Metric().percentile(50)

    def test_single_value(self):
        m = Metric([42.0])
        assert m.percentile(0) == 42.0
        assert m.percentile(50) == 42.0
        assert m.percentile(100) == 42.0

    def test_two_values(self):
        m = Metric([0.0, 100.0])
        assert abs(m.percentile(50) - 50.0) < 1e-9

    def test_median_delegates(self):
        m = Metric([1, 2, 3, 4, 5])
        assert m.median() == m.percentile(50)


class TestVariance:
    def test_sample_variance(self):
        m = Metric([2, 4, 4, 4, 5, 5, 7, 9])
        assert abs(m.variance() - 4.571428571428571) < 1e-9

    def test_stddev(self):
        m = Metric([2, 4, 4, 4, 5, 5, 7, 9])
        assert abs(m.stddev() - math.sqrt(m.variance())) < 1e-9

    def test_too_few_raises(self):
        with pytest.raises(ValueError, match="at least 2"):
            Metric([1.0]).variance()

    def test_identical_values(self):
        m = Metric([5.0, 5.0, 5.0])
        assert m.variance() == 0.0


class TestMetricOps:
    def test_merge(self):
        c = Metric([1, 3]).merge(Metric([2, 4]))
        assert c.values() == (1, 2, 3, 4)

    def test_trim(self):
        assert Metric([1, 2, 3, 4, 5]).trim(2, 4).values() == (2, 3, 4)

    def test_trim_invalid_bounds(self):
        with pytest.raises(ValueError):
            Metric([1, 2]).trim(5, 1)

    def test_reset(self):
        m = Metric([1, 2, 3])
        assert m.reset() == 3
        assert m.count() == 0

    def test_summary_empty(self):
        assert Metric().summary() == {"count": 0}

    def test_summary_populated(self):
        s = Metric([1, 2, 3, 4, 5]).summary()
        assert s["count"] == 5
        assert s["mean"] == 3.0
        assert s["min"] == 1.0
        assert s["max"] == 5.0
        assert "p50" in s and "p95" in s and "p99" in s


# ── WindowedMetric ──────────────────────────────────────────────────

class TestWindowedMetric:
    def test_within_window(self):
        w = WindowedMetric(5)
        for i in range(5):
            w.record(float(i))
        assert w.count() == 5

    def test_expiry(self):
        w = WindowedMetric(3)
        for i in range(10):
            w.record(float(i))
        assert w.count() == 3

    def test_snapshot_values(self):
        w = WindowedMetric(3)
        for i in range(5):
            w.record(float(i))
        assert w.snapshot().values() == (2.0, 3.0, 4.0)

    def test_mean(self):
        w = WindowedMetric(3)
        w.record(1.0)
        w.record(2.0)
        w.record(3.0)
        assert w.mean() == 2.0

    def test_invalid_window(self):
        with pytest.raises(ValueError, match="positive"):
            WindowedMetric(0)
        with pytest.raises(ValueError):
            WindowedMetric(-1)

    def test_rejects_non_finite(self):
        w = WindowedMetric(5)
        with pytest.raises(ValueError):
            w.record(float("inf"))


# ── AlertRule ───────────────────────────────────────────────────────

class TestAlertRule:
    def test_gt_fires(self):
        assert AlertRule("h", 10.0, "gt").evaluate(11.0) is True

    def test_gt_no_fire(self):
        assert AlertRule("h", 10.0, "gt").evaluate(9.0) is False

    def test_lt_fires(self):
        assert AlertRule("l", 5.0, "lt").evaluate(3.0) is True

    def test_gte_boundary(self):
        assert AlertRule("h", 10.0, "gte").evaluate(10.0) is True

    def test_lte_boundary(self):
        assert AlertRule("l", 5.0, "lte").evaluate(5.0) is True

    def test_consecutive_threshold(self):
        r = AlertRule("s", 100.0, "gt", consecutive=3)
        assert r.evaluate(200.0) is False
        assert r.evaluate(200.0) is False
        assert r.evaluate(200.0) is True

    def test_streak_resets(self):
        r = AlertRule("s", 100.0, "gt", consecutive=3)
        r.evaluate(200.0)
        r.evaluate(200.0)
        r.evaluate(50.0)
        assert r.evaluate(200.0) is False

    def test_reset(self):
        r = AlertRule("h", 10.0, "gt")
        r.evaluate(20.0)
        r.reset()
        assert r._streak == 0

    def test_invalid_name(self):
        with pytest.raises(ValueError, match="name"):
            AlertRule("", 10.0)

    def test_invalid_comparator(self):
        with pytest.raises(ValueError, match="comparator"):
            AlertRule("x", 10.0, "eq")

    def test_invalid_consecutive(self):
        with pytest.raises(ValueError):
            AlertRule("x", 10.0, consecutive=0)


# ── MetricRegistry ──────────────────────────────────────────────────

class TestMetricRegistry:
    def test_register_get(self):
        reg = MetricRegistry()
        m = reg.register("lat")
        assert reg.get("lat") is m

    def test_duplicate_raises(self):
        reg = MetricRegistry()
        reg.register("x")
        with pytest.raises(ValueError):
            reg.register("x")

    def test_empty_name_raises(self):
        with pytest.raises(ValueError):
            MetricRegistry().register("")

    def test_names_sorted(self):
        reg = MetricRegistry()
        reg.register("b")
        reg.register("a")
        assert reg.names() == ("a", "b")

    def test_record_and_summary(self):
        reg = MetricRegistry()
        reg.register("m")
        reg.record("m", 1.0)
        reg.record("m", 2.0)
        assert reg.summary()["m"]["count"] == 2

    def test_remove(self):
        reg = MetricRegistry()
        reg.register("m")
        assert isinstance(reg.remove("m"), Metric)
        with pytest.raises(KeyError):
            reg.get("m")

    def test_get_missing(self):
        with pytest.raises(KeyError):
            MetricRegistry().get("nope")


# ── Top-level functions ─────────────────────────────────────────────

class TestRate:
    def test_basic(self):
        assert rate([10, 20, 30], 10.0) == 6.0

    def test_zero_interval(self):
        with pytest.raises(ValueError, match="positive"):
            rate([1], 0)

    def test_negative_interval(self):
        with pytest.raises(ValueError):
            rate([1], -1.0)

    def test_empty_values(self):
        assert rate([], 5.0) == 0.0


class TestBucketHistogram:
    def test_basic(self):
        assert bucket_histogram([1, 5, 10, 15, 20], [5, 10, 15]) == [1, 1, 1, 2]

    def test_on_boundary(self):
        assert bucket_histogram([5, 10], [5, 10]) == [0, 1, 1]

    def test_empty_values(self):
        assert bucket_histogram([], [1, 2, 3]) == [0, 0, 0, 0]

    def test_no_boundaries(self):
        assert bucket_histogram([1, 2, 3], []) == [3]


class TestEwma:
    def test_alpha_one_passthrough(self):
        assert ewma([1, 2, 3, 4], 1.0) == [1, 2, 3, 4]

    def test_constant_input(self):
        result = ewma([10, 10, 10, 10], 0.5)
        for v in result:
            assert abs(v - 10.0) < 1e-9

    def test_alpha_zero_raises(self):
        with pytest.raises(ValueError, match="alpha"):
            ewma([1, 2], 0.0)

    def test_alpha_above_one_raises(self):
        with pytest.raises(ValueError):
            ewma([1, 2], 1.5)

    def test_alpha_negative_raises(self):
        with pytest.raises(ValueError):
            ewma([1, 2], -0.1)

    def test_alpha_one_valid(self):
        assert ewma([5, 10], 1.0) == [5, 10]

    def test_empty(self):
        assert ewma([], 0.5) == []

    def test_single(self):
        assert ewma([42], 0.3) == [42]
