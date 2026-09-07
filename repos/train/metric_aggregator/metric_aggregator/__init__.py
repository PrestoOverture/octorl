"""Time-windowed metric aggregation with percentiles and alerts."""
from __future__ import annotations

import bisect
import math
from collections import deque
from collections.abc import Iterable


class Metric:
    __slots__ = ("_values",)

    def __init__(self, values: Iterable[float] = ()) -> None:
        self._values = sorted(values)

    def __len__(self) -> int:
        return len(self._values)

    def add(self, value: float) -> None:
        if not math.isfinite(value):
            raise ValueError("only finite values accepted")
        bisect.insort(self._values, value)

    def count(self) -> int:
        return len(self._values)

    def total(self) -> float:
        return sum(self._values)

    def mean(self) -> float:
        if not self._values:
            raise ValueError("empty metric")
        return self.total() / self.count()

    def minimum(self) -> float:
        if not self._values:
            raise ValueError("empty metric")
        return self._values[0]

    def maximum(self) -> float:
        if not self._values:
            raise ValueError("empty metric")
        return self._values[-1]

    def percentile(self, p: float) -> float:
        if not 0 <= p <= 100:
            raise ValueError("percentile must be between 0 and 100")
        if not self._values:
            raise ValueError("empty metric")
        if p == 0:
            return self._values[0]
        if p == 100:
            return self._values[-1]
        rank = (p / 100) * (len(self._values) - 1)
        lower = int(rank)
        fraction = rank - lower
        if lower + 1 >= len(self._values):
            return self._values[-1]
        return self._values[lower] + fraction * (self._values[lower + 1] - self._values[lower])

    def median(self) -> float:
        return self.percentile(50)

    def variance(self) -> float:
        if len(self._values) < 2:
            raise ValueError("need at least 2 values")
        m = self.mean()
        return sum((v - m) ** 2 for v in self._values) / (len(self._values) - 1)

    def stddev(self) -> float:
        return math.sqrt(self.variance())

    def values(self) -> tuple[float, ...]:
        return tuple(self._values)

    def merge(self, other: Metric) -> Metric:
        combined = list(self._values) + list(other._values)
        return Metric(combined)

    def trim(self, lower: float, upper: float) -> Metric:
        if lower > upper:
            raise ValueError("lower must not exceed upper")
        return Metric(v for v in self._values if lower <= v <= upper)

    def reset(self) -> int:
        count = len(self._values)
        self._values.clear()
        return count

    def summary(self) -> dict[str, float]:
        if not self._values:
            return {"count": 0}
        return {
            "count": self.count(),
            "mean": self.mean(),
            "min": self.minimum(),
            "max": self.maximum(),
            "p50": self.percentile(50),
            "p95": self.percentile(95),
            "p99": self.percentile(99),
        }


class WindowedMetric:

    def __init__(self, window_size: int) -> None:
        if window_size <= 0:
            raise ValueError("positive window size required")
        self._window = window_size
        self._buffer: deque[tuple[int, float]] = deque()
        self._tick = 0

    def record(self, value: float) -> None:
        if not math.isfinite(value):
            raise ValueError("only finite values accepted")
        self._tick += 1
        self._buffer.append((self._tick, value))
        self._expire()

    def _expire(self) -> None:
        cutoff = self._tick - self._window
        while self._buffer and self._buffer[0][0] <= cutoff:
            self._buffer.popleft()

    def snapshot(self) -> Metric:
        self._expire()
        return Metric(v for _, v in self._buffer)

    def count(self) -> int:
        self._expire()
        return len(self._buffer)

    def mean(self) -> float:
        return self.snapshot().mean()


class AlertRule:

    def __init__(self, name: str, threshold: float, comparator: str = "gt",
                 consecutive: int = 1) -> None:
        if not name:
            raise ValueError("alert name required")
        if comparator not in ("gt", "lt", "gte", "lte"):
            raise ValueError("comparator must be gt, lt, gte, or lte")
        if consecutive < 1:
            raise ValueError("consecutive must be positive")
        self.name = name
        self.threshold = threshold
        self.comparator = comparator
        self.consecutive = consecutive
        self._streak = 0

    def _check(self, value: float) -> bool:
        if self.comparator == "gt":
            return value > self.threshold
        if self.comparator == "lt":
            return value < self.threshold
        if self.comparator == "gte":
            return value >= self.threshold
        return value <= self.threshold

    def evaluate(self, value: float) -> bool:
        if self._check(value):
            self._streak += 1
        else:
            self._streak = 0
        return self._streak >= self.consecutive

    def reset(self) -> None:
        self._streak = 0


class MetricRegistry:

    def __init__(self) -> None:
        self._metrics: dict[str, Metric] = {}

    def register(self, name: str) -> Metric:
        if not name or name in self._metrics:
            raise ValueError("new nonempty metric name required")
        metric = Metric()
        self._metrics[name] = metric
        return metric

    def get(self, name: str) -> Metric:
        return self._metrics[name]

    def names(self) -> tuple[str, ...]:
        return tuple(sorted(self._metrics))

    def record(self, name: str, value: float) -> None:
        self._metrics[name].add(value)

    def summary(self) -> dict[str, dict[str, float]]:
        return {name: metric.summary() for name, metric in sorted(self._metrics.items())}

    def remove(self, name: str) -> Metric:
        return self._metrics.pop(name)


def rate(values: Iterable[float], interval: float) -> float:
    total = sum(values)
    if interval <= 0:
        raise ValueError("positive interval required")
    return total / interval


def bucket_histogram(values: Iterable[float], boundaries: Iterable[float]) -> list[int]:
    bounds = sorted(boundaries)
    counts = [0] * (len(bounds) + 1)
    for v in values:
        idx = bisect.bisect_right(bounds, v)
        counts[idx] += 1
    return counts


def ewma(values: Iterable[float], alpha: float) -> list[float]:
    if not 0 < alpha <= 1:
        raise ValueError("alpha must be in (0, 1]")
    result = []
    current = None
    for v in values:
        if current is None:
            current = v
        else:
            current = alpha * v + (1 - alpha) * current
        result.append(current)
    return result
