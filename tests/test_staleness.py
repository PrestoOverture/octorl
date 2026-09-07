"""S2a synthetic controls for the staleness monitor.

Two tests required by the roadmap:
  1. Zero-drift input → staleness cost within ±1 SE of 0
  2. Injected known staleness → recovered (positive staleness cost)

These are the "synthetic control" that must pass before any real staleness
number is trusted.  S2a must NOT produce any staleness number on real data.
"""

import numpy as np
import pytest
from src.sampling.empirical import est_empirical
from src.sampling.common import G, beta_params
from src.diagnostics.staleness import StalenessMonitor, is_mixed


SEED = 55_000


class TestIsMixed:
    def test_boundaries(self):
        assert not is_mixed(np.array([0]))[0]
        assert not is_mixed(np.array([G]))[0]
        assert is_mixed(np.array([1]))[0]
        assert is_mixed(np.array([G - 1]))[0]

    def test_vectorized(self):
        K = np.array([0, 1, 4, 7, 8])
        result = is_mixed(K)
        np.testing.assert_array_equal(result, [False, True, True, True, False])


class TestStalenessMonitorBasic:
    """Monitor records data and computes Brier scores."""

    def test_record_and_brier(self):
        mon = StalenessMonitor(n_buckets=2)
        rng = np.random.default_rng(SEED)
        bucket_ids = np.array([0] * 10 + [1] * 10)
        for step in range(2):
            K = rng.binomial(G, 0.3, 20)
            mon.record(step, bucket_ids, K, est_empirical)
        b0 = mon.brier_at_lag(0)
        assert isinstance(b0, float)
        assert np.isfinite(b0)

    def test_staleness_cost_at_lag_zero(self):
        """Staleness(0) is always 0 by definition."""
        mon = StalenessMonitor(n_buckets=1)
        rng = np.random.default_rng(SEED + 1)
        for step in range(5):
            K = rng.binomial(G, 0.3, 32)
            mon.record(step, np.zeros(32, dtype=int), K, est_empirical)
        cost = mon.staleness_cost(0)
        assert cost == pytest.approx(0.0, abs=1e-15)


class TestZeroDriftControl:
    """Under a stationary policy, staleness cost should be within ±1 SE of 0.

    This is the synthetic control that must pass before trusting any real
    staleness measurement.  We run multiple independent trials and check
    that the mean staleness is within ±1 SE.
    """

    def test_zero_drift_bemp(self):
        rng = np.random.default_rng(SEED + 10)
        n_buckets = 4
        groups_per_bucket = 8
        n_groups = n_buckets * groups_per_bucket
        n_steps = 40
        L = 8
        n_trials = 200

        a, b = beta_params(0.3, 0.10)
        p_per_bucket = [rng.beta(a, b) for _ in range(n_buckets)]

        costs = []
        for _ in range(n_trials):
            mon = StalenessMonitor(n_buckets=n_buckets)
            for step in range(n_steps):
                bucket_ids = np.repeat(np.arange(n_buckets), groups_per_bucket)
                K = np.array([
                    rng.binomial(G, p_per_bucket[bid])
                    for bid in bucket_ids
                ])
                mon.record(step, bucket_ids, K, est_empirical)
            cost = mon.staleness_cost(L)
            if np.isfinite(cost):
                costs.append(cost)

        costs = np.array(costs)
        mean_cost = costs.mean()
        se = mon.staleness_se(n_groups * (n_steps - L))

        assert abs(mean_cost) < se, (
            f"Zero-drift staleness cost {mean_cost:.6f} exceeds ±1 SE ({se:.6f}). "
            f"The synthetic control has failed."
        )


class TestInjectedStaleness:
    """When the true success rates shift mid-training, the staleness monitor
    should detect it (positive staleness cost, significantly above zero)."""

    def test_sudden_shift_detected(self):
        rng = np.random.default_rng(SEED + 20)
        n_buckets = 4
        groups_per_bucket = 32
        n_steps = 60
        shift_step = 30
        L = 10
        n_trials = 200

        p_before = [0.02, 0.05, 0.03, 0.04]
        p_after = [0.50, 0.50, 0.50, 0.50]

        costs = []
        for _ in range(n_trials):
            mon = StalenessMonitor(n_buckets=n_buckets)
            for step in range(n_steps):
                p_current = p_before if step < shift_step else p_after
                bucket_ids = np.repeat(np.arange(n_buckets), groups_per_bucket)
                K = np.array([
                    rng.binomial(G, p_current[bid])
                    for bid in bucket_ids
                ])
                mon.record(step, bucket_ids, K, est_empirical)
            cost = mon.staleness_cost(L)
            if np.isfinite(cost):
                costs.append(cost)

        costs = np.array(costs)
        mean_cost = costs.mean()
        se_mean = costs.std() / np.sqrt(len(costs))

        assert mean_cost > 0, (
            f"Injected staleness not detected: mean cost {mean_cost:.6f} <= 0"
        )
        assert mean_cost > 3 * se_mean, (
            f"Injected staleness too weak: mean cost {mean_cost:.6f} "
            f"not significantly above 3 SE_mean ({3*se_mean:.6f})"
        )

    def test_gradual_drift_detected(self):
        """A gradual drift (linear increase in p) should also be detected."""
        rng = np.random.default_rng(SEED + 30)
        n_buckets = 2
        groups_per_bucket = 16
        n_groups = n_buckets * groups_per_bucket
        n_steps = 50
        L = 10
        n_trials = 200

        costs = []
        for _ in range(n_trials):
            mon = StalenessMonitor(n_buckets=n_buckets)
            for step in range(n_steps):
                drift = 0.01 * step
                p_current = [min(0.2 + drift, 0.95), min(0.3 + drift, 0.95)]
                bucket_ids = np.repeat(np.arange(n_buckets), groups_per_bucket)
                K = np.array([
                    rng.binomial(G, p_current[bid])
                    for bid in bucket_ids
                ])
                mon.record(step, bucket_ids, K, est_empirical)
            cost = mon.staleness_cost(L)
            if np.isfinite(cost):
                costs.append(cost)

        costs = np.array(costs)
        mean_cost = costs.mean()

        assert mean_cost > 0, (
            f"Gradual drift not detected: mean cost {mean_cost:.6f} <= 0"
        )


class TestStalenessMonitorSE:
    """SE formula matches the documented approximation."""

    def test_se_at_25600_groups(self):
        mon = StalenessMonitor(n_buckets=1)
        se = mon.staleness_se(25600)
        assert se == pytest.approx(1.5e-3, rel=1e-6)

    def test_se_scales_with_groups(self):
        mon = StalenessMonitor(n_buckets=1)
        se_large = mon.staleness_se(25600)
        se_small = mon.staleness_se(6400)
        assert se_small == pytest.approx(se_large * 2, rel=1e-6)
