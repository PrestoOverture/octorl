"""S2a: verify B1 / B_emp / B2 code paths produce numbers on synthetic inputs.

Tests cover:
  - All three estimators return finite floats on normal input
  - Degenerate cases (all-zero, all-success, all-mixed) don't crash
  - Known-distribution Monte Carlo: B_emp is unbiased under a fixed policy
  - B1 underestimates on a homogeneous bucket (the sign-flip from estimator_study.py §2)
  - B2 degenerates to B1 when heterogeneity is undetectable
"""

import numpy as np
import pytest
from src.sampling import est_plugin, est_empirical, est_betabinom, f, Ef_beta, G
from src.sampling.common import beta_params


SEED = 42_000


class TestEstimatorsProduceNumbers:
    """Every estimator returns a finite float in [0, 1] on reasonable input."""

    @pytest.fixture
    def normal_K(self):
        rng = np.random.default_rng(SEED)
        return rng.binomial(G, 0.3, size=32)

    @pytest.mark.parametrize("est", [est_plugin, est_empirical, est_betabinom])
    def test_returns_finite_float(self, est, normal_K):
        result = est(normal_K)
        assert isinstance(result, float)
        assert np.isfinite(result)
        assert 0.0 <= result <= 1.0


class TestDegenerateCases:
    """Estimators handle edge cases without crashing."""

    @pytest.mark.parametrize("est", [est_plugin, est_empirical, est_betabinom])
    def test_all_zero(self, est):
        K = np.zeros(32, dtype=int)
        result = est(K)
        assert isinstance(result, float)
        assert result == pytest.approx(0.0, abs=1e-10)

    @pytest.mark.parametrize("est", [est_plugin, est_empirical, est_betabinom])
    def test_all_success(self, est):
        K = np.full(32, G, dtype=int)
        result = est(K)
        assert isinstance(result, float)
        assert result == pytest.approx(0.0, abs=1e-10)

    @pytest.mark.parametrize("est", [est_plugin, est_empirical, est_betabinom])
    def test_all_mixed(self, est):
        K = np.full(32, G // 2, dtype=int)
        result = est(K)
        assert isinstance(result, float)
        assert result > 0.5

    @pytest.mark.parametrize("est", [est_plugin, est_empirical, est_betabinom])
    def test_single_group(self, est):
        K = np.array([3])
        result = est(K)
        assert isinstance(result, float)
        assert np.isfinite(result)


class TestBempUnbiased:
    """B_emp is unbiased for E[f(p_i)] under a fixed policy (Monte Carlo)."""

    @pytest.mark.parametrize(
        "mu,sd",
        [(0.5, 0.15), (0.3, 0.10), (0.1, 0.05)],
    )
    def test_bemp_unbiased_beta(self, mu, sd):
        rng = np.random.default_rng(SEED + 1)
        a, b = beta_params(mu, sd)
        truth = float(Ef_beta(a, b))
        M = 32
        trials = 5000
        estimates = np.empty(trials)
        for t in range(trials):
            p = rng.beta(a, b, M)
            K = rng.binomial(G, p)
            estimates[t] = est_empirical(K)
        bias = estimates.mean() - truth
        se = estimates.std() / np.sqrt(trials)
        assert abs(bias) < 3 * se, f"B_emp bias {bias:.5f} exceeds 3 SE ({3*se:.5f})"

    def test_bemp_unbiased_homogeneous(self):
        """On a homogeneous bucket, B_emp estimates f(p) unbiasedly."""
        rng = np.random.default_rng(SEED + 2)
        p = 0.3
        truth = float(f(p))
        M = 32
        trials = 5000
        estimates = np.empty(trials)
        for t in range(trials):
            K = rng.binomial(G, p, M)
            estimates[t] = est_empirical(K)
        bias = estimates.mean() - truth
        se = estimates.std() / np.sqrt(trials)
        assert abs(bias) < 3 * se


class TestB1BiasSign:
    """B1 underestimates on homogeneous buckets (no aggregation gap, only
    the finite-sample concavity penalty remains)."""

    def test_b1_underestimates_homogeneous(self):
        rng = np.random.default_rng(SEED + 3)
        p = 0.3
        truth = float(f(p))
        M = 32
        trials = 10000
        estimates = np.empty(trials)
        for t in range(trials):
            K = rng.binomial(G, p, M)
            estimates[t] = est_plugin(K)
        mean_est = estimates.mean()
        assert mean_est < truth, (
            f"B1 should underestimate on homogeneous bucket: "
            f"E[B1]={mean_est:.5f} vs truth={truth:.5f}"
        )


class TestB2FallbackToB1:
    """B2 degenerates to B1 when heterogeneity is undetectable."""

    def test_homogeneous_fallback(self):
        rng = np.random.default_rng(SEED + 4)
        p = 0.5
        M = 32
        K = rng.binomial(G, p, M)
        b1 = est_plugin(K)
        b2 = est_betabinom(K)
        assert b2 == pytest.approx(b1, abs=1e-10), (
            "B2 should fall back to B1 on a homogeneous draw"
        )


class TestClosedForms:
    """Verify f() and Ef_beta() against each other."""

    def test_f_at_zero(self):
        assert f(0.0) == pytest.approx(0.0)

    def test_f_at_one(self):
        assert f(1.0) == pytest.approx(0.0)

    def test_f_at_half(self):
        expected = 1.0 - 0.5**G - 0.5**G
        assert f(0.5) == pytest.approx(expected)

    def test_ef_beta_concentrated(self):
        """A very concentrated Beta(a,b) around mu should give Ef ≈ f(mu)."""
        mu, sd = 0.4, 0.01
        a, b = beta_params(mu, sd)
        assert Ef_beta(a, b) == pytest.approx(float(f(mu)), abs=0.005)

    def test_ef_beta_vs_montecarlo(self):
        """Ef_beta matches Monte Carlo on a moderately spread Beta."""
        mu, sd = 0.3, 0.10
        a, b = beta_params(mu, sd)
        rng = np.random.default_rng(SEED + 5)
        samples = rng.beta(a, b, 200_000)
        mc = float(np.mean(f(samples)))
        closed = float(Ef_beta(a, b))
        assert closed == pytest.approx(mc, abs=0.002)
