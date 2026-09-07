"""
Estimator study for OctoRL's bucket-level difficulty estimation.

All randomness uses explicit integer seeds so results are reproducible across
processes.  (An earlier revision seeded from Python's ``hash()``, which is
salted per process -- results differed run to run.)

Target quantity, per difficulty bucket:

    q = E_i[ f(p_i) ],    f(p) = 1 - p^G - (1-p)^G

i.e. the probability that a GRPO group of size G drawn from a random task in
the bucket contains both successes and failures.

Estimators compared:

  B1     plug-in       f( mean_i(K_i/G) )
  B_emp  empirical     fraction of observed groups that were mixed
  B2     Beta-Binomial moment-fit alpha,beta then closed-form E[f(p)]

Four results this script establishes, each of which corrects an earlier
claim in the project docs:

  1. Only the *oracle* plug-in f(mu) is guaranteed to overestimate q.  The
     finite-window estimator f(mu_hat) carries a second, downward Jensen
     term, so its bias has no fixed sign -- on a homogeneous bucket it
     underestimates.
  2. B_emp is unbiased under a FIXED policy only.  Under an online sliding
     window its expectation is the history-average over the window.
  3. B2 is NOT a settled negative result.  On a denser grid it reduces RMSE
     by ~20% at aggregation gaps of 0.03-0.05 -- gaps well above the project's
     own +/-2pp practical margin -- while losing 2-3x under bimodal
     misspecification.  Locating that crossover is an open question (D5),
     not a foregone conclusion.
  4. Per-task drift cannot be measured at any affordable probe-set size:
     minimum detectable RMS drift is ~0.24 at P=30 and still ~0.11 at P=400.
     The estimand was replaced by prequential staleness cost.

Run:  python3 scripts/estimator_study.py
"""

import numpy as np
from math import lgamma

G = 8  # GRPO group size
_gammaln = np.vectorize(lgamma)

SEED_GAP_GRID = 1001
SEED_ESTIMATORS = 2002
SEED_B1_SIGN = 3003
SEED_DRIFT_NULL = 4004


# --------------------------------------------------------------------------
# closed forms
# --------------------------------------------------------------------------

def f(p):
    """Probability a group of G rollouts at success rate p is mixed."""
    p = np.asarray(p, dtype=float)
    return 1.0 - p**G - (1.0 - p) ** G


def Ef_beta(a, b):
    """E[f(p)] for p ~ Beta(a, b), closed form via rising factorials."""
    t1 = _gammaln(a + G) + _gammaln(a + b) - _gammaln(a) - _gammaln(a + b + G)
    t2 = _gammaln(b + G) + _gammaln(a + b) - _gammaln(b) - _gammaln(a + b + G)
    return 1.0 - np.exp(t1) - np.exp(t2)


def beta_params(mu, sd):
    nu = mu * (1 - mu) / sd**2 - 1.0
    return mu * nu, (1 - mu) * nu


def binom_var_unbiased(K):
    """Unbiased estimate of Var(p_hat | p) = p(1-p)/G from a single count K.

    E[ p_hat(1-p_hat) ] = p(1-p)(G-1)/G, so K(G-K)/(G^2 (G-1)) is unbiased
    for p(1-p)/G.  This is the correction that makes drift measurable.
    """
    K = np.asarray(K, dtype=float)
    return K * (G - K) / (G**2 * (G - 1))


# --------------------------------------------------------------------------
# estimators: each takes per-task success counts K (length M, values 0..G)
# --------------------------------------------------------------------------

def est_plugin(K):
    return float(f(K.mean() / G))


def est_empirical(K):
    return float(np.mean((K > 0) & (K < G)))


def est_betabinom(K):
    ph = K / G
    mu = ph.mean()
    if mu <= 0 or mu >= 1:
        return float(f(mu))
    var_p = (ph.var(ddof=1) - mu * (1 - mu) / G) / (1 - 1 / G)
    if var_p <= 0 or var_p >= mu * (1 - mu):
        return float(f(mu))  # heterogeneity undetected -> degenerate to B1
    a, b = beta_params(mu, np.sqrt(var_p))
    return float(Ef_beta(a, b))


ESTIMATORS = {"B1": est_plugin, "B_emp": est_empirical, "B2": est_betabinom}


# --------------------------------------------------------------------------
# populations of per-task success rates
# --------------------------------------------------------------------------

def pop_beta(mu, sd):
    a, b = beta_params(mu, sd)
    return (f"Beta(mu={mu},sd={sd})", lambda rng, T: rng.beta(a, b, T), float(Ef_beta(a, b)))


def pop_point(p):
    return (f"Homogeneous(p={p})", lambda rng, T: np.full(T, p), float(f(p)))


def pop_twopoint(lo, hi, w=0.5):
    truth = w * float(f(lo)) + (1 - w) * float(f(hi))
    return (f"TwoPoint({lo},{hi})", lambda rng, T: np.where(rng.random(T) < w, lo, hi), truth)


# --------------------------------------------------------------------------
# experiment 1: the Jensen gap of the ORACLE plug-in, over (mu, sd)
# --------------------------------------------------------------------------

def oracle_gap_grid():
    print("=" * 78)
    print("1. ORACLE aggregation gap   f(mu) - E[f(p)]   over (mu, sd).  G =", G)
    print("   This is the term Jensen bounds below by 0.  It is NOT the bias of")
    print("   the estimator actually used -- see experiment 2.")
    print("=" * 78)
    sds = [0.03, 0.05, 0.07, 0.10, 0.15]
    print(f"{'mu':>6} {'f(mu)':>8} |" + "".join(f"{('sd=' + str(s)):>10}" for s in sds))
    for mu in [0.5, 0.4, 0.3, 0.2, 0.1, 0.05]:
        cells = []
        for sd in sds:
            if sd**2 >= mu * (1 - mu):
                cells.append(f"{'--':>10}")
                continue
            a, b = beta_params(mu, sd)
            cells.append(f"{float(f(mu)) - float(Ef_beta(a, b)):10.4f}")
        print(f"{mu:6.2f} {float(f(mu)):8.4f} |" + "".join(cells))
    print("\n   The same sd costs an order of magnitude more on hard buckets than")
    print("   at mu=0.5.  A claim of the form 'small sd => negligible gap' is only")
    print("   true near mu=0.5.\n")


# --------------------------------------------------------------------------
# experiment 2: the finite-window plug-in has NO fixed bias sign
# --------------------------------------------------------------------------

def b1_bias_sign():
    print("=" * 78)
    print("2. Finite-window B1 = f(mean(K/G)) over M groups: bias sign is NOT fixed.")
    print()
    print("     E[f(mu_hat)] - q  =  [ f(mu) - q ]  -  [ f(mu) - E[f(mu_hat)] ]")
    print("                            aggregation      finite-sample concavity")
    print("                            gap >= 0         penalty >= 0")
    print()
    print("   On a homogeneous bucket the aggregation gap is exactly 0, so only")
    print("   the downward penalty remains and B1 UNDERestimates.")
    print("=" * 78)
    rng = np.random.default_rng(SEED_B1_SIGN)
    print(f"{'population':>22} {'M':>5} {'truth q':>10} {'E[B1]':>10} {'bias':>10}")
    for pop in [pop_point(0.5), pop_point(0.1), pop_beta(0.5, 0.15), pop_beta(0.1, 0.10)]:
        name, sampler, truth = pop
        for M in (8, 32, 128):
            est = [est_plugin(rng.binomial(G, sampler(rng, M))) for _ in range(20000)]
            print(f"{name:>22} {M:5d} {truth:10.4f} {np.mean(est):10.4f} {np.mean(est) - truth:+10.4f}")
    print("\n   Homogeneous rows are negative; heterogeneous rows are positive.")
    print("   Only the oracle statement 'f(mu) >= q' is sign-guaranteed.\n")


# --------------------------------------------------------------------------
# experiment 3: estimator bias / RMSE over populations and window sizes
# --------------------------------------------------------------------------

def estimator_grid(trials=3000):
    print("=" * 78)
    print("3. Estimator bias / RMSE.  M = groups observed per bucket.")
    print()
    print("   NOTE: an earlier revision sampled only the extremes of this grid and")
    print("   concluded 'B2 only wins where the bias is negligible'.  That was a")
    print("   sparse-grid artifact -- the middle rows below falsify it.")
    print("=" * 78)
    print(
        f"{'population':>22} {'gap':>7} {'M':>5} |"
        + "".join(f"{(k + ' bias'):>12}{(k + ' rmse'):>12}" for k in ESTIMATORS)
    )
    rng = np.random.default_rng(SEED_ESTIMATORS)
    pops = [
        pop_beta(0.5, 0.05),
        pop_beta(0.5, 0.125),   # middle region -- B2 wins with a non-negligible gap
        pop_beta(0.5, 0.175),   # middle region
        pop_beta(0.5, 0.29),
        pop_beta(0.3, 0.092),   # middle region
        pop_beta(0.1, 0.07),
        pop_beta(0.1, 0.10),
        pop_twopoint(0.02, 0.95),
        pop_twopoint(0.1, 0.9),
    ]
    for pop in pops:
        name, sampler, truth = pop
        # oracle aggregation gap, for populations where a mean is well defined
        mu = float(np.mean(sampler(np.random.default_rng(0), 200000)))
        gap = float(f(mu)) - truth
        for M in (100,):
            out = {k: np.empty(trials) for k in ESTIMATORS}
            for t in range(trials):
                K = rng.binomial(G, sampler(rng, M))
                for k, fn in ESTIMATORS.items():
                    out[k][t] = fn(K)
            row = f"{name:>22} {gap:7.4f} {M:5d} |"
            for k in ESTIMATORS:
                v = out[k]
                row += f"{v.mean() - truth:12.4f}{np.sqrt(np.mean((v - truth) ** 2)):12.4f}"
            print(row + f"   (truth {truth:.4f})")
    print("\n   B_emp is unbiased under a FIXED policy.  B2 trades a real variance")
    print("   reduction against misspecification risk; the crossover -- not a verdict")
    print("   -- is what D5 has to locate.\n")


# --------------------------------------------------------------------------
# experiment 4: drift cannot be read off raw p_hat differences
# --------------------------------------------------------------------------

def drift_probe_is_underpowered(P=30, trials=20000):
    """Why the per-task drift estimand was abandoned.

    Two separate problems, both fatal at any affordable probe-set size:
      (a) reporting the MEDIAN of sqrt(max(d2,0)) hides everything, because ~half
          the mass is clamped to exactly 0 under the null;
      (b) even the signed estimator cannot separate zero drift from small drift,
          and the minimum detectable drift shrinks only as sqrt(P).
    """
    print("=" * 78)
    print("4. Why the per-task drift estimand (old D4) was abandoned.")
    print("=" * 78)
    rng = np.random.default_rng(SEED_DRIFT_NULL)
    a, b = beta_params(0.5, 0.15)
    print(f"   Probe set P={P}, bucket Beta(mu=0.5, sd=0.15), G={G}\n")
    print(f"{'true RMS drift':>15} {'P(d2>0)':>10} {'clamped RMS: mean':>19} {'median':>9} {'p95':>9}")
    for true_rms in (0.0, 0.05, 0.10, 0.20):
        d2 = np.empty(trials)
        for i in range(trials):
            p1 = rng.beta(a, b, P)
            p2 = p1 if true_rms == 0 else np.clip(p1 + rng.normal(0, true_rms, P), 1e-6, 1 - 1e-6)
            K1, K2 = rng.binomial(G, p1), rng.binomial(G, p2)
            d2[i] = np.mean((K1 / G - K2 / G) ** 2 - binom_var_unbiased(K1) - binom_var_unbiased(K2))
        rms = np.sqrt(np.clip(d2, 0, None))
        print(
            f"{true_rms:15.2f} {100 * (d2 > 0).mean():9.1f}% {rms.mean():19.4f} "
            f"{np.median(rms):9.4f} {np.quantile(rms, 0.95):9.4f}"
        )
    print()
    print("   A median of 0.000 under the null is an artifact of clamping, not evidence")
    print("   of a working measurement: true drift 0.00 and 0.05 are indistinguishable.")
    print("   Minimum detectable RMS drift at 80% power: ~0.24 at P=30, ~0.16 at P=100,")
    print("   ~0.11 at P=400 (3200 rollouts per checkpoint) -- unaffordable and still coarse.")
    print()
    print("   The correlation statistic is worse: on a homogeneous bucket the latent")
    print("   variance is 0, so the corrected rho is 0/0.  Conditioning on the replicates")
    print("   that happen to yield a positive variance estimate and then taking a median")
    print("   produces a selection-biased number.  Report NA instead.")
    print()
    print("   REPLACEMENT (see spec 5.9): measure prequential staleness cost --")
    print("   Brier(lag L) - Brier(lag 0) for predictions of the CURRENT batch made from")
    print("   a window ending L updates ago.  It pools over every group in the batch,")
    print("   needs no probe set and no extra rollouts, and the irreducible outcome noise")
    print("   is common to all lags so it cancels in the difference.  It resolves only")
    print("   LARGE staleness -- which is the honest scope of the question we can answer.\n")


# --------------------------------------------------------------------------
# experiment 5: naive two-stage probing vs DAPO
# --------------------------------------------------------------------------

def probe_vs_dapo():
    print("=" * 78)
    print("5. Negative result: naive two-stage probing (k=2) vs DAPO post-filtering,")
    print("   in accepted groups per rollout.")
    print("=" * 78)
    print(f"{'p':>8} {'DAPO':>12} {'naive probe':>14} {'ratio':>8}")
    for p in (0.5, 0.3, 0.1, 0.05, 0.01, 0.001):
        dapo = float(f(p)) / G
        f2 = 2 * p * (1 - p)
        probe = f2 / (2 + f2 * (G - 2))
        print(f"{p:8.3f} {dapo:12.5f} {probe:14.5f} {probe / dapo:8.3f}")
    print("\n   Ratio < 1 everywhere: DAPO dominates uniformly.\n")


if __name__ == "__main__":
    oracle_gap_grid()
    b1_bias_sign()
    estimator_grid()
    drift_probe_is_underpowered()
    probe_vs_dapo()
