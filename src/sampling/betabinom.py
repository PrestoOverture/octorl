import numpy as np
from src.sampling.common import f, G, Ef_beta, beta_params


def est_betabinom(K):
    """B2: Beta-Binomial shrinkage estimator.

    Moment-fits alpha, beta from the noise-corrected variance of per-task
    success rates, then computes E[f(p)] in closed form.  Falls back to the
    plug-in f(mu) when heterogeneity is undetectable or the variance estimate
    is out of range.
    """
    K = np.asarray(K, dtype=float)
    if K.size < 2:
        return float(f(K.mean() / G))
    ph = K / G
    mu = ph.mean()
    if mu <= 0 or mu >= 1:
        return float(f(mu))
    var_p = (ph.var(ddof=1) - mu * (1 - mu) / G) / (1 - 1 / G)
    if var_p <= 0 or var_p >= mu * (1 - mu):
        return float(f(mu))
    a, b = beta_params(mu, np.sqrt(var_p))
    return float(Ef_beta(a, b))
