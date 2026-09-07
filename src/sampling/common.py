import numpy as np
from math import lgamma

G = 8

_gammaln = np.vectorize(lgamma)


def f(p):
    """Probability a GRPO group of G rollouts at success rate p is mixed."""
    p = np.asarray(p, dtype=float)
    return 1.0 - p**G - (1.0 - p) ** G


def Ef_beta(a, b):
    """E[f(p)] for p ~ Beta(a, b), closed form via rising factorials."""
    t1 = _gammaln(a + G) + _gammaln(a + b) - _gammaln(a) - _gammaln(a + b + G)
    t2 = _gammaln(b + G) + _gammaln(a + b) - _gammaln(b) - _gammaln(a + b + G)
    return 1.0 - np.exp(t1) - np.exp(t2)


def beta_params(mu, sd):
    """Moment-match mu, sd to Beta(a, b) parameters."""
    nu = mu * (1 - mu) / sd**2 - 1.0
    return mu * nu, (1 - mu) * nu
