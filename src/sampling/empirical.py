import numpy as np
from src.sampling.common import G


def est_empirical(K):
    """B_emp: observed mixed-group fraction.

    Unbiased for E[f(p_i)] under a fixed policy.  Under an online sliding
    window its expectation is the history-average (1/W) sum q(theta_s), not
    the current q(theta_t).
    """
    K = np.asarray(K, dtype=float)
    return float(np.mean((K > 0) & (K < G)))
