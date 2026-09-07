import numpy as np
from src.sampling.common import f, G


def est_plugin(K):
    """B1: plug-in estimator f(mean(K/G)).

    Carries the Jensen aggregation bias (overestimates q on heterogeneous
    buckets) minus a finite-sample concavity penalty (underestimates on
    homogeneous ones).  Net sign is not fixed.
    """
    K = np.asarray(K, dtype=float)
    return float(f(K.mean() / G))
