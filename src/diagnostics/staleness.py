"""Prequential staleness cost: Brier(L) - Brier(0).

Computes from training rollouts with no probe set and no extra rollouts.
The irreducible outcome noise is common to all lags, so it cancels in the
difference.  Coarse by construction: separates large staleness from none.

Reference: architecture.md §4b, prd.md §4.5.7.
"""

import numpy as np
from src.sampling.common import G


def is_mixed(K):
    """Whether each group's success count indicates a mixed group."""
    K = np.asarray(K, dtype=int)
    return (K > 0) & (K < G)


def brier_score(predictions, outcomes):
    """Mean squared error between bucket-level predictions and binary outcomes."""
    predictions = np.asarray(predictions, dtype=float)
    outcomes = np.asarray(outcomes, dtype=float)
    return float(np.mean((predictions - outcomes) ** 2))


class StalenessMonitor:
    """Tracks bucket-level estimates over policy updates and computes
    prequential staleness cost at any lag L.

    Usage:
        mon = StalenessMonitor(n_buckets=4)
        for step in training:
            # bucket_ids: which bucket each group belongs to (length n_groups)
            # K: success counts per group (length n_groups, values 0..G)
            # estimator: callable(K_bucket) -> q_hat
            mon.record(step, bucket_ids, K, estimator)
        cost = mon.staleness_cost(L=8)
    """

    def __init__(self, n_buckets):
        self.n_buckets = n_buckets
        # history[step] = {bucket_id: q_hat} — the estimate after observing this step
        self.history = {}
        # outcomes[step] = list of (bucket_id, y_i) for each group in the batch
        self.outcomes = {}
        self._steps = []

    def record(self, step, bucket_ids, K, estimator):
        """Record one policy-update step.

        Args:
            step: integer step index
            bucket_ids: array of bucket assignments, one per group
            K: array of success counts, one per group (values 0..G)
            estimator: callable(K_array) -> float, the bucket-level estimator
        """
        bucket_ids = np.asarray(bucket_ids, dtype=int)
        K = np.asarray(K, dtype=int)
        outcomes = is_mixed(K)

        self.outcomes[step] = list(zip(bucket_ids.tolist(), outcomes.tolist()))
        self._steps.append(step)

        estimates = {}
        for b in range(self.n_buckets):
            mask = bucket_ids == b
            if mask.any():
                estimates[b] = estimator(K[mask])
        self.history[step] = estimates

    def brier_at_lag(self, L):
        """Compute Brier score using predictions from L steps before each batch.

        Prequential: the lag-0 estimate is the one that existed *before* the
        current batch (i.e. from the previous step), so the prediction is
        always independent of the outcome.  This makes the outcome noise
        cancel in the staleness difference.
        """
        predictions = []
        outcomes = []

        for i, step in enumerate(self._steps):
            pred_idx = i - L - 1
            if pred_idx < 0:
                continue
            pred_step = self._steps[pred_idx]
            pred_estimates = self.history[pred_step]

            for bucket_id, y in self.outcomes[step]:
                if bucket_id in pred_estimates:
                    predictions.append(pred_estimates[bucket_id])
                    outcomes.append(float(y))

        if not predictions:
            return float("nan")
        return brier_score(predictions, outcomes)

    def staleness_cost(self, L):
        """Paired Brier(L) - Brier(0) over the common observation window.

        Computing the difference observation-by-observation guarantees the
        irreducible outcome noise (y^2 term) cancels exactly, regardless of
        whether the policy is changing.  The two lags must both have valid
        predictions for each included step.
        """
        diffs = []
        for i, step in enumerate(self._steps):
            pred0_idx = i - 1
            predL_idx = i - L - 1
            if pred0_idx < 0 or predL_idx < 0:
                continue
            est0 = self.history[self._steps[pred0_idx]]
            estL = self.history[self._steps[predL_idx]]
            for bucket_id, y in self.outcomes[step]:
                if bucket_id in est0 and bucket_id in estL:
                    y_f = float(y)
                    diffs.append((estL[bucket_id] - y_f) ** 2
                                 - (est0[bucket_id] - y_f) ** 2)
        if not diffs:
            return float("nan")
        return float(np.mean(diffs))

    def staleness_se(self, n_groups):
        """Approximate SE of the staleness cost given the number of groups.

        From architecture.md §4b: SE = 1.5e-3 * sqrt(25600 / n_groups).
        """
        return 1.5e-3 * np.sqrt(25600 / n_groups)
