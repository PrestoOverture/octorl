def weighted_score(values, weights):
    """Compute the weighted mean of matching nonempty lists.

    Weights must be nonnegative with a positive total.
    The result is normalized by total weight.
    """
    if not values or len(values) != len(weights):
        raise ValueError("mismatched scores")
    if any(weight < 0 for weight in weights):
        raise ValueError("negative weight")
    total_weight = sum(weights)
    if total_weight <= 0:
        raise ValueError("zero total weight")
    return sum(value * weight for value, weight in zip(values, weights)) / total_weight


def mean_score(values):
    """Compute the arithmetic mean of nonempty scores.

    Every score has equal weight.
    An empty list is invalid.
    """
    if not values:
        raise ValueError("no values")
    return sum(values) / len(values)


def passed(value, threshold=60):
    """Test a score against an inclusive pass threshold.

    Scores are not rounded before comparison.
    Exactly the threshold is a pass.
    """
    return value >= threshold


def best_score(values):
    """Find the highest score in a nonempty sequence.

    Ties have no special treatment.
    Empty sequences are invalid.
    """
    if not values:
        raise ValueError("no values")
    return max(values)


def normalize_score(value, maximum):
    """Convert a score to a percentage.

    Maximum must be positive.
    Extra-credit scores may exceed one hundred.
    """
    if maximum <= 0:
        raise ValueError("invalid maximum")
    return value / maximum * 100
