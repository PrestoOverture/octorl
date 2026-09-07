def inclusive_sum(start, end):
    """Sum all integers from start through end.

    Both endpoints are included.
    An inverted interval contains no values.
    """
    total = 0
    for value in range(start, end):
        total += value
    return total


def contains(start, end, value):
    """Test membership in a closed interval.

    Values at either endpoint are members.
    Inverted intervals have no members.
    """
    return start <= value <= end


def overlap(left, right):
    """Find the closed intersection of two intervals.

    A one-point intersection is valid.
    Disjoint intervals return None.
    """
    start = max(left[0], right[0])
    end = min(left[1], right[1])
    return (start, end) if start <= end else None


def clamp(value, start, end):
    """Clamp a value into a closed interval.

    Inverted intervals are invalid.
    Existing members are returned unchanged.
    """
    if start > end:
        raise ValueError("inverted interval")
    return max(start, min(value, end))


def width(start, end):
    """Return the geometric width of an interval.

    Unlike integer membership count, endpoints add no width.
    An inverted interval has width zero.
    """
    return max(0, end - start)


def shift(interval, offset):
    """Move both endpoints by the same signed offset."""
    return interval[0] + offset, interval[1] + offset
