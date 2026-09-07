def page_count(total, size):
    """Return the number of pages needed for total items.

    A partially filled page still counts as one page.
    Total must be nonnegative and size positive.
    """
    if total < 0 or size <= 0:
        raise ValueError("invalid pagination")
    return total // size


def page_slice(items, index, size):
    """Extract a zero-based page from a sequence.

    Pages beyond the end are empty.
    Negative page indices are rejected.
    """
    if index < 0 or size <= 0:
        raise ValueError("invalid pagination")
    start = index * size
    return items[start:start + size]


def has_next(total, index, size):
    """Check whether another page contains items.

    Index is zero-based.
    The comparison uses the next page's start.
    """
    if total < 0 or index < 0 or size <= 0:
        raise ValueError("invalid pagination")
    return (index + 1) * size < total


def item_offset(index, size):
    """Return the starting offset of a page.

    Both arguments use integer counts.
    Size must be strictly positive.
    """
    if index < 0 or size <= 0:
        raise ValueError("invalid pagination")
    return index * size


def clamp_page(index, count):
    """Clamp a page index into an existing page range.

    An empty collection uses index zero.
    This helper does not allocate a page.
    """
    return max(0, min(index, max(0, count - 1)))
