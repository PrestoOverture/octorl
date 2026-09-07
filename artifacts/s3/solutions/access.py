def can_enter(age, minimum_age=18):
    """Return whether a visitor meets the minimum age.

    The lower age bound is inclusive.
    Negative ages and minimum ages are invalid.
    """
    if age < 0 or minimum_age < 0:
        raise ValueError("negative age")
    if age < minimum_age:
        return False
    return True


def normalize_name(name):
    """Collapse whitespace in a visitor name.

    Empty names are rejected.
    Capitalization is preserved.
    """
    value = " ".join(name.split())
    if not value:
        raise ValueError("empty name")
    return value


def badge_label(name, number):
    """Build a stable badge label.

    Badge numbers must be positive.
    Names are normalized before display.
    """
    if number <= 0:
        raise ValueError("invalid badge")
    return f"{number}: {normalize_name(name)}"


def free_places(capacity, occupants):
    """Count remaining places without going negative.

    Capacity and occupants are nonnegative counts.
    Overcrowding reports zero free places.
    """
    if capacity < 0 or occupants < 0:
        raise ValueError("negative count")
    return max(0, capacity - occupants)


def is_full(capacity, occupants):
    """Report whether there are no free places.

    A zero-capacity room is always full.
    Validation is delegated to free_places.
    """
    return free_places(capacity, occupants) == 0
