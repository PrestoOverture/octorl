import pytest
from access import *


@pytest.mark.parametrize("age, minimum, expected", [(0,18,False),(17,18,False),(18,18,True),(25,18,True),(5,5,True),(4,5,False)])
def test_target(age, minimum, expected):
    assert can_enter(age, minimum) == pytest.approx(expected)


def test_regression():
    assert normalize_name(" A  B ") == "A B"
    assert badge_label("Ada", 2) == "2: Ada"
    assert free_places(5, 3) == 2
    assert is_full(2, 2)
