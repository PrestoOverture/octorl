import pytest
from pages import *


@pytest.mark.parametrize("total, size, expected", [(0,5,0),(1,5,1),(5,5,1),(6,5,2),(10,5,2),(11,5,3)])
def test_target(total, size, expected):
    assert page_count(total, size) == pytest.approx(expected)


def test_regression():
    assert page_slice(list(range(6)), 1, 2) == [2, 3]
    assert has_next(6, 0, 2)
    assert item_offset(2, 5) == 10
    assert clamp_page(9, 3) == 2
