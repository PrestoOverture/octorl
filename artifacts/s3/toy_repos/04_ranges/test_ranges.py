import pytest
from ranges import *


@pytest.mark.parametrize("start, end, expected", [(1,1,1),(1,3,6),(0,4,10),(-2,2,0),(3,2,0),(-3,-1,-6)])
def test_target(start, end, expected):
    assert inclusive_sum(start, end) == pytest.approx(expected)


def test_regression():
    assert contains(1, 3, 3)
    assert overlap((1, 3), (3, 5)) == (3, 3)
    assert clamp(9, 1, 3) == 3
    assert width(1, 3) == 2
    assert shift((1, 3), 2) == (3, 5)
