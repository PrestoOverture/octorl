import pytest
from scores import *


@pytest.mark.parametrize("values, weights, expected", [([10,20],[1,3],17.5),([5],[2],5),([0,10],[0,2],10),([2,8],[1,1],5),([3,9],[2,4],7),([1,2,3],[2,2,2],2)])
def test_target(values, weights, expected):
    assert weighted_score(values, weights) == pytest.approx(expected)


def test_regression():
    assert mean_score([2,4]) == 3
    assert passed(60)
    assert best_score([2,3]) == 3
    assert normalize_score(5,10) == 50
