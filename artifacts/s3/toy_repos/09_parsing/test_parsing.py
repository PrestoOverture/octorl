import pytest
from parsing import *


@pytest.mark.parametrize("text, default, expected", [("12",0,12),(" -3 ",0,-3),("bad",7,7),("",9,9),(None,4,4),("2.5",8,8)])
def test_target(text, default, expected):
    assert parse_integer(text, default) == pytest.approx(expected)


def test_regression():
    assert parse_boolean(" YES ")
    assert split_fields("a, b") == ["a", "b"]
    assert required_text(" x ") == "x"
    assert join_fields([1,2]) == "1,2"
