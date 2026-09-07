import pytest
from temperature import *


@pytest.mark.parametrize("value, expected", [(0,32),(100,212),(-40,-40),(10,50),(25,77),(-10,14)])
def test_target(value, expected):
    assert celsius_to_fahrenheit(value) == pytest.approx(expected)


def test_regression():
    assert fahrenheit_to_celsius(32) == 0
    assert celsius_to_kelvin(0) == 273.15
    assert is_freezing(0)
    assert average_temperature([10,20]) == 15
    assert label(2, "C") == "2.0 C"
