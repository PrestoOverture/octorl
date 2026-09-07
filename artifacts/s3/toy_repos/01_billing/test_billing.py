import pytest
from billing import *


@pytest.mark.parametrize("price, percentage, expected", [(100, 0, 100), (100, 10, 90), (100, 50, 50), (100, 75, 50), (80, 100, 40), (0, 70, 0)])
def test_target(price, percentage, expected):
    assert calculate_discount(price, percentage) == pytest.approx(expected)


def test_regression():
    assert subtotal([2, 3]) == 5
    assert add_tax(10, .1) == 11
    assert split_bill(9, 3) == 3
    assert format_money(2) == "2.00"
