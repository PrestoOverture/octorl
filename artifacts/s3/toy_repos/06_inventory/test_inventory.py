import pytest
from inventory import *


@pytest.mark.parametrize("on_hand, reserved, expected", [(10,0,10),(10,3,7),(10,10,0),(2,5,0),(0,0,0),(100,25,75)])
def test_target(on_hand, reserved, expected):
    assert available_stock(on_hand, reserved) == pytest.approx(expected)


def test_regression():
    assert needs_reorder(2, 3)
    assert receive_stock(2, 3) == 5
    assert stock_value(3, 2.5) == 7.5
    assert normalize_sku(" ab ") == "AB"
