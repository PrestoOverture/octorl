import pytest
from shipping import *


@pytest.mark.parametrize("weight, base_fee, per_kg, expected", [(0,5,2,5),(1,5,2,7),(3,5,2,11),(2,0,4,8),(4,3,1,7),(1.5,2,3,6.5)])
def test_target(weight, base_fee, per_kg, expected):
    assert shipping_cost(weight, base_fee, per_kg) == pytest.approx(expected)


def test_regression():
    assert volumetric_weight(10,10,10) == .2
    assert chargeable_weight(2,3) == 3
    assert delivery_label(" X ", " 12 ") == "X 12"
    assert is_oversize(101)
