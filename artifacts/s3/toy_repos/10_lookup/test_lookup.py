import pytest
from lookup import *


@pytest.mark.parametrize("catalog, code, default, expected", [({"a":2},"a",0,2),({"a":2},"b",7,7),({},"x",3,3),({"a":0},"a",9,0),({},[],4,4),({"b":5},"c",0,0)])
def test_target(catalog, code, default, expected):
    assert lookup_price(catalog, code, default) == pytest.approx(expected)


def test_regression():
    assert product_codes({"b":2,"a":1}) == ["a","b"]
    assert add_product({},"x",2) == {"x":2}
    assert remove_product({"a":1},"a") == {}
    assert total_value({"a":2,"b":3}) == 5
    assert contains_product({"a":1},"a")
