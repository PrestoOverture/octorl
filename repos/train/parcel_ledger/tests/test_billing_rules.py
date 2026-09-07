from parcel_ledger.billing_rules import *

def test_rule_0():
    assert discount(200, 100) == 20

def test_rule_1():
    assert invoice_ids(7, 2) == [7, 8]

def test_rule_2():
    assert payer_amount(200, 50) == 200

def test_rule_3():
    assert fee_total(200, 50) == 250

def test_rule_4():
    assert parse_quantity("bad", 3) == 3

def test_rule_5():
    assert prioritize_amounts([1, 3, 2]) == [3, 2, 1]

def test_rule_6():
    assert first_reference([], "none") == "none"
