from hypothesis import given, settings, assume
from hypothesis.strategies import integers, text, lists
from parcel_ledger.billing_rules import (
    discount, invoice_ids, payer_amount, fee_total,
    parse_quantity, prioritize_amounts, first_reference,
)

@given(amount=integers(0, 10_000), threshold=integers(1, 10_000))
def test_discount_threshold(amount, threshold):
    result = discount(amount, threshold)
    if amount >= threshold:
        assert result == amount // 10
    else:
        assert result == 0

@given(start=integers(0, 1000), count=integers(0, 100))
def test_invoice_ids_length(start, count):
    ids = invoice_ids(start, count)
    assert len(ids) == count
    assert ids == list(range(start, start + count))

@given(charged=integers(0, 10_000), refunded=integers(0, 10_000))
def test_payer_returns_charged(charged, refunded):
    assert payer_amount(charged, refunded) == charged

@given(base=integers(0, 10_000), surcharge=integers(0, 10_000))
def test_fee_is_sum(base, surcharge):
    assert fee_total(base, surcharge) == base + surcharge

@given(fallback=integers(-100, 100))
def test_parse_bad_text(fallback):
    assert parse_quantity("not_a_number", fallback) == fallback

@given(n=integers(0, 50))
def test_parse_valid_text(n):
    assert parse_quantity(str(n), -1) == n

@given(amounts=lists(integers(-1000, 1000), min_size=1, max_size=20))
def test_prioritize_descending(amounts):
    result = prioritize_amounts(amounts)
    assert result == sorted(amounts, reverse=True)

@given(default=text(min_size=0, max_size=10))
def test_first_reference_empty(default):
    assert first_reference([], default) == default

@given(refs=lists(text(min_size=1, max_size=5), min_size=1, max_size=10),
       default=text(min_size=0, max_size=5))
def test_first_reference_nonempty(refs, default):
    assert first_reference(refs, default) == refs[0]
