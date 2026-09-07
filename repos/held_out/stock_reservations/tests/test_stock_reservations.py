import pytest
from stock_reservations import Lot, Reservation, Warehouse, reorder_quantity, pick_list, parse_delivery


# ---- Warehouse basics ----

def test_receive_lot():
    w = Warehouse()
    lot = w.receive("L1", "SKU-A", 10, expires=5)
    assert lot == Lot("L1", "SKU-A", 10, 5)
    assert w.on_hand("SKU-A") == 10


def test_receive_no_expiry():
    w = Warehouse()
    lot = w.receive("L1", "SKU-A", 7)
    assert lot.expires is None
    assert w.available("SKU-A") == 7


def test_receive_duplicate_raises():
    w = Warehouse()
    w.receive("L1", "SKU-A", 10)
    with pytest.raises(ValueError):
        w.receive("L1", "SKU-A", 5)


def test_receive_invalid_quantity():
    w = Warehouse()
    with pytest.raises(ValueError):
        w.receive("L1", "SKU-A", 0)


def test_receive_already_expired():
    w = Warehouse(now=5)
    with pytest.raises(ValueError):
        w.receive("L1", "SKU-A", 10, expires=3)


def test_negative_clock_raises():
    with pytest.raises(ValueError):
        Warehouse(now=-1)


# ---- Liveness / expiry (catches condition-inversion) ----

def test_live_lot_available():
    w = Warehouse(now=0)
    w.receive("L1", "SKU-A", 10, expires=5)
    assert w.available("SKU-A") == 10


def test_expired_lot_not_available():
    """Catches condition-inversion: _live inverts the comparison."""
    w = Warehouse(now=0)
    w.receive("L1", "SKU-A", 10, expires=5)
    w.advance(5)
    assert w.available("SKU-A") == 0


# ---- available() excludes expired (catches boundary-condition-omission) ----

def test_available_excludes_expired_lots():
    """Catches boundary-condition-omission: available() missing _live check."""
    w = Warehouse(now=0)
    w.receive("L1", "SKU-A", 10, expires=3)
    w.receive("L2", "SKU-A", 5)
    w.advance(3)
    assert w.available("SKU-A") == 5


# ---- on_hand (catches wrong-return-value) ----

def test_on_hand_includes_reserved():
    """Catches wrong-return-value: on_hand deducts reserved."""
    w = Warehouse()
    w.receive("L1", "SKU-A", 10)
    w.reserve("R1", "SKU-A", 3, ttl=10)
    assert w.on_hand("SKU-A") == 10


def test_on_hand_include_expired():
    w = Warehouse()
    w.receive("L1", "SKU-A", 10, expires=3)
    w.advance(3)
    assert w.on_hand("SKU-A") == 0
    assert w.on_hand("SKU-A", include_expired=True) == 10


# ---- reserve sort order (catches api-parameter-error) ----

def test_reserve_earliest_expiry_first():
    """Catches api-parameter-error: reserve sorts by identifier not expiry."""
    w = Warehouse()
    w.receive("Z1", "SKU-A", 10, expires=20)
    w.receive("A1", "SKU-A", 10, expires=30)
    r = w.reserve("R1", "SKU-A", 5, ttl=5)
    assert r.allocations[0][0] == "Z1"


# ---- ship (catches variable-misuse) ----

def test_ship_reduces_lot():
    """Catches variable-misuse: ship adds instead of subtracts."""
    w = Warehouse()
    w.receive("L1", "SKU-A", 10)
    w.reserve("R1", "SKU-A", 3, ttl=10)
    w.ship("R1")
    assert w.on_hand("SKU-A") == 7
    assert w.shipped("SKU-A") == 3


# ---- reorder_quantity (catches off-by-one) ----

def test_reorder_exact_packs():
    """Catches off-by-one: ceiling division formula overshoots."""
    assert reorder_quantity(7, 10, 3) == 3


def test_reorder_no_shortage():
    assert reorder_quantity(10, 10) == 0


def test_reorder_from_zero():
    assert reorder_quantity(0, 6, 3) == 6


def test_reorder_invalid():
    with pytest.raises(ValueError):
        reorder_quantity(-1, 10)


# ---- parse_delivery (catches missing-exception-handling) ----

def test_parse_delivery_bad_line():
    """Catches missing-exception-handling: no try/except wrapping."""
    with pytest.raises(ValueError, match="invalid delivery row"):
        parse_delivery("bad-line-no-colon")


def test_parse_delivery_valid():
    result = parse_delivery("SKU-A: 10\nSKU-B: 5\nSKU-A: 3")
    assert result == [("SKU-A", 13), ("SKU-B", 5)]


def test_parse_delivery_blank_lines():
    result = parse_delivery("\nSKU-A: 2\n\n")
    assert result == [("SKU-A", 2)]


# ---- reservation lifecycle ----

def test_reserve_and_release():
    w = Warehouse()
    w.receive("L1", "SKU-A", 10)
    w.reserve("R1", "SKU-A", 5, ttl=10)
    assert w.available("SKU-A") == 5
    w.release("R1")
    assert w.available("SKU-A") == 10


def test_reserve_insufficient_stock():
    w = Warehouse()
    w.receive("L1", "SKU-A", 5)
    with pytest.raises(ValueError):
        w.reserve("R1", "SKU-A", 20, ttl=10)


def test_reservation_quantity_property():
    r = Reservation("R1", "SKU-A", (("L1", 3), ("L2", 2)), expires=10)
    assert r.quantity == 5


def test_advance_expires_reservations():
    w = Warehouse()
    w.receive("L1", "SKU-A", 10)
    w.reserve("R1", "SKU-A", 3, ttl=5)
    expired = w.advance(5)
    assert "R1" in expired
    assert w.available("SKU-A") == 10


def test_advance_backward_raises():
    w = Warehouse(now=5)
    with pytest.raises(ValueError):
        w.advance(3)


def test_discard_expired_lots():
    w = Warehouse()
    w.receive("L1", "SKU-A", 10, expires=3)
    w.advance(3)
    discarded = w.discard_expired()
    assert len(discarded) == 1
    assert discarded[0].identifier == "L1"


def test_discard_keeps_live_lots():
    w = Warehouse()
    w.receive("L1", "SKU-A", 10, expires=3)
    w.receive("L2", "SKU-A", 5, expires=10)
    w.advance(3)
    discarded = w.discard_expired()
    assert len(discarded) == 1
    assert discarded[0].identifier == "L1"
    assert len(w.lots()) == 1


# ---- lots / reservations ----

def test_lots_all():
    w = Warehouse()
    w.receive("L2", "SKU-A", 5)
    w.receive("L1", "SKU-B", 3)
    assert len(w.lots()) == 2
    assert w.lots()[0].identifier == "L1"


def test_lots_by_sku():
    w = Warehouse()
    w.receive("L1", "SKU-A", 5)
    w.receive("L2", "SKU-B", 3)
    assert len(w.lots("SKU-A")) == 1


def test_reservations():
    w = Warehouse()
    w.receive("L1", "SKU-A", 10)
    w.reserve("R1", "SKU-A", 3, ttl=10)
    assert len(w.reservations()) == 1


# ---- audit ----

def test_audit_clean():
    w = Warehouse()
    w.receive("L1", "SKU-A", 10)
    w.reserve("R1", "SKU-A", 3, ttl=10)
    assert w.audit() is True


# ---- summary ----

def test_summary():
    w = Warehouse()
    w.receive("L1", "SKU-A", 10)
    w.reserve("R1", "SKU-A", 3, ttl=10)
    w.ship("R1")
    s = w.summary()
    assert s["SKU-A"]["on_hand"] == 7
    assert s["SKU-A"]["available"] == 7
    assert s["SKU-A"]["shipped"] == 3


# ---- adjust ----

def test_adjust():
    w = Warehouse()
    w.receive("L1", "SKU-A", 10)
    lot = w.adjust("L1", 15)
    assert lot.quantity == 15
    assert w.on_hand("SKU-A") == 15


def test_adjust_violates_reservation():
    w = Warehouse()
    w.receive("L1", "SKU-A", 10)
    w.reserve("R1", "SKU-A", 5, ttl=10)
    with pytest.raises(ValueError):
        w.adjust("L1", 3)


# ---- transfer_sku ----

def test_transfer_sku():
    w = Warehouse()
    w.receive("L1", "SKU-A", 10)
    lot = w.transfer_sku("L1", "SKU-B")
    assert lot.sku == "SKU-B"
    assert w.on_hand("SKU-A") == 0
    assert w.on_hand("SKU-B") == 10


def test_transfer_reserved_raises():
    w = Warehouse()
    w.receive("L1", "SKU-A", 10)
    w.reserve("R1", "SKU-A", 5, ttl=10)
    with pytest.raises(ValueError):
        w.transfer_sku("L1", "SKU-B")


# ---- pick_list ----

def test_pick_list():
    r1 = Reservation("R1", "SKU-A", (("L1", 3), ("L2", 2)), expires=10)
    r2 = Reservation("R2", "SKU-A", (("L1", 1),), expires=10)
    pl = pick_list([r1, r2])
    assert pl == [("L1", 4), ("L2", 2)]


def test_pick_list_empty():
    assert pick_list([]) == []


# ---- shipped ----

def test_shipped_default_zero():
    w = Warehouse()
    assert w.shipped("SKU-X") == 0
