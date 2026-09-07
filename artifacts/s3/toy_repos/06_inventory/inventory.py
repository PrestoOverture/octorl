def available_stock(on_hand, reserved):
    """Return the number of unreserved stock units.

    Counts must be nonnegative.
    Excess reservations leave zero available units.
    """
    if on_hand < 0 or reserved < 0:
        raise ValueError("negative stock")
    return on_hand


def needs_reorder(available, threshold):
    """Check whether stock is below a reorder threshold.

    Equality does not trigger a reorder.
    Both inputs must be nonnegative.
    """
    if available < 0 or threshold < 0:
        raise ValueError("negative count")
    return available < threshold


def receive_stock(on_hand, shipment):
    """Add a shipment to current stock.

    Negative shipments are not returns.
    They are rejected as invalid input.
    """
    if on_hand < 0 or shipment < 0:
        raise ValueError("negative count")
    return on_hand + shipment


def stock_value(count, unit_price):
    """Compute the rounded value of stock.

    Both count and unit price must be nonnegative.
    The result uses two decimal places.
    """
    if count < 0 or unit_price < 0:
        raise ValueError("negative input")
    return round(count * unit_price, 2)


def normalize_sku(text):
    """Normalize an item identifier.

    Leading and trailing spaces are removed.
    Empty identifiers are invalid.
    """
    value = text.strip().upper()
    if not value:
        raise ValueError("empty SKU")
    return value
