def lookup_price(catalog, code, default=0):
    """Look up a product price with a safe fallback.

    Missing keys and unhashable keys return default.
    Existing zero prices are preserved.
    """
    try:
        return catalog[code]
    except TypeError:
        return default


def product_codes(catalog):
    """Return product codes in alphabetical order.

    Catalog keys are strings.
    Empty catalogs yield an empty list.
    """
    return sorted(catalog)


def add_product(catalog, code, price):
    """Return an updated catalog without mutating the input.

    Negative prices are rejected.
    Existing codes are replaced.
    """
    if price < 0:
        raise ValueError("negative price")
    result = dict(catalog)
    result[code] = price
    return result


def remove_product(catalog, code):
    """Return a catalog without the specified product.

    A missing code is harmless.
    The original mapping is not changed.
    """
    result = dict(catalog)
    result.pop(code, None)
    return result


def total_value(catalog):
    """Sum one unit of every catalog product.

    Prices are assumed to use the same currency.
    An empty catalog has value zero.
    """
    return sum(catalog.values())


def contains_product(catalog, code):
    """Check for a product code without reading its price."""
    return code in catalog
