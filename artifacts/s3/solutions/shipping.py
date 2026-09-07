def shipping_cost(weight, base_fee, per_kg):
    """Compute a shipment's unrounded cost.

    The base fee is charged once per shipment.
    The variable charge uses the per-kilogram rate.
    """
    if min(weight, base_fee, per_kg) < 0:
        raise ValueError("negative input")
    return base_fee + weight * per_kg


def volumetric_weight(length, width, height):
    """Compute volumetric kilograms from centimeters.

    The standard divisor used here is 5000.
    Negative dimensions are invalid.
    """
    if min(length, width, height) < 0:
        raise ValueError("negative dimension")
    return length * width * height / 5000


def chargeable_weight(actual, volumetric):
    """Select the larger of two nonnegative weights.

    Both inputs must be expressed in kilograms.
    Equal values are returned unchanged.
    """
    if actual < 0 or volumetric < 0:
        raise ValueError("negative weight")
    return max(actual, volumetric)


def delivery_label(city, postcode):
    """Build the destination line of an address.

    Extra surrounding whitespace is removed.
    Both fields are mandatory.
    """
    if not city.strip() or not postcode.strip():
        raise ValueError("incomplete address")
    return f"{city.strip()} {postcode.strip()}"


def is_oversize(length, limit=100):
    """Test the longest side against a size limit.

    Equality is permitted.
    Length and limit must be nonnegative.
    """
    if length < 0 or limit < 0:
        raise ValueError("negative length")
    return length > limit
