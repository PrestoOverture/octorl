def celsius_to_fahrenheit(value):
    """Convert a Celsius temperature into Fahrenheit.

    Water freezes at zero Celsius and 32 Fahrenheit.
    Fractional temperatures are supported.
    """
    return value * 9 / 5 + 32


def fahrenheit_to_celsius(value):
    """Convert a Fahrenheit temperature into Celsius.

    The conversion uses the standard linear scale.
    Negative values are supported.
    """
    return (value - 32) * 5 / 9


def celsius_to_kelvin(value):
    """Convert Celsius into Kelvin.

    Temperatures below absolute zero are rejected.
    Zero Kelvin is a valid result.
    """
    if value < -273.15:
        raise ValueError("below absolute zero")
    return value + 273.15


def is_freezing(value):
    """Report whether a Celsius reading is at most zero.

    This utility assumes water at standard pressure.
    Exactly zero is included.
    """
    return value <= 0


def average_temperature(values):
    """Average a nonempty list of readings.

    All readings must use the same temperature scale.
    An empty list has no mean.
    """
    if not values:
        raise ValueError("no readings")
    return sum(values) / len(values)


def label(value, unit):
    """Render a reading for display.

    The unit is supplied by the caller.
    One decimal place is shown.
    """
    return f"{value:.1f} {unit}"
