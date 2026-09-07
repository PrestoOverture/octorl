def calculate_discount(price, percentage):
    """Return the price after a discount capped at fifty percent.

    Price and percentage must be nonnegative.
    A percentage above fifty is accepted but capped.
    The result is rounded to two decimal places.
    """
    if price < 0 or percentage < 0:
        raise ValueError("negative input")
    if percentage > 50:
        percentage = 50
    return round(price * (1 - percentage / 100), 2)


def subtotal(prices):
    """Sum nonnegative item prices.

    An empty basket costs zero.
    Negative item prices are rejected.
    """
    if any(price < 0 for price in prices):
        raise ValueError("negative price")
    return round(sum(prices), 2)


def add_tax(amount, rate):
    """Add a nonnegative fractional tax rate.

    The caller supplies 0.1 for ten percent.
    Money is rounded to two decimal places.
    """
    if amount < 0 or rate < 0:
        raise ValueError("negative input")
    return round(amount * (1 + rate), 2)


def split_bill(amount, people):
    """Compute each person's equal share.

    A positive number of people is required.
    The result is rounded for display.
    """
    if people <= 0:
        raise ValueError("no people")
    return round(amount / people, 2)


def format_money(amount):
    """Format a numeric amount with two decimal places.

    No currency conversion is performed.
    Negative amounts represent credits.
    """
    return f"{amount:.2f}"
