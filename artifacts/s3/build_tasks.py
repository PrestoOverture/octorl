"""Deterministic S3 fixtures. One LibCST statement replacement per task."""
from pathlib import Path
import difflib
import json
import libcst as cst

ROOT = Path(__file__).resolve().parent
SEED = 20260905  # No random task generation; recorded for the experiment.
# Each module contains useful related helpers and public contract documentation.
SPECS = [
('billing', 'calculate_discount', 'condition inversion',
 'if percentage > 50:', 'if percentage < 50:',
 'returns incorrect totals when the discount percentage exceeds 50%. Discounts must be capped at 50%.',
 '''def calculate_discount(price, percentage):
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
''',
 '[(100, 0, 100), (100, 10, 90), (100, 50, 50), (100, 75, 50), (80, 100, 40), (0, 70, 0)]', 'price, percentage, expected', 'calculate_discount(price, percentage)',
 'assert subtotal([2, 3]) == 5\n    assert add_tax(10, .1) == 11\n    assert split_bill(9, 3) == 3\n    assert format_money(2) == "2.00"'),
('access', 'can_enter', 'condition inversion', 'if age < minimum_age:', 'if age >= minimum_age:',
 'incorrectly accepts underage visitors and rejects eligible visitors. The minimum age is inclusive.',
 '''def can_enter(age, minimum_age=18):
    """Return whether a visitor meets the minimum age.

    The lower age bound is inclusive.
    Negative ages and minimum ages are invalid.
    """
    if age < 0 or minimum_age < 0:
        raise ValueError("negative age")
    if age < minimum_age:
        return False
    return True


def normalize_name(name):
    """Collapse whitespace in a visitor name.

    Empty names are rejected.
    Capitalization is preserved.
    """
    value = " ".join(name.split())
    if not value:
        raise ValueError("empty name")
    return value


def badge_label(name, number):
    """Build a stable badge label.

    Badge numbers must be positive.
    Names are normalized before display.
    """
    if number <= 0:
        raise ValueError("invalid badge")
    return f"{number}: {normalize_name(name)}"


def free_places(capacity, occupants):
    """Count remaining places without going negative.

    Capacity and occupants are nonnegative counts.
    Overcrowding reports zero free places.
    """
    if capacity < 0 or occupants < 0:
        raise ValueError("negative count")
    return max(0, capacity - occupants)


def is_full(capacity, occupants):
    """Report whether there are no free places.

    A zero-capacity room is always full.
    Validation is delegated to free_places.
    """
    return free_places(capacity, occupants) == 0
''', '[(0,18,False),(17,18,False),(18,18,True),(25,18,True),(5,5,True),(4,5,False)]', 'age, minimum, expected', 'can_enter(age, minimum)',
 'assert normalize_name(" A  B ") == "A B"\n    assert badge_label("Ada", 2) == "2: Ada"\n    assert free_places(5, 3) == 2\n    assert is_full(2, 2)'),
('pages', 'page_count', 'off-by-one', 'return (total + size - 1) // size', 'return total // size',
 'undercounts pages when the final page is only partially filled. Empty input requires zero pages.',
 '''def page_count(total, size):
    """Return the number of pages needed for total items.

    A partially filled page still counts as one page.
    Total must be nonnegative and size positive.
    """
    if total < 0 or size <= 0:
        raise ValueError("invalid pagination")
    return (total + size - 1) // size


def page_slice(items, index, size):
    """Extract a zero-based page from a sequence.

    Pages beyond the end are empty.
    Negative page indices are rejected.
    """
    if index < 0 or size <= 0:
        raise ValueError("invalid pagination")
    start = index * size
    return items[start:start + size]


def has_next(total, index, size):
    """Check whether another page contains items.

    Index is zero-based.
    The comparison uses the next page's start.
    """
    if total < 0 or index < 0 or size <= 0:
        raise ValueError("invalid pagination")
    return (index + 1) * size < total


def item_offset(index, size):
    """Return the starting offset of a page.

    Both arguments use integer counts.
    Size must be strictly positive.
    """
    if index < 0 or size <= 0:
        raise ValueError("invalid pagination")
    return index * size


def clamp_page(index, count):
    """Clamp a page index into an existing page range.

    An empty collection uses index zero.
    This helper does not allocate a page.
    """
    return max(0, min(index, max(0, count - 1)))
''', '[(0,5,0),(1,5,1),(5,5,1),(6,5,2),(10,5,2),(11,5,3)]', 'total, size, expected', 'page_count(total, size)',
 'assert page_slice(list(range(6)), 1, 2) == [2, 3]\n    assert has_next(6, 0, 2)\n    assert item_offset(2, 5) == 10\n    assert clamp_page(9, 3) == 2'),
('ranges', 'inclusive_sum', 'off-by-one', 'for value in range(start, end + 1):', 'for value in range(start, end):',
 'omits the upper endpoint. Both start and end must be included in the sum.',
 '''def inclusive_sum(start, end):
    """Sum all integers from start through end.

    Both endpoints are included.
    An inverted interval contains no values.
    """
    total = 0
    for value in range(start, end + 1):
        total += value
    return total


def contains(start, end, value):
    """Test membership in a closed interval.

    Values at either endpoint are members.
    Inverted intervals have no members.
    """
    return start <= value <= end


def overlap(left, right):
    """Find the closed intersection of two intervals.

    A one-point intersection is valid.
    Disjoint intervals return None.
    """
    start = max(left[0], right[0])
    end = min(left[1], right[1])
    return (start, end) if start <= end else None


def clamp(value, start, end):
    """Clamp a value into a closed interval.

    Inverted intervals are invalid.
    Existing members are returned unchanged.
    """
    if start > end:
        raise ValueError("inverted interval")
    return max(start, min(value, end))


def width(start, end):
    """Return the geometric width of an interval.

    Unlike integer membership count, endpoints add no width.
    An inverted interval has width zero.
    """
    return max(0, end - start)


def shift(interval, offset):
    """Move both endpoints by the same signed offset."""
    return interval[0] + offset, interval[1] + offset
''', '[(1,1,1),(1,3,6),(0,4,10),(-2,2,0),(3,2,0),(-3,-1,-6)]', 'start, end, expected', 'inclusive_sum(start, end)',
 'assert contains(1, 3, 3)\n    assert overlap((1, 3), (3, 5)) == (3, 3)\n    assert clamp(9, 1, 3) == 3\n    assert width(1, 3) == 2\n    assert shift((1, 3), 2) == (3, 5)'),
('temperature', 'celsius_to_fahrenheit', 'wrong return value', 'return value * 9 / 5 + 32', 'return value * 9 / 5',
 'returns Fahrenheit values without the required 32-degree offset.',
 '''def celsius_to_fahrenheit(value):
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
''', '[(0,32),(100,212),(-40,-40),(10,50),(25,77),(-10,14)]', 'value, expected', 'celsius_to_fahrenheit(value)',
 'assert fahrenheit_to_celsius(32) == 0\n    assert celsius_to_kelvin(0) == 273.15\n    assert is_freezing(0)\n    assert average_temperature([10,20]) == 15\n    assert label(2, "C") == "2.0 C"'),
('inventory', 'available_stock', 'wrong return value', 'return max(0, on_hand - reserved)', 'return on_hand',
 'does not subtract reserved units. Availability is on-hand minus reserved, with a floor of zero.',
 '''def available_stock(on_hand, reserved):
    """Return the number of unreserved stock units.

    Counts must be nonnegative.
    Excess reservations leave zero available units.
    """
    if on_hand < 0 or reserved < 0:
        raise ValueError("negative stock")
    return max(0, on_hand - reserved)


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
''', '[(10,0,10),(10,3,7),(10,10,0),(2,5,0),(0,0,0),(100,25,75)]', 'on_hand, reserved, expected', 'available_stock(on_hand, reserved)',
 'assert needs_reorder(2, 3)\n    assert receive_stock(2, 3) == 5\n    assert stock_value(3, 2.5) == 7.5\n    assert normalize_sku(" ab ") == "AB"'),
('shipping', 'shipping_cost', 'variable misuse', 'return base_fee + weight * per_kg', 'return base_fee + weight * base_fee',
 'uses the base fee as the per-kilogram rate. Total cost is base fee plus weight times per_kg.',
 '''def shipping_cost(weight, base_fee, per_kg):
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
''', '[(0,5,2,5),(1,5,2,7),(3,5,2,11),(2,0,4,8),(4,3,1,7),(1.5,2,3,6.5)]', 'weight, base_fee, per_kg, expected', 'shipping_cost(weight, base_fee, per_kg)',
 'assert volumetric_weight(10,10,10) == .2\n    assert chargeable_weight(2,3) == 3\n    assert delivery_label(" X ", " 12 ") == "X 12"\n    assert is_oversize(101)'),
('scores', 'weighted_score', 'variable misuse', 'return sum(value * weight for value, weight in zip(values, weights)) / total_weight', 'return sum(value * weight for value, weight in zip(values, weights)) / len(values)',
 'divides by the number of values instead of the sum of weights. Return the weighted mean.',
 '''def weighted_score(values, weights):
    """Compute the weighted mean of matching nonempty lists.

    Weights must be nonnegative with a positive total.
    The result is normalized by total weight.
    """
    if not values or len(values) != len(weights):
        raise ValueError("mismatched scores")
    if any(weight < 0 for weight in weights):
        raise ValueError("negative weight")
    total_weight = sum(weights)
    if total_weight <= 0:
        raise ValueError("zero total weight")
    return sum(value * weight for value, weight in zip(values, weights)) / total_weight


def mean_score(values):
    """Compute the arithmetic mean of nonempty scores.

    Every score has equal weight.
    An empty list is invalid.
    """
    if not values:
        raise ValueError("no values")
    return sum(values) / len(values)


def passed(value, threshold=60):
    """Test a score against an inclusive pass threshold.

    Scores are not rounded before comparison.
    Exactly the threshold is a pass.
    """
    return value >= threshold


def best_score(values):
    """Find the highest score in a nonempty sequence.

    Ties have no special treatment.
    Empty sequences are invalid.
    """
    if not values:
        raise ValueError("no values")
    return max(values)


def normalize_score(value, maximum):
    """Convert a score to a percentage.

    Maximum must be positive.
    Extra-credit scores may exceed one hundred.
    """
    if maximum <= 0:
        raise ValueError("invalid maximum")
    return value / maximum * 100
''', '[([10,20],[1,3],17.5),([5],[2],5),([0,10],[0,2],10),([2,8],[1,1],5),([3,9],[2,4],7),([1,2,3],[2,2,2],2)]', 'values, weights, expected', 'weighted_score(values, weights)',
 'assert mean_score([2,4]) == 3\n    assert passed(60)\n    assert best_score([2,3]) == 3\n    assert normalize_score(5,10) == 50'),
('parsing', 'parse_integer', 'missing exception handling', 'except (TypeError, ValueError):', 'except TypeError:',
 'raises ValueError for invalid integer text instead of returning the supplied default.',
 '''def parse_integer(text, default=0):
    """Parse an integer, returning default for invalid input.

    Both invalid types and invalid text are handled.
    Whitespace and signed integer strings are accepted.
    """
    try:
        return int(text)
    except (TypeError, ValueError):
        return default


def parse_boolean(text):
    """Parse an explicit boolean word.

    Matching ignores whitespace and case.
    Unknown words are rejected.
    """
    value = text.strip().lower()
    if value in ("true", "yes", "1"):
        return True
    if value in ("false", "no", "0"):
        return False
    raise ValueError("invalid boolean")


def split_fields(text, separator=","):
    """Split and trim delimited text.

    Empty fields are retained.
    This is not a quoted CSV parser.
    """
    return [part.strip() for part in text.split(separator)]


def required_text(text):
    """Normalize a mandatory text field.

    Surrounding whitespace is removed.
    Empty text is invalid.
    """
    result = text.strip()
    if not result:
        raise ValueError("missing text")
    return result


def join_fields(fields, separator=","):
    """Join fields with the requested separator.

    Every field is converted to text.
    No quoting or escaping is performed.
    """
    return separator.join(str(field) for field in fields)
''', '[("12",0,12),(" -3 ",0,-3),("bad",7,7),("",9,9),(None,4,4),("2.5",8,8)]', 'text, default, expected', 'parse_integer(text, default)',
 'assert parse_boolean(" YES ")\n    assert split_fields("a, b") == ["a", "b"]\n    assert required_text(" x ") == "x"\n    assert join_fields([1,2]) == "1,2"'),
('lookup', 'lookup_price', 'missing exception handling', 'except (KeyError, TypeError):', 'except TypeError:',
 'raises KeyError for absent product codes instead of returning the default price.',
 '''def lookup_price(catalog, code, default=0):
    """Look up a product price with a safe fallback.

    Missing keys and unhashable keys return default.
    Existing zero prices are preserved.
    """
    try:
        return catalog[code]
    except (KeyError, TypeError):
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
''', '[({"a":2},"a",0,2),({"a":2},"b",7,7),({},"x",3,3),({"a":0},"a",9,0),({},[],4,4),({"b":5},"c",0,0)]', 'catalog, code, default, expected', 'lookup_price(catalog, code, default)',
 'assert product_codes({"b":2,"a":1}) == ["a","b"]\n    assert add_product({},"x",2) == {"x":2}\n    assert remove_product({"a":1},"a") == {}\n    assert total_value({"a":2,"b":3}) == 5\n    assert contains_product({"a":1},"a")'),
]

class Inject(cst.CSTTransformer):
    def __init__(self, old, new):
        self.old, self.new, self.count = old, new, 0

    def on_leave(self, original_node, updated_node):
        text = cst.Module([]).code_for_node(original_node).strip()
        # Header edits preserve the original body; expression edits affect one return.
        if isinstance(original_node, cst.If) and text.splitlines()[0] == self.old:
            self.count += 1
            return updated_node.with_changes(test=cst.parse_expression(self.new[3:-1]))
        if isinstance(original_node, cst.For) and text.splitlines()[0] == self.old:
            self.count += 1
            replacement = cst.parse_statement(self.new + '\n    pass\n')
            return updated_node.with_changes(iter=replacement.iter)
        if isinstance(original_node, cst.ExceptHandler) and text.splitlines()[0] == self.old:
            self.count += 1
            return updated_node.with_changes(type=cst.parse_expression(self.new[7:-1]))
        if isinstance(original_node, cst.Return) and text == self.old:
            self.count += 1
            return updated_node.with_changes(value=cst.parse_expression(self.new[7:]))
        return updated_node


def build():
    manifest = []
    for index, spec in enumerate(SPECS):
        name, function, kind, old, new, issue, source, cases, args, call, regression = spec
        assert 50 <= len(source.splitlines()) <= 150, (name, len(source.splitlines()))
        mutation = Inject(old, new)
        buggy = cst.parse_module(source).visit(mutation).code
        assert mutation.count == 1, (name, mutation.count)
        task_id = f'{index + 1:02d}_{name}'
        task_dir = ROOT / 'toy_repos' / task_id
        task_dir.mkdir(parents=True, exist_ok=True)
        (task_dir / f'{name}.py').write_text(buggy)
        (ROOT / 'solutions').mkdir(exist_ok=True)
        (ROOT / 'solutions' / f'{name}.py').write_text(source)
        test = f'import pytest\nfrom {name} import *\n\n\n@pytest.mark.parametrize("{args}", {cases})\ndef test_target({args}):\n    assert {call} == pytest.approx(expected)\n\n\ndef test_regression():\n    {regression}\n'
        (task_dir / f'test_{name}.py').write_text(test)
        diff = ''.join(difflib.unified_diff(buggy.splitlines(True), source.splitlines(True), fromfile=f'a/{name}.py', tofile=f'b/{name}.py'))
        (ROOT / 'solutions' / f'{name}.patch').write_text(diff)
        manifest.append(dict(task_id=task_id, module=f'{name}.py', function=function,
                             bug_type=kind, source='mutation', count=1, hint='L0', span='single_function',
                             issue=f'The function `{function}` in `{name}.py` {issue}',
                             generation_seed=SEED, source_lines=len(source.splitlines())))
    (ROOT / 'task_manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')

if __name__ == '__main__':
    build()
