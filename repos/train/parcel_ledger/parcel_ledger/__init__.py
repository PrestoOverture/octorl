"""Integer-cent, append-only parcel billing ledger authored for OctoRL."""
from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Iterable


@dataclass(frozen=True)
class Entry:
    """A posted transaction. Positive amounts credit an account."""
    sequence: int
    account: str
    amount: int
    reference: str
    kind: str


def cents(text: str) -> int:
    """Parse fixed-point money without binary floating point rounding."""
    value = text.strip()
    if not value:
        raise ValueError("empty amount")
    sign = -1 if value.startswith("-") else 1
    value = value.removeprefix("-").removeprefix("+")
    parts = value.split(".")
    if len(parts) > 2 or not parts[0].isdigit():
        raise ValueError("invalid amount")
    fraction = parts[1] if len(parts) == 2 else ""
    if len(fraction) > 2 or (fraction and not fraction.isdigit()):
        raise ValueError("invalid fraction")
    return sign * (int(parts[0]) * 100 + int(fraction.ljust(2, "0")))


def format_cents(amount: int) -> str:
    """Render cents exactly, including negative amounts below one unit."""
    sign = "-" if amount < 0 else ""
    whole, fraction = divmod(abs(amount), 100)
    return f"{sign}{whole}.{fraction:02d}"


def allocate(amount: int, weights: Iterable[int]) -> list[int]:
    """Largest-remainder allocation; ties favor the earliest recipient."""
    values = list(weights)
    if amount < 0 or any(weight < 0 for weight in values):
        raise ValueError("negative allocation input")
    total = sum(values)
    if total == 0:
        if amount == 0:
            return [0] * len(values)
        raise ValueError("positive weight required")
    shares = [amount * weight // total for weight in values]
    remainder = amount - sum(shares)
    order = sorted(range(len(values)), key=lambda i: (-(amount * values[i] % total), i))
    for index in order[:remainder]:
        shares[index] += 1
    return shares


class Ledger:
    """In-memory ledger with atomic transfers and idempotency references.

    Accounts cannot be overdrawn. Repeating a reference is accepted only
    for the exact same request; conflicting retries raise ValueError.
    """

    def __init__(self) -> None:
        self._accounts: dict[str, int] = {}
        self._entries: list[Entry] = []
        self._requests: dict[str, tuple[object, ...]] = {}
        self._closed: set[str] = set()

    def open(self, name: str) -> None:
        if not name or name.strip() != name:
            raise ValueError("account name must be nonempty and trimmed")
        if name in self._accounts:
            raise ValueError("account already exists")
        self._accounts[name] = 0

    def _check(self, name: str) -> None:
        if name not in self._accounts:
            raise KeyError(name)
        if name in self._closed:
            raise ValueError("account is closed")

    def _retry(self, reference: str, request: tuple[object, ...]) -> bool:
        if not reference:
            raise ValueError("reference required")
        previous = self._requests.get(reference)
        if previous is None:
            return False
        if previous != request:
            raise ValueError("reference conflict")
        return True

    def _post(self, name: str, amount: int, reference: str, kind: str) -> None:
        sequence = len(self._entries) + 1
        self._entries.append(Entry(sequence, name, amount, reference, kind))
        self._accounts[name] += amount

    def deposit(self, name: str, amount: int, reference: str) -> None:
        request = ("deposit", name, amount)
        if self._retry(reference, request):
            return
        self._check(name)
        if amount <= 0:
            raise ValueError("deposit must be positive")
        self._post(name, amount, reference, "deposit")
        self._requests[reference] = request

    def withdraw(self, name: str, amount: int, reference: str) -> None:
        request = ("withdraw", name, amount)
        if self._retry(reference, request):
            return
        self._check(name)
        if amount <= 0 or amount > self._accounts[name]:
            raise ValueError("invalid withdrawal")
        self._post(name, -amount, reference, "withdraw")
        self._requests[reference] = request

    def transfer(self, source: str, target: str, amount: int, reference: str) -> None:
        request = ("transfer", source, target, amount)
        if self._retry(reference, request):
            return
        self._check(source)
        self._check(target)
        if source == target:
            raise ValueError("distinct accounts required")
        if amount <= 0 or amount > self._accounts[source]:
            raise ValueError("invalid transfer")
        self._post(source, -amount, reference, "transfer_out")
        self._post(target, amount, reference, "transfer_in")
        self._requests[reference] = request

    def balance(self, name: str) -> int:
        if name not in self._accounts:
            raise KeyError(name)
        return self._accounts[name]

    def statement(self, name: str, after: int = 0) -> tuple[Entry, ...]:
        if name not in self._accounts:
            raise KeyError(name)
        return tuple(entry for entry in self._entries if entry.account == name and entry.sequence > after)

    def close(self, name: str) -> None:
        self._check(name)
        if self._accounts[name] != 0:
            raise ValueError("cannot close funded account")
        self._closed.add(name)

    def accounts(self, include_closed: bool = False) -> tuple[str, ...]:
        return tuple(sorted(name for name in self._accounts if include_closed or name not in self._closed))

    def trial_balance(self) -> dict[str, int]:
        result = dict.fromkeys(self._accounts, 0)
        for entry in self._entries:
            result[entry.account] += entry.amount
        return result

    def audit(self) -> bool:
        sequences = [entry.sequence for entry in self._entries]
        return sequences == list(range(1, len(sequences) + 1)) and self.trial_balance() == self._accounts

    def refund(self, reference: str, refund_reference: str) -> None:
        request = ("refund", reference)
        if self._retry(refund_reference, request):
            return
        original = self._requests.get(reference)
        if original is None or original[0] != "withdraw":
            raise ValueError("only withdrawals can be refunded")
        if any(value == request for value in self._requests.values()):
            raise ValueError("withdrawal already refunded")
        _, name, amount = original
        self._check(str(name))
        self._post(str(name), int(amount), refund_reference, "refund")
        self._requests[refund_reference] = request

    def export(self) -> list[dict[str, str | int]]:
        return [dict(sequence=e.sequence, account=e.account, amount=e.amount,
                     reference=e.reference, kind=e.kind) for e in self._entries]


def shipping_charge(weight: int, base: int, per_block: int, block_size: int = 100) -> int:
    """Round positive gram weights upward to billable blocks."""
    if weight <= 0 or block_size <= 0 or base < 0 or per_block < 0:
        raise ValueError("invalid tariff")
    blocks = (weight + block_size - 1) // block_size
    return base + blocks * per_block


def invoice(lines: Iterable[tuple[int, int]], discount_bps: int = 0) -> int:
    """Sum quantity/unit-cent lines, then apply a floor-rounded discount."""
    if not 0 <= discount_bps <= 10000:
        raise ValueError("discount out of range")
    total = 0
    for quantity, price in lines:
        if quantity < 0 or price < 0:
            raise ValueError("negative invoice line")
        total += quantity * price
    return total - total * discount_bps // 10000


def parse_lines(text: str) -> list[tuple[int, int]]:
    """Parse quantity,price rows, reporting the row of a malformed item."""
    result = []
    for row, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            continue
        try:
            quantity_text, price_text = line.split(",")
            quantity = int(quantity_text)
            price = cents(price_text)
            if quantity < 0 or price < 0:
                raise ValueError("negative item")
        except ValueError as error:
            raise ValueError(f"invalid invoice row {row}") from error
        result.append((quantity, price))
    return result
