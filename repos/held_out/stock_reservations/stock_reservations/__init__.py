"""Stock lots and expiring reservations using an explicit logical clock."""
from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Iterable


@dataclass(frozen=True)
class Lot:
    identifier: str
    sku: str
    quantity: int
    expires: int | None


@dataclass(frozen=True)
class Reservation:
    identifier: str
    sku: str
    allocations: tuple[tuple[str, int], ...]
    expires: int

    @property
    def quantity(self) -> int:
        return sum(quantity for _, quantity in self.allocations)


class Warehouse:
    """Reserve earliest-expiring lots first. Expiry is exclusive at now."""

    def __init__(self, now: int = 0) -> None:
        if now < 0:
            raise ValueError("negative clock")
        self.now = now
        self._lots: dict[str, Lot] = {}
        self._reservations: dict[str, Reservation] = {}
        self._used_ids: set[str] = set()
        self._shipped: dict[str, int] = {}

    def receive(self, identifier: str, sku: str, quantity: int,
                expires: int | None = None) -> Lot:
        if not identifier or identifier in self._lots or not sku:
            raise ValueError("new lot identifier and SKU required")
        if quantity <= 0 or (expires is not None and expires <= self.now):
            raise ValueError("invalid lot quantity or expiry")
        lot = Lot(identifier, sku, quantity, expires)
        self._lots[identifier] = lot
        return lot

    def _live(self, lot: Lot) -> bool:
        return lot.expires is None or lot.expires > self.now

    def reserved(self, lot_id: str) -> int:
        if lot_id not in self._lots:
            raise KeyError(lot_id)
        return sum(quantity for reservation in self._reservations.values()
                   for identifier, quantity in reservation.allocations if identifier == lot_id)

    def available(self, sku: str) -> int:
        return sum(lot.quantity - self.reserved(lot.identifier)
                   for lot in self._lots.values() if lot.sku == sku and self._live(lot))

    def on_hand(self, sku: str, include_expired: bool = False) -> int:
        return sum(lot.quantity for lot in self._lots.values()
                   if lot.sku == sku and (include_expired or self._live(lot)))

    def reserve(self, identifier: str, sku: str, quantity: int, ttl: int) -> Reservation:
        if not identifier or identifier in self._used_ids:
            raise ValueError("reservation identifier already used or empty")
        if quantity <= 0 or ttl <= 0:
            raise ValueError("positive quantity and TTL required")
        expires = self.now + ttl
        eligible = [lot for lot in self._lots.values() if lot.sku == sku
                    and (lot.expires is None or lot.expires >= expires)]
        eligible.sort(key=lambda lot: (lot.expires is None, lot.expires or 0, lot.identifier))
        remaining = quantity
        allocations = []
        for lot in eligible:
            take = min(remaining, lot.quantity - self.reserved(lot.identifier))
            if take > 0:
                allocations.append((lot.identifier, take))
                remaining -= take
            if remaining == 0:
                break
        if remaining > 0:
            raise ValueError("insufficient stock valid through reservation")
        reservation = Reservation(identifier, sku, tuple(allocations), expires)
        self._reservations[identifier] = reservation
        self._used_ids.add(identifier)
        return reservation

    def release(self, identifier: str) -> Reservation:
        return self._reservations.pop(identifier)

    def ship(self, identifier: str) -> Reservation:
        reservation = self._reservations.pop(identifier)
        for lot_id, quantity in reservation.allocations:
            lot = self._lots[lot_id]
            self._lots[lot_id] = Lot(lot.identifier, lot.sku, lot.quantity - quantity, lot.expires)
        self._shipped[reservation.sku] = self._shipped.get(reservation.sku, 0) + reservation.quantity
        return reservation

    def advance(self, now: int) -> tuple[str, ...]:
        if now < self.now:
            raise ValueError("clock cannot move backward")
        self.now = now
        expired = sorted(identifier for identifier, reservation in self._reservations.items()
                         if reservation.expires <= now)
        for identifier in expired:
            self.release(identifier)
        return tuple(expired)

    def discard_expired(self) -> tuple[Lot, ...]:
        expired = tuple(lot for _, lot in sorted(self._lots.items()) if not self._live(lot))
        for lot in expired:
            if self.reserved(lot.identifier):
                raise RuntimeError("expired lot is reserved")
        for lot in expired:
            del self._lots[lot.identifier]
        return expired

    def lots(self, sku: str | None = None) -> tuple[Lot, ...]:
        return tuple(lot for _, lot in sorted(self._lots.items()) if sku is None or lot.sku == sku)

    def reservations(self) -> tuple[Reservation, ...]:
        return tuple(value for _, value in sorted(self._reservations.items()))

    def shipped(self, sku: str) -> int:
        return self._shipped.get(sku, 0)

    def adjust(self, lot_id: str, quantity: int) -> Lot:
        lot = self._lots[lot_id]
        if quantity < self.reserved(lot_id):
            raise ValueError("adjustment violates reservation")
        updated = Lot(lot_id, lot.sku, quantity, lot.expires)
        self._lots[lot_id] = updated
        return updated

    def transfer_sku(self, lot_id: str, sku: str) -> Lot:
        lot = self._lots[lot_id]
        if not sku or self.reserved(lot_id):
            raise ValueError("cannot relabel reserved lot or empty SKU")
        updated = Lot(lot_id, sku, lot.quantity, lot.expires)
        self._lots[lot_id] = updated
        return updated

    def audit(self) -> bool:
        for lot in self._lots.values():
            if lot.quantity < self.reserved(lot.identifier):
                return False
        for reservation in self._reservations.values():
            if reservation.expires <= self.now or reservation.quantity <= 0:
                return False
            for lot_id, quantity in reservation.allocations:
                lot = self._lots.get(lot_id)
                if lot is None or lot.sku != reservation.sku or quantity <= 0:
                    return False
        return True

    def summary(self) -> dict[str, dict[str, int]]:
        skus = sorted({lot.sku for lot in self._lots.values()} | set(self._shipped))
        return {sku: dict(on_hand=self.on_hand(sku), available=self.available(sku),
                          shipped=self.shipped(sku)) for sku in skus}


def reorder_quantity(available: int, target: int, pack_size: int = 1) -> int:
    """Order complete packs to reach target without ordering excess packs."""
    if available < 0 or target < 0 or pack_size <= 0:
        raise ValueError("invalid reorder policy")
    missing = max(0, target - available)
    return ((missing + pack_size - 1) // pack_size) * pack_size


def pick_list(reservations: Iterable[Reservation]) -> list[tuple[str, int]]:
    quantities: dict[str, int] = {}
    for reservation in reservations:
        for lot_id, quantity in reservation.allocations:
            quantities[lot_id] = quantities.get(lot_id, 0) + quantity
    return sorted(quantities.items())


def parse_delivery(text: str) -> list[tuple[str, int]]:
    """Parse SKU:quantity lines and combine repeated SKUs in input order."""
    quantities: dict[str, int] = {}
    for number, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            continue
        try:
            sku, value = line.split(":")
            sku = sku.strip()
            quantity = int(value)
            if not sku or quantity <= 0:
                raise ValueError("invalid delivery")
        except ValueError as error:
            raise ValueError(f"invalid delivery row {number}") from error
        quantities[sku] = quantities.get(sku, 0) + quantity
    return list(quantities.items())
