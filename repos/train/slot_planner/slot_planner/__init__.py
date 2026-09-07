"""Half-open integer-time reservations with deterministic scheduling."""
from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Iterable


@dataclass(frozen=True, order=True)
class Slot:
    start: int
    end: int

    def __post_init__(self) -> None:
        if self.start < 0 or self.end <= self.start:
            raise ValueError("slot must have positive duration and nonnegative start")

    @property
    def duration(self) -> int:
        return self.end - self.start

    def overlaps(self, other: Slot) -> bool:
        return self.start < other.end and other.start < self.end

    def contains(self, point: int) -> bool:
        return self.start <= point < self.end

    def intersection(self, other: Slot) -> Slot | None:
        start = max(self.start, other.start)
        end = min(self.end, other.end)
        return Slot(start, end) if start < end else None


def merge(slots: Iterable[Slot]) -> list[Slot]:
    """Merge overlaps and touching slots into their union."""
    result: list[Slot] = []
    for slot in sorted(slots):
        if result and slot.start <= result[-1].end:
            result[-1] = Slot(result[-1].start, max(result[-1].end, slot.end))
        else:
            result.append(slot)
    return result


def gaps(window: Slot, occupied: Iterable[Slot]) -> list[Slot]:
    """Return all free intervals clipped to window."""
    cursor = window.start
    result = []
    for slot in merge(occupied):
        clipped = slot.intersection(window)
        if clipped is None:
            continue
        if cursor < clipped.start:
            result.append(Slot(cursor, clipped.start))
        cursor = max(cursor, clipped.end)
    if cursor < window.end:
        result.append(Slot(cursor, window.end))
    return result


def parse_clock(value: str) -> int:
    """Parse HH:MM, accepting 24:00 only as the end of day."""
    try:
        hours, minutes = (int(part) for part in value.split(":"))
    except ValueError as error:
        raise ValueError("expected HH:MM") from error
    if hours == 24 and minutes == 0:
        return 1440
    if not 0 <= hours < 24 or not 0 <= minutes < 60:
        raise ValueError("clock outside day")
    return hours * 60 + minutes


def format_clock(minutes: int) -> str:
    if not 0 <= minutes <= 1440:
        raise ValueError("clock outside day")
    hours, minute = divmod(minutes, 60)
    return f"{hours:02d}:{minute:02d}"


@dataclass(frozen=True)
class Booking:
    identifier: str
    room: str
    slot: Slot
    attendees: int


class Planner:
    """Bookings are atomic; touching reservations do not conflict."""

    def __init__(self, capacities: dict[str, int]) -> None:
        if not capacities or any(not name or value <= 0 for name, value in capacities.items()):
            raise ValueError("rooms require positive capacity")
        self._capacities = dict(capacities)
        self._bookings: dict[str, Booking] = {}

    def rooms(self, attendees: int = 1) -> tuple[str, ...]:
        if attendees <= 0:
            raise ValueError("positive attendance required")
        return tuple(sorted(name for name, capacity in self._capacities.items() if capacity >= attendees))

    def conflicts(self, room: str, slot: Slot, excluding: str | None = None) -> tuple[Booking, ...]:
        if room not in self._capacities:
            raise KeyError(room)
        return tuple(sorted((booking for booking in self._bookings.values()
                             if booking.room == room and booking.identifier != excluding
                             and booking.slot.overlaps(slot)), key=lambda booking: booking.slot))

    def book(self, identifier: str, room: str, slot: Slot, attendees: int = 1) -> Booking:
        if not identifier or identifier in self._bookings:
            raise ValueError("new booking identifier required")
        if room not in self._capacities:
            raise KeyError(room)
        if attendees <= 0 or attendees > self._capacities[room]:
            raise ValueError("room capacity exceeded")
        if self.conflicts(room, slot):
            raise ValueError("room occupied")
        booking = Booking(identifier, room, slot, attendees)
        self._bookings[identifier] = booking
        return booking

    def cancel(self, identifier: str) -> Booking:
        return self._bookings.pop(identifier)

    def move(self, identifier: str, room: str, slot: Slot) -> Booking:
        previous = self._bookings[identifier]
        if room not in self._capacities:
            raise KeyError(room)
        if previous.attendees > self._capacities[room]:
            raise ValueError("room capacity exceeded")
        if self.conflicts(room, slot, excluding=identifier):
            raise ValueError("room occupied")
        updated = Booking(identifier, room, slot, previous.attendees)
        self._bookings[identifier] = updated
        return updated

    def agenda(self, room: str | None = None) -> tuple[Booking, ...]:
        if room is not None and room not in self._capacities:
            raise KeyError(room)
        return tuple(sorted((booking for booking in self._bookings.values()
                             if room is None or booking.room == room),
                            key=lambda booking: (booking.slot, booking.room, booking.identifier)))

    def availability(self, room: str, window: Slot) -> list[Slot]:
        return gaps(window, (booking.slot for booking in self.agenda(room)))

    def first_fit(self, window: Slot, duration: int, attendees: int = 1) -> tuple[str, Slot] | None:
        if duration <= 0:
            raise ValueError("positive duration required")
        candidates = []
        for room in self.rooms(attendees):
            for slot in self.availability(room, window):
                if slot.duration >= duration:
                    candidates.append((slot.start, room, Slot(slot.start, slot.start + duration)))
                    break
        if not candidates:
            return None
        _, room, slot = min(candidates)
        return room, slot

    def utilization(self, room: str, window: Slot) -> tuple[int, int]:
        free = sum(slot.duration for slot in self.availability(room, window))
        return window.duration - free, window.duration

    def book_series(self, prefix: str, room: str, first: Slot, period: int,
                    count: int, attendees: int = 1) -> tuple[Booking, ...]:
        """Reserve all occurrences or none, including identifier validation."""
        if period <= 0 or count <= 0:
            raise ValueError("positive period and count required")
        created = []
        try:
            for index in range(count):
                slot = Slot(first.start + index * period, first.end + index * period)
                booking = self.book(f"{prefix}-{index}", room, slot, attendees)
                created.append(booking)
        except (ValueError, KeyError):
            for booking in created:
                self.cancel(booking.identifier)
            raise
        return tuple(created)

    def cancel_series(self, prefix: str) -> tuple[Booking, ...]:
        """Cancel only exact numeric series suffixes, not similar prefixes."""
        identifiers = [identifier for identifier in self._bookings
                       if identifier.startswith(prefix + "-")
                       and identifier[len(prefix) + 1:].isdigit()]
        return tuple(self.cancel(identifier) for identifier in sorted(identifiers))

    def resize_room(self, room: str, capacity: int) -> None:
        if room not in self._capacities:
            raise KeyError(room)
        if capacity <= 0:
            raise ValueError("positive capacity required")
        if any(booking.attendees > capacity for booking in self.agenda(room)):
            raise ValueError("existing bookings exceed capacity")
        self._capacities[room] = capacity


def split(slot: Slot, duration: int) -> list[Slot]:
    """Split a slot into bounded pieces, retaining a short final piece."""
    if duration <= 0:
        raise ValueError("positive duration required")
    return [Slot(start, min(start + duration, slot.end))
            for start in range(slot.start, slot.end, duration)]


def common_free(window: Slot, schedules: Iterable[Iterable[Slot]]) -> list[Slot]:
    occupied = [slot for schedule in schedules for slot in schedule]
    return gaps(window, occupied)
