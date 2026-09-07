import pytest
from slot_planner import (
    Slot, merge, gaps, parse_clock, format_clock,
    Booking, Planner, split, common_free,
)


# --- Slot dataclass ---

class TestSlot:
    def test_duration(self):
        assert Slot(0, 10).duration == 10
        assert Slot(5, 15).duration == 10

    def test_invalid_negative_start(self):
        with pytest.raises(ValueError):
            Slot(-1, 5)

    def test_invalid_zero_duration(self):
        with pytest.raises(ValueError):
            Slot(5, 5)

    def test_invalid_negative_duration(self):
        with pytest.raises(ValueError):
            Slot(10, 5)

    def test_overlaps_true(self):
        assert Slot(0, 10).overlaps(Slot(5, 15)) is True

    def test_overlaps_symmetric(self):
        assert Slot(5, 15).overlaps(Slot(0, 10)) is True

    def test_overlaps_false_touching(self):
        assert Slot(0, 10).overlaps(Slot(10, 20)) is False

    def test_overlaps_false_disjoint(self):
        assert Slot(0, 10).overlaps(Slot(20, 30)) is False

    def test_overlaps_contained(self):
        assert Slot(0, 20).overlaps(Slot(5, 10)) is True

    def test_contains_start(self):
        assert Slot(5, 15).contains(5) is True

    def test_contains_mid(self):
        assert Slot(5, 15).contains(10) is True

    def test_contains_end_exclusive(self):
        assert Slot(5, 15).contains(15) is False

    def test_contains_before(self):
        assert Slot(5, 15).contains(4) is False

    def test_intersection_overlap(self):
        assert Slot(0, 10).intersection(Slot(5, 15)) == Slot(5, 10)

    def test_intersection_none_touching(self):
        assert Slot(0, 10).intersection(Slot(10, 20)) is None

    def test_intersection_none_disjoint(self):
        assert Slot(0, 10).intersection(Slot(20, 30)) is None

    def test_intersection_contained(self):
        assert Slot(0, 20).intersection(Slot(5, 10)) == Slot(5, 10)

    def test_ordering(self):
        assert Slot(0, 10) < Slot(1, 5)
        assert Slot(0, 5) < Slot(0, 10)


# --- merge ---

class TestMerge:
    def test_empty(self):
        assert merge([]) == []

    def test_single(self):
        assert merge([Slot(0, 10)]) == [Slot(0, 10)]

    def test_no_overlap(self):
        assert merge([Slot(0, 5), Slot(10, 15)]) == [Slot(0, 5), Slot(10, 15)]

    def test_touching(self):
        assert merge([Slot(0, 10), Slot(10, 20)]) == [Slot(0, 20)]

    def test_overlapping(self):
        assert merge([Slot(0, 15), Slot(10, 20)]) == [Slot(0, 20)]

    def test_out_of_order(self):
        assert merge([Slot(10, 20), Slot(0, 5)]) == [Slot(0, 5), Slot(10, 20)]

    def test_contained(self):
        assert merge([Slot(0, 20), Slot(5, 10)]) == [Slot(0, 20)]

    def test_three_way_merge(self):
        assert merge([Slot(0, 10), Slot(5, 15), Slot(10, 20)]) == [Slot(0, 20)]


# --- gaps ---

class TestGaps:
    def test_no_occupied(self):
        assert gaps(Slot(0, 100), []) == [Slot(0, 100)]

    def test_fully_occupied(self):
        assert gaps(Slot(0, 100), [Slot(0, 100)]) == []

    def test_front_occupied(self):
        assert gaps(Slot(0, 100), [Slot(0, 50)]) == [Slot(50, 100)]

    def test_back_occupied(self):
        assert gaps(Slot(0, 100), [Slot(50, 100)]) == [Slot(0, 50)]

    def test_middle_gap(self):
        assert gaps(Slot(0, 100), [Slot(0, 30), Slot(70, 100)]) == [Slot(30, 70)]

    def test_trailing_gap_after_occupied(self):
        result = gaps(Slot(0, 100), [Slot(0, 40)])
        assert result == [Slot(40, 100)]

    def test_occupied_outside_window(self):
        assert gaps(Slot(10, 20), [Slot(0, 5), Slot(25, 30)]) == [Slot(10, 20)]

    def test_multiple_gaps(self):
        result = gaps(Slot(0, 100), [Slot(10, 20), Slot(50, 60)])
        assert result == [Slot(0, 10), Slot(20, 50), Slot(60, 100)]


# --- parse_clock ---

class TestParseClock:
    def test_midnight(self):
        assert parse_clock("00:00") == 0

    def test_noon(self):
        assert parse_clock("12:00") == 720

    def test_end_of_day(self):
        assert parse_clock("24:00") == 1440

    def test_arbitrary(self):
        assert parse_clock("09:30") == 570

    def test_one_thirty(self):
        assert parse_clock("01:30") == 90

    def test_invalid_format(self):
        with pytest.raises(ValueError, match="expected HH:MM"):
            parse_clock("abc")

    def test_invalid_non_numeric(self):
        with pytest.raises(ValueError, match="expected HH:MM"):
            parse_clock("xx:yy")

    def test_out_of_range_hour(self):
        with pytest.raises(ValueError, match="clock outside day"):
            parse_clock("25:00")

    def test_out_of_range_minute(self):
        with pytest.raises(ValueError, match="clock outside day"):
            parse_clock("12:60")


# --- format_clock ---

class TestFormatClock:
    def test_midnight(self):
        assert format_clock(0) == "00:00"

    def test_noon(self):
        assert format_clock(720) == "12:00"

    def test_end_of_day(self):
        assert format_clock(1440) == "24:00"

    def test_ninety_minutes(self):
        assert format_clock(90) == "01:30"

    def test_single_digit_minute(self):
        assert format_clock(5) == "00:05"

    def test_round_trip(self):
        for minutes in (0, 1, 59, 60, 90, 719, 720, 1439, 1440):
            assert parse_clock(format_clock(minutes)) == minutes

    def test_out_of_range_high(self):
        with pytest.raises(ValueError):
            format_clock(1441)

    def test_out_of_range_negative(self):
        with pytest.raises(ValueError):
            format_clock(-1)


# --- Planner lifecycle ---

class TestPlanner:
    def test_init_empty(self):
        with pytest.raises(ValueError):
            Planner({})

    def test_init_zero_capacity(self):
        with pytest.raises(ValueError):
            Planner({"A": 0})

    def test_rooms_filter_by_attendees(self):
        p = Planner({"small": 2, "big": 10})
        assert p.rooms(5) == ("big",)

    def test_rooms_all(self):
        p = Planner({"small": 2, "big": 10})
        assert set(p.rooms(1)) == {"small", "big"}

    def test_rooms_invalid_attendees(self):
        p = Planner({"A": 5})
        with pytest.raises(ValueError):
            p.rooms(0)

    def test_book_and_cancel(self):
        p = Planner({"A": 5})
        b = p.book("x", "A", Slot(0, 60))
        assert b.identifier == "x"
        assert b.room == "A"
        assert b.slot == Slot(0, 60)
        assert b.attendees == 1
        cancelled = p.cancel("x")
        assert cancelled.identifier == "x"
        assert p.agenda() == ()

    def test_book_conflict(self):
        p = Planner({"A": 5})
        p.book("x", "A", Slot(0, 60))
        with pytest.raises(ValueError, match="room occupied"):
            p.book("y", "A", Slot(30, 90))

    def test_book_touching_no_conflict(self):
        p = Planner({"A": 5})
        p.book("x", "A", Slot(0, 60))
        b = p.book("y", "A", Slot(60, 120))
        assert b.identifier == "y"

    def test_book_duplicate_id(self):
        p = Planner({"A": 5})
        p.book("x", "A", Slot(0, 60))
        with pytest.raises(ValueError):
            p.book("x", "A", Slot(100, 200))

    def test_book_unknown_room(self):
        p = Planner({"A": 5})
        with pytest.raises(KeyError):
            p.book("x", "Z", Slot(0, 60))

    def test_book_capacity_exceeded(self):
        p = Planner({"A": 5})
        with pytest.raises(ValueError):
            p.book("x", "A", Slot(0, 60), attendees=6)

    def test_conflicts(self):
        p = Planner({"A": 5})
        p.book("x", "A", Slot(0, 60))
        assert len(p.conflicts("A", Slot(30, 90))) == 1
        assert len(p.conflicts("A", Slot(60, 120))) == 0

    def test_conflicts_excluding(self):
        p = Planner({"A": 5})
        p.book("x", "A", Slot(0, 60))
        assert len(p.conflicts("A", Slot(30, 90), excluding="x")) == 0

    def test_move(self):
        p = Planner({"A": 5, "B": 5})
        p.book("x", "A", Slot(0, 60))
        moved = p.move("x", "B", Slot(100, 160))
        assert moved.room == "B"
        assert moved.slot == Slot(100, 160)

    def test_agenda_sorted_by_time(self):
        p = Planner({"A": 10})
        p.book("z", "A", Slot(0, 30))
        p.book("a", "A", Slot(60, 90))
        result = p.agenda()
        assert result[0].identifier == "z"
        assert result[1].identifier == "a"

    def test_agenda_room_filter(self):
        p = Planner({"A": 5, "B": 5})
        p.book("x", "A", Slot(0, 60))
        p.book("y", "B", Slot(0, 60))
        assert len(p.agenda("A")) == 1
        assert len(p.agenda("B")) == 1

    def test_agenda_unknown_room(self):
        p = Planner({"A": 5})
        with pytest.raises(KeyError):
            p.agenda("Z")

    def test_availability(self):
        p = Planner({"A": 10})
        p.book("x", "A", Slot(20, 40))
        avail = p.availability("A", Slot(0, 60))
        assert avail == [Slot(0, 20), Slot(40, 60)]

    def test_first_fit(self):
        p = Planner({"A": 10})
        p.book("x", "A", Slot(0, 30))
        result = p.first_fit(Slot(0, 120), 30)
        assert result is not None
        room, slot = result
        assert room == "A"
        assert slot == Slot(30, 60)

    def test_first_fit_no_space(self):
        p = Planner({"A": 10})
        p.book("x", "A", Slot(0, 100))
        assert p.first_fit(Slot(0, 100), 30) is None

    def test_first_fit_picks_earliest(self):
        p = Planner({"A": 10, "B": 10})
        p.book("x", "A", Slot(0, 50))
        result = p.first_fit(Slot(0, 100), 30)
        assert result is not None
        room, slot = result
        assert slot.start == 0  # B is free from 0

    def test_utilization(self):
        p = Planner({"A": 10})
        p.book("x", "A", Slot(0, 40))
        used, total = p.utilization("A", Slot(0, 100))
        assert (used, total) == (40, 100)

    def test_utilization_empty(self):
        p = Planner({"A": 10})
        used, total = p.utilization("A", Slot(0, 100))
        assert (used, total) == (0, 100)

    def test_book_series(self):
        p = Planner({"A": 10})
        series = p.book_series("mtg", "A", Slot(0, 60), 1440, 3)
        assert len(series) == 3
        assert series[0].slot == Slot(0, 60)
        assert series[1].slot == Slot(1440, 1500)
        assert series[2].slot == Slot(2880, 2940)
        assert series[0].identifier == "mtg-0"

    def test_book_series_rollback_on_conflict(self):
        p = Planner({"A": 10})
        p.book("block", "A", Slot(1440, 1500))
        with pytest.raises(ValueError):
            p.book_series("mtg", "A", Slot(0, 60), 1440, 3)
        # only the blocker should remain
        assert len(p.agenda()) == 1

    def test_cancel_series(self):
        p = Planner({"A": 10})
        p.book_series("mtg", "A", Slot(0, 60), 1440, 3)
        cancelled = p.cancel_series("mtg")
        assert len(cancelled) == 3
        assert p.agenda() == ()

    def test_cancel_series_no_false_prefix_match(self):
        p = Planner({"A": 10})
        p.book("mtg-extra", "A", Slot(0, 60))
        p.book("mtg-0", "A", Slot(100, 200))
        cancelled = p.cancel_series("mtg")
        assert len(cancelled) == 1
        assert cancelled[0].identifier == "mtg-0"

    def test_resize_room(self):
        p = Planner({"A": 10})
        p.resize_room("A", 20)
        b = p.book("x", "A", Slot(0, 60), attendees=15)
        assert b.attendees == 15

    def test_resize_room_too_small(self):
        p = Planner({"A": 10})
        p.book("x", "A", Slot(0, 60), attendees=5)
        with pytest.raises(ValueError, match="existing bookings exceed"):
            p.resize_room("A", 3)

    def test_resize_room_unknown(self):
        p = Planner({"A": 10})
        with pytest.raises(KeyError):
            p.resize_room("Z", 5)


# --- split ---

class TestSplit:
    def test_exact_division(self):
        result = split(Slot(0, 60), 30)
        assert result == [Slot(0, 30), Slot(30, 60)]

    def test_remainder(self):
        result = split(Slot(0, 50), 30)
        assert result == [Slot(0, 30), Slot(30, 50)]

    def test_single_piece(self):
        result = split(Slot(0, 10), 30)
        assert result == [Slot(0, 10)]

    def test_duration_equals_slot(self):
        result = split(Slot(0, 30), 30)
        assert result == [Slot(0, 30)]

    def test_invalid_duration(self):
        with pytest.raises(ValueError):
            split(Slot(0, 60), 0)


# --- common_free ---

class TestCommonFree:
    def test_all_free(self):
        assert common_free(Slot(0, 100), [[], []]) == [Slot(0, 100)]

    def test_no_common(self):
        result = common_free(Slot(0, 100), [[Slot(0, 50)], [Slot(50, 100)]])
        assert result == []

    def test_partial_overlap(self):
        result = common_free(Slot(0, 100), [[Slot(0, 30)], [Slot(70, 100)]])
        assert result == [Slot(30, 70)]

    def test_single_schedule(self):
        result = common_free(Slot(0, 100), [[Slot(20, 40)]])
        assert result == [Slot(0, 20), Slot(40, 100)]

    def test_empty_schedules(self):
        assert common_free(Slot(0, 60), []) == [Slot(0, 60)]
