"""Generate property-based hidden tests for each micro-repo.

Hidden tests are a strengthened superset of visible tests: same functions,
hypothesis-driven inputs, and more boundary cases.  They live in each repo's
hidden_tests/ directory and are never visible to the agent.
"""
from __future__ import annotations

from pathlib import Path
from textwrap import dedent

REPO_GENERATORS: dict[str, str] = {}


def _register(name: str, code: str) -> None:
    REPO_GENERATORS[name] = dedent(code).lstrip()


def generate(repo_name: str, repo_root: Path) -> Path:
    if repo_name not in REPO_GENERATORS:
        raise KeyError(f"no hidden-test generator for repo {repo_name!r}")
    dest = repo_root / "hidden_tests"
    dest.mkdir(exist_ok=True)
    (dest / "__init__.py").write_text("")
    (dest / f"test_hidden_{repo_name}.py").write_text(REPO_GENERATORS[repo_name])
    return dest


# ── parcel_ledger ──────────────────────────────────────────────────────

_register("parcel_ledger", """
    from hypothesis import given, settings, assume
    from hypothesis.strategies import integers, text, lists
    from parcel_ledger.billing_rules import (
        discount, invoice_ids, payer_amount, fee_total,
        parse_quantity, prioritize_amounts, first_reference,
    )

    @given(amount=integers(0, 10_000), threshold=integers(1, 10_000))
    def test_discount_threshold(amount, threshold):
        result = discount(amount, threshold)
        if amount >= threshold:
            assert result == amount // 10
        else:
            assert result == 0

    @given(start=integers(0, 1000), count=integers(0, 100))
    def test_invoice_ids_length(start, count):
        ids = invoice_ids(start, count)
        assert len(ids) == count
        assert ids == list(range(start, start + count))

    @given(charged=integers(0, 10_000), refunded=integers(0, 10_000))
    def test_payer_returns_charged(charged, refunded):
        assert payer_amount(charged, refunded) == charged

    @given(base=integers(0, 10_000), surcharge=integers(0, 10_000))
    def test_fee_is_sum(base, surcharge):
        assert fee_total(base, surcharge) == base + surcharge

    @given(fallback=integers(-100, 100))
    def test_parse_bad_text(fallback):
        assert parse_quantity("not_a_number", fallback) == fallback

    @given(n=integers(0, 50))
    def test_parse_valid_text(n):
        assert parse_quantity(str(n), -1) == n

    @given(amounts=lists(integers(-1000, 1000), min_size=1, max_size=20))
    def test_prioritize_descending(amounts):
        result = prioritize_amounts(amounts)
        assert result == sorted(amounts, reverse=True)

    @given(default=text(min_size=0, max_size=10))
    def test_first_reference_empty(default):
        assert first_reference([], default) == default

    @given(refs=lists(text(min_size=1, max_size=5), min_size=1, max_size=10),
           default=text(min_size=0, max_size=5))
    def test_first_reference_nonempty(refs, default):
        assert first_reference(refs, default) == refs[0]
""")


# ── record_index ───────────────────────────────────────────────────────

_register("record_index", """
    from hypothesis import given, settings, assume
    from hypothesis.strategies import (
        text, integers, lists, sampled_from, just, tuples,
    )
    from record_index import (
        tokenize, positions, excerpt, parse_query, paginate,
        concordance, filtered_search, Index,
    )

    @given(t=text(min_size=0, max_size=200))
    def test_tokenize_lowercase(t):
        for token in tokenize(t):
            assert token == token.casefold()

    @given(t=text(min_size=1, max_size=100))
    def test_positions_roundtrip(t):
        tokens = tokenize(t)
        pos = positions(t)
        total = sum(len(v) for v in pos.values())
        assert total == len(tokens)

    @given(width=integers(1, 20))
    def test_excerpt_bounded(width):
        text_val = "the quick brown fox jumps over the lazy dog"
        result = excerpt(text_val, "fox", width)
        assert len(result.split()) <= width

    @given(page=integers(1, 10), size=integers(1, 5))
    def test_paginate_bounds(page, size):
        items = [str(i) for i in range(20)]
        result, has_more = paginate(items, page, size)
        assert len(result) <= size
        if (page - 1) * size < len(items):
            assert len(result) > 0

    def test_index_add_remove_roundtrip():
        idx = Index()
        idx.add("d1", "hello world", "body text", ["tag1"])
        assert len(idx) == 1
        assert idx.get("d1").title == "hello world"
        idx.remove("d1")
        assert len(idx) == 0

    def test_search_all_mode():
        idx = Index()
        idx.add("d1", "alpha beta", "gamma")
        idx.add("d2", "alpha", "delta")
        assert "d1" in idx.search("alpha beta", mode="all")
        assert "d2" not in idx.search("alpha beta", mode="all")

    def test_search_any_mode():
        idx = Index()
        idx.add("d1", "alpha", "gamma")
        idx.add("d2", "beta", "delta")
        results = idx.search("alpha beta", mode="any")
        assert "d1" in results
        assert "d2" in results

    def test_search_with_tag_filter():
        idx = Index()
        idx.add("d1", "alpha", "body", ["important"])
        idx.add("d2", "alpha", "body", ["trivial"])
        results = idx.search("alpha", tag="important")
        assert results == ["d1"]

    def test_phrase_search():
        idx = Index()
        idx.add("d1", "the quick brown fox", "body")
        idx.add("d2", "brown the quick fox", "body")
        assert "d1" in idx.phrase("quick brown")
        assert "d2" not in idx.phrase("quick brown")

    def test_export_import_roundtrip():
        idx = Index()
        idx.add("a", "title a", "body a", ["t1"])
        idx.add("b", "title b", "body b", ["t2"])
        exported = idx.export()
        restored = Index.from_rows(exported)
        assert restored.export() == exported
        assert restored.audit()

    @given(query=text(min_size=1, max_size=30))
    def test_parse_query_partitions(query):
        include, exclude = parse_query(query)
        for word in include:
            assert not word.startswith("-")
        for word in exclude:
            assert not word.startswith("-")

    def test_related_excludes_self():
        idx = Index()
        idx.add("a", "shared words here", "body")
        idx.add("b", "shared words there", "body")
        related = idx.related("a")
        assert "a" not in related

    def test_vocabulary_prefix():
        idx = Index()
        idx.add("d1", "apple application", "body")
        vocab = idx.vocabulary("app")
        assert all(w.startswith("app") for w in vocab)
        assert len(vocab) >= 2

    @given(n=integers(0, 10))
    def test_search_limit(n):
        idx = Index()
        for i in range(15):
            idx.add(f"d{i}", "shared", f"body {i}")
        results = idx.search("shared", limit=n)
        assert len(results) <= n
""")


# ── slot_planner ───────────────────────────────────────────────────────

_register("slot_planner", """
    from hypothesis import given, settings, assume
    from hypothesis.strategies import integers, lists, tuples
    from slot_planner import (
        Slot, merge, gaps, parse_clock, format_clock, split,
        common_free, Planner,
    )

    @given(s=integers(0, 1000), d=integers(1, 100))
    def test_slot_duration(s, d):
        slot = Slot(s, s + d)
        assert slot.duration == d

    @given(s=integers(0, 100), d=integers(1, 50))
    def test_slot_contains_boundaries(s, d):
        slot = Slot(s, s + d)
        assert slot.contains(s)
        assert not slot.contains(s + d)

    @given(s1=integers(0, 50), d1=integers(1, 30),
           s2=integers(0, 50), d2=integers(1, 30))
    def test_overlap_symmetric(s1, d1, s2, d2):
        a = Slot(s1, s1 + d1)
        b = Slot(s2, s2 + d2)
        assert a.overlaps(b) == b.overlaps(a)

    @given(s=integers(0, 50), d=integers(1, 30))
    def test_self_overlap(s, d):
        slot = Slot(s, s + d)
        assert slot.overlaps(slot)

    def test_merge_touching():
        assert merge([Slot(0, 5), Slot(5, 10)]) == [Slot(0, 10)]

    def test_merge_gap():
        assert merge([Slot(0, 3), Slot(5, 8)]) == [Slot(0, 3), Slot(5, 8)]

    def test_gaps_full_window():
        result = gaps(Slot(0, 100), [])
        assert result == [Slot(0, 100)]

    def test_gaps_partial():
        result = gaps(Slot(0, 100), [Slot(20, 40)])
        assert result == [Slot(0, 20), Slot(40, 100)]

    @given(h=integers(0, 23), m=integers(0, 59))
    def test_clock_roundtrip(h, m):
        minutes = h * 60 + m
        assert parse_clock(format_clock(minutes)) == minutes

    def test_clock_midnight():
        assert parse_clock("00:00") == 0
        assert parse_clock("24:00") == 1440

    @given(d=integers(1, 20))
    def test_split_covers_slot(d):
        slot = Slot(0, 50)
        pieces = split(slot, d)
        assert pieces[0].start == 0
        assert pieces[-1].end == 50
        for a, b in zip(pieces, pieces[1:]):
            assert a.end == b.start

    def test_planner_book_cancel():
        p = Planner({"A": 10})
        p.book("b1", "A", Slot(0, 60))
        assert len(p.agenda()) == 1
        p.cancel("b1")
        assert len(p.agenda()) == 0

    def test_planner_conflict():
        p = Planner({"A": 10})
        p.book("b1", "A", Slot(0, 60))
        assert len(p.conflicts("A", Slot(30, 90))) == 1

    def test_planner_no_double_book():
        p = Planner({"A": 10})
        p.book("b1", "A", Slot(0, 60))
        try:
            p.book("b2", "A", Slot(30, 90))
            assert False, "should have raised"
        except ValueError:
            pass

    def test_first_fit():
        p = Planner({"A": 5, "B": 10})
        p.book("b1", "A", Slot(0, 60))
        result = p.first_fit(Slot(0, 120), 30, attendees=1)
        assert result is not None
        room, slot = result
        assert slot.duration == 30

    def test_utilization():
        p = Planner({"A": 10})
        p.book("b1", "A", Slot(0, 50))
        used, total = p.utilization("A", Slot(0, 100))
        assert used == 50
        assert total == 100

    def test_common_free():
        result = common_free(Slot(0, 100), [[Slot(10, 30)], [Slot(20, 50)]])
        assert all(s.start >= 0 and s.end <= 100 for s in result)
""")


# ── route_graph ────────────────────────────────────────────────────────

_register("route_graph", """
    from hypothesis import given, settings, assume
    from hypothesis.strategies import (
        integers, text, lists, sampled_from, just,
    )
    from route_graph import Graph, itinerary

    def test_add_and_list_nodes():
        g = Graph()
        g.add_node("a")
        g.add_node("b")
        assert g.nodes() == ("a", "b")

    def test_connect_and_neighbors():
        g = Graph()
        g.connect("a", "b", 5)
        assert g.neighbors("a") == (("b", 5),)

    def test_bidirectional():
        g = Graph()
        g.connect("a", "b", 3, bidirectional=True)
        assert ("b", 3) in g.neighbors("a")
        assert ("a", 3) in g.neighbors("b")

    def test_shortest_path_simple():
        g = Graph()
        g.connect("a", "b", 1)
        g.connect("b", "c", 2)
        g.connect("a", "c", 10)
        cost, path = g.shortest("a", "c")
        assert cost == 3
        assert path == ("a", "b", "c")

    def test_shortest_unreachable():
        g = Graph()
        g.add_node("a")
        g.add_node("b")
        assert g.shortest("a", "b") is None

    def test_shortest_same_node():
        g = Graph()
        g.add_node("a")
        cost, path = g.shortest("a", "a")
        assert cost == 0
        assert path == ("a",)

    def test_reachable_with_blocked():
        g = Graph()
        g.connect("a", "b", 1)
        g.connect("b", "c", 1)
        assert "c" not in g.reachable("a", blocked=["b"])
        assert "c" in g.reachable("a")

    def test_path_cost():
        g = Graph()
        g.connect("a", "b", 3)
        g.connect("b", "c", 7)
        assert g.path_cost(["a", "b", "c"]) == 10

    def test_reverse():
        g = Graph()
        g.connect("a", "b", 5)
        r = g.reverse()
        assert r.neighbors("b") == (("a", 5),)
        assert r.neighbors("a") == ()

    def test_induced_subgraph():
        g = Graph()
        g.connect("a", "b", 1)
        g.connect("b", "c", 2)
        g.connect("c", "d", 3)
        sub = g.induced(["a", "b"])
        assert sub.nodes() == ("a", "b")
        assert ("d",) not in [e[:1] for e in sub.edges()]

    def test_topological_sort():
        g = Graph()
        g.connect("a", "b", 1)
        g.connect("b", "c", 1)
        order = g.topological()
        assert order.index("a") < order.index("b") < order.index("c")

    def test_topological_cycle_raises():
        g = Graph()
        g.connect("a", "b", 1)
        g.connect("b", "a", 1)
        try:
            g.topological()
            assert False, "should raise"
        except ValueError:
            pass

    def test_components():
        g = Graph()
        g.connect("a", "b", 1)
        g.add_node("c")
        comps = g.components()
        assert len(comps) == 2

    def test_within_budget():
        g = Graph()
        g.connect("a", "b", 3)
        g.connect("b", "c", 4)
        result = g.within("a", 5)
        assert "a" in result
        assert "b" in result
        assert "c" not in result

    def test_serialization_roundtrip():
        g = Graph()
        g.connect("x", "y", 7)
        g.add_node("z")
        rows = g.to_rows()
        restored = Graph.from_rows(rows)
        assert restored.nodes() == g.nodes()
        assert restored.edges() == g.edges()

    def test_itinerary():
        g = Graph()
        g.connect("a", "b", 2)
        g.connect("b", "c", 3)
        cost, path = itinerary(g, ["a", "b", "c"])
        assert cost == 5
        assert path == ("a", "b", "c")

    def test_itinerary_unreachable():
        g = Graph()
        g.add_node("a")
        g.add_node("b")
        assert itinerary(g, ["a", "b"]) is None

    def test_disconnect():
        g = Graph()
        g.connect("a", "b", 5)
        g.disconnect("a", "b")
        assert g.neighbors("a") == ()

    def test_remove_node():
        g = Graph()
        g.connect("a", "b", 1)
        g.connect("b", "c", 1)
        g.remove_node("b")
        assert "b" not in g.nodes()
""")


# ── stock_reservations ─────────────────────────────────────────────────

_register("stock_reservations", """
    from hypothesis import given, settings, assume
    from hypothesis.strategies import integers, text, lists
    from stock_reservations import (
        Lot, Reservation, Warehouse, reorder_quantity,
        pick_list, parse_delivery,
    )

    @given(available=integers(0, 100), target=integers(0, 200),
           pack=integers(1, 20))
    def test_reorder_reaches_target(available, target, pack):
        qty = reorder_quantity(available, target, pack)
        assert qty >= 0
        assert qty % pack == 0
        assert available + qty >= target

    @given(available=integers(0, 100), target=integers(0, 200),
           pack=integers(1, 20))
    def test_reorder_minimal(available, target, pack):
        qty = reorder_quantity(available, target, pack)
        if qty > 0:
            assert available + qty - pack < target

    def test_receive_and_available():
        w = Warehouse()
        w.receive("L1", "SKU-A", 100)
        assert w.available("SKU-A") == 100
        assert w.on_hand("SKU-A") == 100

    def test_reserve_reduces_available():
        w = Warehouse()
        w.receive("L1", "SKU-A", 100)
        w.reserve("R1", "SKU-A", 30, ttl=10)
        assert w.available("SKU-A") == 70
        assert w.on_hand("SKU-A") == 100

    def test_ship_reduces_on_hand():
        w = Warehouse()
        w.receive("L1", "SKU-A", 100)
        w.reserve("R1", "SKU-A", 30, ttl=10)
        w.ship("R1")
        assert w.on_hand("SKU-A") == 70
        assert w.shipped("SKU-A") == 30

    def test_release_restores_available():
        w = Warehouse()
        w.receive("L1", "SKU-A", 100)
        w.reserve("R1", "SKU-A", 30, ttl=10)
        w.release("R1")
        assert w.available("SKU-A") == 100

    def test_advance_expires_reservations():
        w = Warehouse()
        w.receive("L1", "SKU-A", 100)
        w.reserve("R1", "SKU-A", 30, ttl=5)
        expired = w.advance(10)
        assert "R1" in expired
        assert w.available("SKU-A") == 100

    def test_lot_expiry():
        w = Warehouse(now=0)
        w.receive("L1", "SKU-A", 50, expires=5)
        assert w.available("SKU-A") == 50
        w.advance(5)
        assert w.available("SKU-A") == 0
        assert w.on_hand("SKU-A", include_expired=True) == 50

    def test_discard_expired():
        w = Warehouse(now=0)
        w.receive("L1", "SKU-A", 50, expires=5)
        w.advance(5)
        discarded = w.discard_expired()
        assert len(discarded) == 1
        assert w.on_hand("SKU-A") == 0

    def test_adjust_lot():
        w = Warehouse()
        w.receive("L1", "SKU-A", 100)
        w.adjust("L1", 200)
        assert w.available("SKU-A") == 200

    def test_transfer_sku():
        w = Warehouse()
        w.receive("L1", "SKU-A", 100)
        w.transfer_sku("L1", "SKU-B")
        assert w.available("SKU-A") == 0
        assert w.available("SKU-B") == 100

    def test_audit_clean():
        w = Warehouse()
        w.receive("L1", "SKU-A", 100)
        w.reserve("R1", "SKU-A", 30, ttl=10)
        assert w.audit()

    def test_summary():
        w = Warehouse()
        w.receive("L1", "SKU-A", 100)
        w.reserve("R1", "SKU-A", 30, ttl=10)
        w.ship("R1")
        s = w.summary()
        assert s["SKU-A"]["on_hand"] == 70
        assert s["SKU-A"]["shipped"] == 30

    def test_parse_delivery():
        result = parse_delivery("SKU-A: 10\\nSKU-B: 20")
        assert ("SKU-A", 10) in result
        assert ("SKU-B", 20) in result

    def test_parse_delivery_combines():
        result = parse_delivery("SKU-A: 10\\nSKU-A: 5")
        assert dict(result)["SKU-A"] == 15

    def test_pick_list():
        r1 = Reservation("R1", "SKU", (("L1", 5), ("L2", 3)), expires=10)
        r2 = Reservation("R2", "SKU", (("L1", 2),), expires=10)
        picks = pick_list([r1, r2])
        assert dict(picks)["L1"] == 7
        assert dict(picks)["L2"] == 3

    def test_reserve_fefo_order():
        w = Warehouse(now=0)
        w.receive("L1", "SKU-A", 50, expires=10)
        w.receive("L2", "SKU-A", 50, expires=5)
        r = w.reserve("R1", "SKU-A", 30, ttl=3)
        assert r.allocations[0][0] == "L2"

    def test_insufficient_stock():
        w = Warehouse()
        w.receive("L1", "SKU-A", 10)
        try:
            w.reserve("R1", "SKU-A", 20, ttl=5)
            assert False, "should raise"
        except ValueError:
            pass

    def test_id_reuse_blocked():
        w = Warehouse()
        w.receive("L1", "SKU-A", 100)
        w.reserve("R1", "SKU-A", 10, ttl=5)
        w.release("R1")
        try:
            w.reserve("R1", "SKU-A", 10, ttl=5)
            assert False, "should raise"
        except ValueError:
            pass
""")


# ── config_parser ──────────────────────────────────────────────────────

_register("config_parser", """
    from hypothesis import given, settings, assume
    from hypothesis.strategies import (
        text, integers, booleans, dictionaries, fixed_dictionaries,
        from_regex, just, one_of,
    )
    from config_parser import (
        get_path, set_path, delete_path, flatten, unflatten,
        merge, diff, validate, parse_env, to_env, interpolate,
        select, coerce, ConfigError,
    )

    def test_get_set_roundtrip():
        d = {}
        set_path(d, "a.b.c", 42)
        assert get_path(d, "a.b.c") == 42

    def test_get_missing_default():
        assert get_path({}, "x.y", default="fallback") == "fallback"

    def test_get_missing_raises():
        try:
            get_path({}, "x.y")
            assert False
        except KeyError:
            pass

    def test_delete_path():
        d = {"a": {"b": 1, "c": 2}}
        deleted = delete_path(d, "a.b")
        assert deleted == 1
        assert "b" not in d["a"]

    def test_flatten_unflatten_roundtrip():
        original = {"a": {"b": 1, "c": {"d": 2}}, "e": 3}
        assert unflatten(flatten(original)) == original

    @given(k=from_regex(r"[a-z]{1,5}", fullmatch=True),
           v=integers(0, 100))
    def test_flatten_single(k, v):
        flat = flatten({k: v})
        assert flat == {k: v}

    def test_merge_deep():
        base = {"a": {"b": 1, "c": 2}}
        override = {"a": {"c": 3, "d": 4}}
        result = merge(base, override)
        assert result == {"a": {"b": 1, "c": 3, "d": 4}}

    def test_merge_does_not_mutate():
        base = {"a": 1}
        override = {"b": 2}
        merge(base, override)
        assert "b" not in base

    def test_diff_detects_changes():
        old = {"a": 1, "b": 2}
        new = {"a": 1, "b": 3, "c": 4}
        changes = diff(old, new)
        assert changes["b"] == (2, 3)
        assert changes["c"] == (None, 4)

    def test_validate_passes():
        data = {"name": "test", "count": 5}
        schema = {"name": "str", "count": ["int", "positive"]}
        assert validate(data, schema) == []

    def test_validate_fails():
        data = {"count": -1}
        schema = {"count": ["int", "positive"]}
        errors = validate(data, schema)
        assert any("positive" in e for e in errors)

    def test_validate_missing():
        errors = validate({}, {"required_field": "str"})
        assert any("missing" in e for e in errors)

    def test_parse_env():
        text = 'KEY=value\\nNAME="hello world"'
        result = parse_env(text)
        assert result["KEY"] == "value"
        assert result["NAME"] == "hello world"

    def test_parse_env_comments():
        result = parse_env("# comment\\nKEY=val")
        assert result == {"KEY": "val"}

    def test_to_env_roundtrip():
        data = {"db": {"host": "localhost", "port": 5432}}
        env_text = to_env(data)
        assert "DB_HOST=localhost" in env_text
        assert "DB_PORT=5432" in env_text

    def test_interpolate():
        data = {"host": "localhost", "url": "http://${host}:8080"}
        result = interpolate(data)
        assert get_path(result, "url") == "http://localhost:8080"

    def test_interpolate_with_context():
        data = {"url": "http://${host}"}
        context = {"host": "example.com"}
        result = interpolate(data, context)
        assert get_path(result, "url") == "http://example.com"

    def test_select():
        data = {"a": 1, "b": {"c": 2}, "d": 3}
        result = select(data, ["a", "b.c"])
        assert result == {"a": 1, "b": {"c": 2}}

    @given(v=one_of(just("true"), just("false"), just("42"),
                    just("3.14"), just("hello")))
    def test_coerce_deterministic(v):
        result = coerce(v)
        assert coerce(v) == result

    def test_coerce_types():
        assert coerce("true") is True
        assert coerce("false") is False
        assert coerce("42") == 42
        assert coerce("3.14") == 3.14
        assert coerce("hello") == "hello"
""")


# ── metric_aggregator ─────────────────────────────────────────────────

_register("metric_aggregator", """
    from hypothesis import given, settings, assume
    from hypothesis.strategies import (
        integers, floats, lists,
    )
    import math
    from metric_aggregator import (
        Metric, WindowedMetric, AlertRule, MetricRegistry,
        rate, bucket_histogram, ewma,
    )

    @given(values=lists(floats(min_value=-1e6, max_value=1e6,
                               allow_nan=False, allow_infinity=False),
                        min_size=1, max_size=50))
    def test_metric_mean_bounded(values):
        m = Metric(values)
        # total()/count() can land an ulp outside [min, max] on correct code;
        # hypothesis finds such a list at some seeds, so the bound needs a
        # relative tolerance or the property rejects a correct implementation.
        tol = 1e-9 * max(1.0, abs(m.minimum()), abs(m.maximum()))
        assert m.minimum() - tol <= m.mean() <= m.maximum() + tol

    @given(values=lists(floats(min_value=-1e6, max_value=1e6,
                               allow_nan=False, allow_infinity=False),
                        min_size=2, max_size=50))
    def test_metric_variance_nonneg(values):
        m = Metric(values)
        assert m.variance() >= 0

    def test_metric_percentile_boundaries():
        m = Metric([1, 2, 3, 4, 5])
        assert m.percentile(0) == 1
        assert m.percentile(100) == 5
        assert 2 <= m.percentile(50) <= 3

    def test_metric_merge():
        a = Metric([1, 3, 5])
        b = Metric([2, 4])
        merged = a.merge(b)
        assert merged.count() == 5
        assert merged.minimum() == 1
        assert merged.maximum() == 5

    def test_metric_trim():
        m = Metric([1, 2, 3, 4, 5, 100])
        trimmed = m.trim(1, 10)
        assert trimmed.maximum() <= 10

    def test_metric_reset():
        m = Metric([1, 2, 3])
        count = m.reset()
        assert count == 3
        assert m.count() == 0

    def test_metric_empty_summary():
        m = Metric()
        assert m.summary() == {"count": 0}

    def test_windowed_metric():
        w = WindowedMetric(window_size=3)
        for v in [1, 2, 3, 4, 5]:
            w.record(v)
        assert w.count() == 3
        snap = w.snapshot()
        assert snap.count() == 3

    def test_alert_rule_consecutive():
        rule = AlertRule("high_cpu", threshold=90, comparator="gt", consecutive=3)
        assert not rule.evaluate(95)
        assert not rule.evaluate(95)
        assert rule.evaluate(95)

    def test_alert_rule_reset_on_normal():
        rule = AlertRule("high_cpu", threshold=90, comparator="gt", consecutive=2)
        rule.evaluate(95)
        rule.evaluate(50)
        assert not rule.evaluate(95)

    def test_registry():
        reg = MetricRegistry()
        reg.register("latency")
        reg.record("latency", 100)
        reg.record("latency", 200)
        assert reg.get("latency").count() == 2
        assert "latency" in reg.names()

    @given(values=lists(floats(min_value=0, max_value=1e6,
                               allow_nan=False, allow_infinity=False),
                        min_size=1, max_size=20))
    def test_rate_positive(values):
        r = rate(values, 1.0)
        assert r >= 0

    def test_bucket_histogram():
        values = [1, 5, 10, 15, 20]
        counts = bucket_histogram(values, [5, 10, 15])
        assert sum(counts) == 5
        assert len(counts) == 4

    @given(values=lists(floats(min_value=-100, max_value=100,
                               allow_nan=False, allow_infinity=False),
                        min_size=1, max_size=20))
    def test_ewma_same_length(values):
        result = ewma(values, 0.5)
        assert len(result) == len(values)

    def test_ewma_alpha_one():
        result = ewma([1, 2, 3, 4], 1.0)
        assert result == [1, 2, 3, 4]

    def test_nan_rejected():
        m = Metric()
        try:
            m.add(float('nan'))
            assert False
        except ValueError:
            pass

    def test_inf_rejected():
        m = Metric()
        try:
            m.add(float('inf'))
            assert False
        except ValueError:
            pass
""")


# ── task_scheduler ─────────────────────────────────────────────────────

_register("task_scheduler", """
    from hypothesis import given, settings, assume
    from hypothesis.strategies import integers, text, lists
    from task_scheduler import (
        Status, Task, Scheduler, parse_task_list, execution_order,
    )

    def test_add_and_start():
        s = Scheduler()
        s.add("t1", priority=1)
        t = s.next_task()
        assert t.identifier == "t1"
        s.start("t1")
        assert s.get("t1").status == Status.RUNNING

    def test_complete():
        s = Scheduler()
        s.add("t1")
        s.start("t1")
        s.complete("t1", "done")
        assert s.get("t1").status == Status.DONE
        assert s.get("t1").result == "done"

    def test_fail_and_retry():
        s = Scheduler()
        s.add("t1")
        s.start("t1")
        s.fail("t1", "oops")
        assert s.get("t1").status == Status.FAILED
        s.retry("t1")
        assert s.get("t1").status == Status.READY

    def test_dependencies():
        s = Scheduler()
        s.add("t1")
        s.add("t2", dependencies=["t1"])
        assert s.get("t2").status == Status.PENDING
        s.start("t1")
        s.complete("t1")
        assert s.get("t2").status == Status.READY

    def test_cancel_cascades():
        s = Scheduler()
        s.add("t1")
        s.add("t2", dependencies=["t1"])
        s.cancel("t1")
        assert s.get("t2").status == Status.CANCELLED

    def test_priority_ordering():
        s = Scheduler()
        s.add("low", priority=10)
        s.add("high", priority=1)
        assert s.next_task().identifier == "high"

    def test_deadline_expiry():
        s = Scheduler()
        s.add("t1", deadline=5)
        expired = s.advance(5)
        assert "t1" in expired
        assert s.get("t1").status == Status.CANCELLED

    def test_by_status():
        s = Scheduler()
        s.add("t1")
        s.add("t2")
        s.start("t1")
        running = s.by_status(Status.RUNNING)
        assert len(running) == 1
        assert running[0].identifier == "t1"

    def test_is_blocked():
        s = Scheduler()
        s.add("t1")
        s.add("t2", dependencies=["t1"])
        assert s.is_blocked("t2")
        s.start("t1")
        s.complete("t1")
        assert not s.is_blocked("t2")

    def test_dependents():
        s = Scheduler()
        s.add("t1")
        s.add("t2", dependencies=["t1"])
        s.add("t3", dependencies=["t1"])
        assert set(s.dependents("t1")) == {"t2", "t3"}

    def test_critical_path():
        s = Scheduler()
        s.add("a")
        s.add("b", dependencies=["a"])
        s.add("c", dependencies=["b"])
        path = s.critical_path()
        assert path == ["a", "b", "c"]

    def test_utilization():
        s = Scheduler(now=0)
        s.add("t1")
        s.start("t1")
        s.now = 50
        s.complete("t1")
        util = s.utilization(0, 100)
        assert util == 0.5

    def test_parse_task_list():
        result = parse_task_list("task1: 1\\ntask2: 2")
        assert result == [("task1", 1), ("task2", 2)]

    def test_parse_task_list_comments():
        result = parse_task_list("# comment\\ntask1: 1")
        assert result == [("task1", 1)]

    def test_execution_order():
        tasks = [
            Task("b", 1, dependencies=frozenset(["a"]), status=Status.READY),
            Task("a", 0, status=Status.READY),
        ]
        order = execution_order(tasks)
        assert order == ["a", "b"]

    def test_execution_order_circular():
        tasks = [
            Task("a", 0, dependencies=frozenset(["b"]), status=Status.READY),
            Task("b", 0, dependencies=frozenset(["a"]), status=Status.READY),
        ]
        try:
            execution_order(tasks)
            assert False
        except ValueError:
            pass

    def test_duplicate_id_rejected():
        s = Scheduler()
        s.add("t1")
        try:
            s.add("t1")
            assert False
        except ValueError:
            pass

    def test_self_dependency_rejected():
        s = Scheduler()
        try:
            s.add("t1", dependencies=["t1"])
            assert False
        except (ValueError, KeyError):
            pass
""")


# Targeted additions from the seed-11 F1 audit. Existing generators are retained;
# these test public behavior and boundary/error contracts, not mutation syntax.
def _extend(repo_name: str, code: str) -> None:
    REPO_GENERATORS[repo_name] += '\n' + dedent(code).lstrip()


_extend('route_graph', r'''
    import pytest

    @given(cost=integers(0, 100))
    def test_blocked_endpoint_results(cost):
        graph = Graph()
        graph.connect('a', 'b', cost)
        assert graph.reachable('a', blocked=['a']) == set()
        assert graph.shortest('a', 'b', blocked=['b']) is None

    def test_unknown_single_node_path():
        with pytest.raises(KeyError):
            Graph().path_cost(['unknown'])

    @given(prefix=integers(0, 5))
    def test_invalid_graph_row_number(prefix):
        rows = ['node\ta'] * prefix + ['edge\ta\tb\tnot-a-cost']
        with pytest.raises(ValueError, match=f'invalid graph row {prefix + 1}$'):
            Graph.from_rows(rows)
''')

_extend('stock_reservations', r'''
    import pytest
    from stock_reservations import Warehouse, Lot

    @given(quantity=integers(0, 100))
    def test_adjust_returns_updated_lot(quantity):
        warehouse = Warehouse()
        warehouse.receive('lot', 'sku', 10)
        assert warehouse.adjust('lot', quantity) == Lot('lot', 'sku', quantity, None)

    @given(prefix=integers(0, 5))
    def test_invalid_delivery_reports_row(prefix):
        rows = ['sku:2'] * prefix + ['sku:not-a-quantity']
        with pytest.raises(ValueError, match=f'invalid delivery row {prefix + 1}$'):
            parse_delivery('\n'.join(rows))

    @pytest.mark.parametrize('quantity,ttl', [(0, 1), (-1, 1), (1, 0), (1, -1)])
    def test_reservation_requires_positive_quantity_and_ttl(quantity, ttl):
        warehouse = Warehouse()
        warehouse.receive('lot', 'sku', 10)
        with pytest.raises(ValueError, match='positive quantity and TTL'):
            warehouse.reserve('reservation', 'sku', quantity, ttl)
''')

_extend('config_parser', r'''
    @given(value=integers(-100, 100))
    def test_interpolation_chains_to_terminal_value(value):
        result = interpolate({'a': '${b}', 'b': '${c}', 'c': value})
        assert result == {'a': str(value), 'b': str(value), 'c': value}
''')

_extend('metric_aggregator', r'''
    import pytest
    from metric_aggregator import MetricRegistry, Metric, WindowedMetric, AlertRule

    def test_registered_metric_is_returned():
        registry = MetricRegistry()
        metric = registry.register('load')
        assert isinstance(metric, Metric)
        assert metric is registry.get('load')

    @given(threshold=integers(-100, 100), distance=integers(1, 100))
    def test_less_than_alert(threshold, distance):
        assert AlertRule('low', threshold, 'lt').evaluate(threshold - distance) is True
        assert AlertRule('low', threshold, 'lt').evaluate(threshold) is False

    @pytest.mark.parametrize('value', [float('nan'), float('inf'), -float('inf')])
    def test_window_rejects_nonfinite_values(value):
        with pytest.raises(ValueError, match='finite'):
            WindowedMetric(3).record(value)
''')

_extend('record_index', r'''
    import pytest
    from record_index import Document

    @given(title=text(min_size=1, max_size=20))
    def test_add_returns_document(title):
        index = Index()
        document = index.add('id', title, 'body')
        assert isinstance(document, Document)
        assert document == index.get('id')

    def test_empty_phrase_is_empty_list():
        assert Index().phrase('') == []

    @pytest.mark.parametrize('mode', ['', 'neither', 'ALL'])
    def test_unknown_search_mode_is_rejected(mode):
        with pytest.raises(ValueError, match='mode'):
            Index().search('term', mode=mode)
''')

_extend('slot_planner', r'''
    import pytest
    from slot_planner import Booking

    @given(count=integers(1, 5))
    def test_series_has_exact_occurrences(count):
        planner = Planner({'A': 5})
        result = planner.book_series('series', 'A', Slot(0, 2), 10, count)
        assert len(result) == count
        assert [booking.slot for booking in result] == [Slot(10*i, 10*i+2) for i in range(count)]

    def test_move_returns_new_booking():
        planner = Planner({'A': 5, 'B': 5})
        planner.book('id', 'A', Slot(0, 2))
        assert planner.move('id', 'B', Slot(4, 6)) == Booking('id', 'B', Slot(4, 6), 1)

    @pytest.mark.parametrize('clock', ['oops', 'aa:bb', '1:2:3'])
    def test_malformed_clock_message(clock):
        with pytest.raises(ValueError, match='expected HH:MM'):
            parse_clock(clock)

    @pytest.mark.parametrize('attendees', [0, -1, 6])
    def test_booking_capacity_bounds(attendees):
        with pytest.raises(ValueError, match='capacity'):
            Planner({'A': 5}).book('id', 'A', Slot(0, 2), attendees=attendees)
''')

_extend('task_scheduler', r'''
    import pytest

    @given(priority=integers(0, 100))
    def test_add_returns_task(priority):
        scheduler = Scheduler()
        task = scheduler.add('id', priority=priority)
        assert isinstance(task, Task)
        assert task is scheduler.get('id')

    @given(prefix=integers(0, 5))
    def test_invalid_priority_reports_line(prefix):
        lines = ['# padding'] * prefix + ['id:not-a-priority']
        with pytest.raises(ValueError, match=f'line {prefix + 1}: invalid priority$'):
            parse_task_list('\n'.join(lines))

    def test_empty_scheduler_critical_path():
        assert Scheduler().critical_path() == []
''')
