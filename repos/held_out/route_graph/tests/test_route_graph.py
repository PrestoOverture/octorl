import pytest
from route_graph import Graph, itinerary


# -- helpers --

def diamond():
    """A -> B (1), A -> C (2), B -> D (3), C -> D (1)."""
    g = Graph()
    g.connect("A", "B", 1)
    g.connect("A", "C", 2)
    g.connect("B", "D", 3)
    g.connect("C", "D", 1)
    return g


def linear():
    """A -> B (5) -> C (3) -> D (2)."""
    g = Graph()
    g.connect("A", "B", 5)
    g.connect("B", "C", 3)
    g.connect("C", "D", 2)
    return g


# -- node / edge manipulation --

def test_add_node_and_list():
    g = Graph()
    g.add_node("X")
    g.add_node("Y")
    assert g.nodes() == ("X", "Y")


def test_add_empty_node_raises():
    g = Graph()
    with pytest.raises(ValueError):
        g.add_node("")


def test_connect_creates_nodes():
    g = Graph()
    g.connect("A", "B", 4)
    assert set(g.nodes()) == {"A", "B"}
    assert g.neighbors("A") == (("B", 4),)


def test_connect_negative_cost_raises():
    g = Graph()
    with pytest.raises(ValueError):
        g.connect("A", "B", -1)


def test_connect_bidirectional():
    g = Graph()
    g.connect("X", "Y", 7, bidirectional=True)
    assert ("X", "Y", 7) in g.edges()
    assert ("Y", "X", 7) in g.edges()


def test_disconnect():
    g = Graph()
    g.connect("A", "B", 5)
    cost = g.disconnect("A", "B")
    assert cost == 5
    assert g.neighbors("A") == ()


def test_remove_node():
    g = Graph()
    g.connect("A", "B", 1)
    g.connect("B", "C", 2)
    g.remove_node("B")
    assert "B" not in g.nodes()
    assert g.neighbors("A") == ()


# -- reachable --

def test_reachable_basic():
    g = diamond()
    assert g.reachable("A") == {"A", "B", "C", "D"}


def test_reachable_with_blocked():
    g = diamond()
    # blocking B forces the C path
    assert g.reachable("A", blocked=["B"]) == {"A", "C", "D"}


def test_reachable_source_blocked():
    g = diamond()
    assert g.reachable("A", blocked=["A"]) == set()


def test_reachable_unknown_source():
    g = Graph()
    with pytest.raises(KeyError):
        g.reachable("Z")


# -- shortest path (Dijkstra) --

def test_shortest_diamond():
    g = diamond()
    result = g.shortest("A", "D")
    assert result is not None
    cost, path = result
    # A->C->D = 3 is cheaper than A->B->D = 4
    assert cost == 3
    assert path == ("A", "C", "D")


def test_shortest_no_path():
    g = Graph()
    g.connect("A", "B", 1)
    g.add_node("C")
    assert g.shortest("A", "C") is None


def test_shortest_source_equals_target():
    g = diamond()
    result = g.shortest("A", "A")
    assert result is not None
    assert result == (0, ("A",))


def test_shortest_with_blocked():
    g = diamond()
    result = g.shortest("A", "D", blocked=["C"])
    assert result is not None
    assert result == (4, ("A", "B", "D"))


def test_shortest_unknown_raises():
    g = Graph()
    g.add_node("A")
    with pytest.raises(KeyError):
        g.shortest("A", "Z")


# -- path_cost --

def test_path_cost_linear():
    g = linear()
    assert g.path_cost(["A", "B", "C"]) == 8


def test_path_cost_single_node():
    g = Graph()
    g.add_node("X")
    assert g.path_cost(["X"]) == 0


def test_path_cost_empty_raises():
    g = Graph()
    with pytest.raises(ValueError):
        g.path_cost([])


# -- reverse --

def test_reverse():
    g = Graph()
    g.connect("A", "B", 3)
    g.connect("B", "C", 5)
    r = g.reverse()
    assert ("B", "A", 3) in r.edges()
    assert ("C", "B", 5) in r.edges()
    # original direction absent
    assert ("A", "B", 3) not in r.edges()


# -- induced subgraph --

def test_induced():
    g = diamond()
    sub = g.induced(["A", "B"])
    assert sub.nodes() == ("A", "B")
    assert sub.edges() == (("A", "B", 1),)


def test_induced_unknown_raises():
    g = Graph()
    g.add_node("A")
    with pytest.raises(KeyError):
        g.induced(["A", "Z"])


# -- topological sort --

def test_topological_dag():
    g = diamond()
    order = g.topological()
    assert order.index("A") < order.index("B")
    assert order.index("A") < order.index("C")
    assert order.index("B") < order.index("D")
    assert order.index("C") < order.index("D")


def test_topological_cycle_raises():
    g = Graph()
    g.connect("A", "B", 1)
    g.connect("B", "A", 1)
    with pytest.raises(ValueError, match="cycle"):
        g.topological()


# -- connected components --

def test_components_single():
    g = diamond()
    comps = g.components()
    assert len(comps) == 1
    assert set(comps[0]) == {"A", "B", "C", "D"}


def test_components_isolated():
    g = Graph()
    g.add_node("X")
    g.add_node("Y")
    g.connect("A", "B", 1)
    comps = g.components()
    assert len(comps) == 3


# -- within budget --

def test_within_budget():
    g = diamond()
    # A->B=1, A->C=2, A->C->D=3, A->B->D=4; budget=3 => A,B,C,D all reachable
    result = g.within("A", 3)
    assert set(result) == {"A", "B", "C", "D"}


def test_within_tight_budget():
    g = diamond()
    # budget=2 => A(0), B(1), C(2)
    result = g.within("A", 2)
    assert set(result) == {"A", "B", "C"}


def test_within_negative_budget_raises():
    g = Graph()
    g.add_node("A")
    with pytest.raises(ValueError):
        g.within("A", -1)


# -- serialization roundtrip --

def test_to_rows_from_rows_roundtrip():
    g = diamond()
    rows = g.to_rows()
    restored = Graph.from_rows(rows)
    assert restored.nodes() == g.nodes()
    assert restored.edges() == g.edges()


def test_from_rows_bad_row():
    with pytest.raises(ValueError, match="invalid graph row 1"):
        Graph.from_rows(["garbage"])


def test_from_rows_line_number():
    rows = ["node\tA", "badline"]
    with pytest.raises(ValueError, match="invalid graph row 2"):
        Graph.from_rows(rows)


# -- itinerary --

def test_itinerary_simple():
    g = linear()
    result = itinerary(g, ["A", "C", "D"])
    assert result is not None
    cost, path = result
    assert cost == 10
    assert path == ("A", "B", "C", "D")


def test_itinerary_single_stop():
    g = linear()
    result = itinerary(g, ["B"])
    assert result == (0, ("B",))


def test_itinerary_unreachable():
    g = Graph()
    g.connect("A", "B", 1)
    g.add_node("C")
    assert itinerary(g, ["A", "C"]) is None


def test_itinerary_empty_raises():
    g = Graph()
    with pytest.raises(ValueError, match="at least one stop"):
        itinerary(g, [])
