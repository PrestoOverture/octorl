"""Directed transport network with exact nonnegative integer edge costs."""
from __future__ import annotations

from collections import deque
from collections.abc import Iterable
import heapq


class Graph:
    """Isolated nodes are retained; replacing an edge replaces its cost."""

    def __init__(self) -> None:
        self._edges: dict[str, dict[str, int]] = {}

    def add_node(self, node: str) -> None:
        if not node:
            raise ValueError("nonempty node required")
        self._edges.setdefault(node, {})

    def connect(self, source: str, target: str, cost: int, bidirectional: bool = False) -> None:
        if not source or not target or cost < 0:
            raise ValueError("invalid edge")
        self.add_node(source)
        self.add_node(target)
        self._edges[source][target] = cost
        if bidirectional:
            self._edges[target][source] = cost

    def disconnect(self, source: str, target: str) -> int:
        return self._edges[source].pop(target)

    def remove_node(self, node: str) -> None:
        del self._edges[node]
        for adjacent in self._edges.values():
            adjacent.pop(node, None)

    def nodes(self) -> tuple[str, ...]:
        return tuple(sorted(self._edges))

    def neighbors(self, node: str) -> tuple[tuple[str, int], ...]:
        return tuple(sorted(self._edges[node].items()))

    def edges(self) -> tuple[tuple[str, str, int], ...]:
        return tuple((source, target, cost) for source in self.nodes()
                     for target, cost in self.neighbors(source))

    def reachable(self, source: str, blocked: Iterable[str] = ()) -> set[str]:
        if source not in self._edges:
            raise KeyError(source)
        excluded = set(blocked)
        if source in excluded:
            return set()
        found = {source}
        pending = deque([source])
        while pending:
            node = pending.popleft()
            for neighbor in self._edges[node]:
                if neighbor not in excluded and neighbor not in found:
                    found.add(neighbor)
                    pending.append(neighbor)
        return found

    def shortest(self, source: str, target: str,
                 blocked: Iterable[str] = ()) -> tuple[int, tuple[str, ...]] | None:
        """Dijkstra; equal costs choose the first discovered sorted-edge path."""
        if source not in self._edges or target not in self._edges:
            raise KeyError("unknown endpoint")
        excluded = set(blocked)
        if source in excluded or target in excluded:
            return None
        distances = {source: 0}
        parents: dict[str, str] = {}
        pending = [(0, source)]
        while pending:
            distance, node = heapq.heappop(pending)
            if distance != distances[node]:
                continue
            if node == target:
                path = [node]
                while path[-1] != source:
                    path.append(parents[path[-1]])
                return distance, tuple(reversed(path))
            for neighbor, cost in self.neighbors(node):
                if neighbor in excluded:
                    continue
                candidate = distance + cost
                if neighbor not in distances or candidate < distances[neighbor]:
                    distances[neighbor] = candidate
                    parents[neighbor] = node
                    heapq.heappush(pending, (candidate, neighbor))
        return None

    def path_cost(self, path: Iterable[str]) -> int:
        nodes = list(path)
        if not nodes:
            raise ValueError("empty path")
        for node in nodes:
            if node not in self._edges:
                raise KeyError(node)
        return sum(self._edges[source][target] for source, target in zip(nodes, nodes[1:]))

    def reverse(self) -> Graph:
        graph = Graph()
        for node in self.nodes():
            graph.add_node(node)
        for source, target, cost in self.edges():
            graph.connect(target, source, cost)
        return graph

    def induced(self, nodes: Iterable[str]) -> Graph:
        selected = set(nodes)
        if not selected.issubset(self._edges):
            raise KeyError("unknown node")
        graph = Graph()
        for node in sorted(selected):
            graph.add_node(node)
        for source, target, cost in self.edges():
            if source in selected and target in selected:
                graph.connect(source, target, cost)
        return graph

    def topological(self) -> tuple[str, ...]:
        """Lexicographically smallest available node first; cycles reject."""
        degrees = dict.fromkeys(self._edges, 0)
        for adjacent in self._edges.values():
            for target in adjacent:
                degrees[target] += 1
        ready = [node for node, degree in degrees.items() if degree == 0]
        heapq.heapify(ready)
        result = []
        while ready:
            node = heapq.heappop(ready)
            result.append(node)
            for target in self._edges[node]:
                degrees[target] -= 1
                if degrees[target] == 0:
                    heapq.heappush(ready, target)
        if len(result) != len(self._edges):
            raise ValueError("graph contains a cycle")
        return tuple(result)

    def components(self) -> tuple[tuple[str, ...], ...]:
        """Weakly connected components, including isolated nodes."""
        undirected = self.reverse()
        for source, target, cost in self.edges():
            undirected.connect(source, target, cost)
        remaining = set(self._edges)
        result = []
        while remaining:
            component = undirected.reachable(min(remaining))
            result.append(tuple(sorted(component)))
            remaining.difference_update(component)
        return tuple(result)

    def within(self, source: str, budget: int) -> tuple[str, ...]:
        if source not in self._edges:
            raise KeyError(source)
        if budget < 0:
            raise ValueError("negative budget")
        result = []
        for target in self.nodes():
            route = self.shortest(source, target)
            if route is not None and route[0] <= budget:
                result.append(target)
        return tuple(result)

    def copy(self) -> Graph:
        return self.induced(self.nodes())

    def to_rows(self) -> list[str]:
        """Tab separated rows; every node is explicitly declared."""
        rows = ["node\t" + node for node in self.nodes()]
        rows.extend(f"edge\t{source}\t{target}\t{cost}" for source, target, cost in self.edges())
        return rows

    @classmethod
    def from_rows(cls, rows: Iterable[str]) -> Graph:
        graph = cls()
        for number, row in enumerate(rows, start=1):
            parts = row.split("\t")
            try:
                if len(parts) == 2 and parts[0] == "node":
                    graph.add_node(parts[1])
                elif len(parts) == 4 and parts[0] == "edge":
                    graph.connect(parts[1], parts[2], int(parts[3]))
                else:
                    raise ValueError("unknown row")
            except ValueError as error:
                raise ValueError(f"invalid graph row {number}") from error
        return graph


def itinerary(graph: Graph, stops: Iterable[str]) -> tuple[int, tuple[str, ...]] | None:
    selected = list(stops)
    if not selected:
        raise ValueError("at least one stop required")
    if selected[0] not in graph.nodes():
        raise KeyError(selected[0])
    cost = 0
    path = [selected[0]]
    for source, target in zip(selected, selected[1:]):
        segment = graph.shortest(source, target)
        if segment is None:
            return None
        cost += segment[0]
        path.extend(segment[1][1:])
    return cost, tuple(path)
