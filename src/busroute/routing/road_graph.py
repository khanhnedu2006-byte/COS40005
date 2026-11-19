"""Time-budgeted POI connections over the complete directed road graph."""

from __future__ import annotations

import math
import json
import runpy
from pathlib import Path
from dataclasses import dataclass, field
from itertools import pairwise

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra

from busroute.routing.road_dijkstra import road_instructions, road_widths
from busroute.utils.geo import normalize_road_name
from busroute.utils.timeslots import TIME_SLOTS, slot_of, to_minutes
from busroute.graph.ab_estimator import GROUP_OF, GROUPS, SLOTS, PARAMS_FILE


SLOT_STARTS = tuple(
    (to_minutes(start) * 60, slot) for slot, (start, _) in TIME_SLOTS.items()
)


class NoRoadRouteError(ValueError):
    """The supplied road problem has no verified direct route within budget."""


def snap_points(nodes, points):
    """Return graph indices and distances; keep IDs sharing a node separate."""
    scale = np.array(
        [111320 * math.cos(math.radians(float(nodes.lat.mean()))), 111320.0]
    )
    xy = nodes[["lon", "lat"]].to_numpy() * scale
    snapped, distances = {}, {}
    for point in points:
        delta = xy - np.array([point["lon"], point["lat"]]) * scale
        squared = np.einsum("ij,ij->i", delta, delta)
        index = int(np.argmin(squared))
        distances[point["id"]] = float(np.sqrt(squared[index]))
        if distances[point["id"]] <= 300:
            snapped[point["id"]] = index
    return snapped, distances, scale


@dataclass(frozen=True)
class RoadLeg:
    distance_m: float
    weighted_m: float
    collected: int
    edges: tuple[int, ...]
    duration_s: float
    prefixes: dict = field(default_factory=dict, compare=False, repr=False)


@dataclass(frozen=True)
class RoadResult:
    route: tuple[str, ...]
    score: float
    distance_m: float
    weighted_m: float
    collected: int
    duration_s: float = 0.0
    slots: tuple[str, ...] = ()
    connection_slots: tuple[str, ...] = ()

    @property
    def quality(self):
        return (
            self.score,
            -round(self.weighted_m, 6),
            -round(self.duration_s, 6),
            -round(self.distance_m, 6),
        )


class RoadGraph:
    """Load graph attributes once; preserve the selected parallel-edge identity."""

    def __init__(
        self, nodes, edges, widths, points, cancelled=lambda: False, *, time_params=None
    ):
        self.cancelled = cancelled
        self.time_params = (
            time_params
            if time_params is not None
            else json.loads(PARAMS_FILE.read_text(encoding="utf-8"))
        )
        for slot in SLOTS:
            for group in GROUPS:
                coefficient = self.time_params["s_per_m"][slot][group]
                if not math.isfinite(coefficient) or coefficient <= 0:
                    raise ValueError(
                        "Travel-time coefficients must be positive and finite"
                    )
        self.points = {point["id"]: point for point in points}
        self.nodes = nodes.reset_index(drop=True)
        self.snapped, self.snap_distances, self.scale = snap_points(self.nodes, points)
        self.coords = self.nodes[["lon", "lat"]].to_numpy()
        self.pois = sorted(p["id"] for p in points if p["kind"] == "poi")
        self.scores = [float(self.points[p]["score"]) for p in self.pois]
        position = {int(n): i for i, n in enumerate(self.nodes.node_idx)}
        edges = edges.copy()
        edges["u"] = edges.from_idx.map(position)
        edges["v"] = edges.to_idx.map(position)
        if edges[["u", "v"]].isna().any().any():
            raise ValueError("Road edge references an unknown node")
        edges = edges[np.isfinite(edges.length_m) & (edges.length_m > 0)].copy()

        def width_of(name):
            values = {
                widths[n]
                for part in str(name or "").split(" | ")
                if (n := normalize_road_name(part)) in widths
            }
            return next(iter(values)) if len(values) == 1 else None

        edges["width"] = edges.road_name.fillna("").map(width_of)
        edges["narrow"] = edges.width.notna() & (edges.width < 5.5)
        edges["weighted"] = edges.length_m * (1 + edges.narrow.astype(int))
        if "edge_idx" not in edges:
            edges["edge_idx"] = edges.index
        # Keep parallel edges: the fastest one can differ by departure slot.
        self.edges = edges.sort_values("edge_idx", kind="stable").reset_index(drop=True)
        classes = self.edges.get(
            "road_class", pd.Series("residential", index=self.edges.index)
        )
        groups = classes.map(lambda value: GROUP_OF.get(value, "local"))
        self.times = {
            slot: self.edges.length_m.to_numpy()
            * groups.map(self.time_params["s_per_m"][slot]).to_numpy()
            for slot in SLOTS
        }
        self.u = self.edges.u.to_numpy(dtype=np.int32)
        self.v = self.edges.v.to_numpy(dtype=np.int32)
        self.lengths = self.edges.length_m.to_numpy(dtype=float)
        self.weights = self.edges.weighted.to_numpy(dtype=float)
        self.widths = self.edges.width.to_numpy(dtype=float)
        self.labels = []
        for row in self.edges.to_dict("records"):
            label = next(
                (
                    row.get(key)
                    for key in ("road_name_raw", "road_ref", "road_name")
                    if isinstance(row.get(key), str) and row[key].strip()
                ),
                "Unnamed road",
            )
            self.labels.append(label.strip().replace(" | ", " / "))
        self.masks = [0] * len(self.edges)
        a = self.coords[self.u] * self.scale
        delta = (self.coords[self.v] - self.coords[self.u]) * self.scale
        denominator = np.einsum("ij,ij->i", delta, delta)
        for index, poi in enumerate(self.pois):
            self.check_cancelled()
            point = self.points[poi]
            q = np.array([point["lon"], point["lat"]]) * self.scale
            ratio = np.divide(
                np.einsum("ij,ij->i", q - a, delta),
                denominator,
                out=np.zeros_like(denominator),
                where=denominator > 0,
            )
            nearest = a + np.clip(ratio, 0, 1)[:, None] * delta
            for edge in np.flatnonzero(np.sum((nearest - q) ** 2, axis=1) <= 100**2):
                self.masks[edge] |= 1 << index

    @classmethod
    def from_files(
        cls, nodes_path, edges_path, widths_path, points, cancelled=lambda: False
    ):
        return cls(
            pd.read_parquet(nodes_path),
            pd.read_parquet(edges_path),
            road_widths(widths_path),
            points,
            cancelled,
        )

    def check_cancelled(self):
        if self.cancelled():
            raise InterruptedError("Search cancelled")

    def node_mask(self, index):
        mask = 0
        for bit, poi in enumerate(self.pois):
            p = self.points[poi]
            delta = (self.coords[index] - [p["lon"], p["lat"]]) * self.scale
            if float(np.dot(delta, delta)) <= 100**2:
                mask |= 1 << bit
        return mask

    def mask_score(self, mask):
        return sum(score for bit, score in enumerate(self.scores) if mask & (1 << bit))

    def advance_leg(self, leg, start_minute, elapsed):
        """Replay edge departure slots, skipping constant-slot spans by prefix sums."""
        index, count = 0, len(leg.edges)
        while index < count:
            clock = (start_minute * 60 + elapsed) % 86400
            slot = slot_of(int(clock // 60))
            if slot not in leg.prefixes:
                leg.prefixes[slot] = np.r_[
                    0.0, np.cumsum(self.times[slot][list(leg.edges)])
                ]
            prefix = leg.prefixes[slot]
            boundary = next(
                t for t in (18000, 37800, 43200, 59400, 68400, 104400) if t > clock
            )
            end = min(
                count,
                max(
                    index + 1,
                    int(
                        np.searchsorted(
                            prefix, prefix[index] + boundary - clock, side="left"
                        )
                    ),
                ),
            )
            elapsed += float(prefix[end] - prefix[index])
            index = end
        return elapsed

    def render_path(self, route, edges, start_minute, budget_s):
        """Evaluate and draw one actual edge sequence with the shared edge clock."""
        source, target = self.snapped[route[0]], self.snapped[route[-1]]
        current, elapsed, mask = source, 0.0, self.node_mask(source)
        for edge in edges:
            self.check_cancelled()
            if int(self.u[edge]) != current or current == target:
                raise ValueError(
                    "Route is disconnected or continues past its destination"
                )
            slot = slot_of((start_minute + int(elapsed // 60)) % 1440)
            elapsed += float(self.times[slot][edge])
            current = int(self.v[edge])
            mask |= self.masks[edge]
        if current != target or elapsed > budget_s + 1e-7:
            raise ValueError(
                "Road route did not reach the destination within the time budget"
            )
        coords = [self.coords[source].tolist()] + [
            self.coords[self.v[e]].tolist() for e in edges
        ]
        if len(coords) == 1:
            coords.append(coords[0][:])
        details = [
            (
                float(self.lengths[e]),
                None if np.isnan(self.widths[e]) else float(self.widths[e]),
                self.labels[e],
            )
            for e in edges
        ]
        return {
            "route": list(route),
            "score": self.mask_score(mask),
            "distance_km": sum(self.lengths[e] for e in edges) / 1000,
            "duration_min": elapsed / 60,
            "collected": [p for bit, p in enumerate(self.pois) if mask & (1 << bit)],
            "geometry": {"type": "LineString", "coordinates": coords},
            "narrow_m": sum(
                length
                for length, width, _ in details
                if width is not None and width < 5.5
            ),
            "unknown_width_m": sum(
                length for length, width, _ in details if width is None
            ),
            "instructions": road_instructions(coords, details, self.scale),
        }

    def snapshot(self, problem, result):
        slots = result.connection_slots or result.slots
        edges = [
            edge
            for (a, b), slot in zip(pairwise(result.route), slots)
            for edge in problem.leg(a, b, slot).edges
        ]
        return self.render_path(
            result.route, edges, problem.start_minute, problem.budget_s
        )


def _parallel_edges(graph, slot):
    # Choose the same directed parallel edge for every solver.
    order = np.lexsort(
        (
            graph.edges.edge_idx.to_numpy(),
            graph.weights,
            graph.times[slot],
            graph.v,
            graph.u,
        )
    )
    origins, targets = graph.u[order], graph.v[order]
    different = np.ones(len(order), dtype=bool)
    different[1:] = (origins[1:] != origins[:-1]) | (targets[1:] != targets[:-1])
    return order[different]


def _road_leg(graph, source, target, distances, predecessors, edge_of, source_mask):
    if not np.isfinite(distances[target]):
        return None
    current, path = target, []
    while current != source:
        previous = int(predecessors[current])
        path.append(edge_of[(previous, current)])
        current = previous
    path.reverse()
    mask = source_mask
    for edge in path:
        mask |= graph.masks[edge]
    return RoadLeg(
        sum(float(graph.lengths[e]) for e in path),
        sum(float(graph.weights[e]) for e in path),
        mask,
        tuple(path),
        float(distances[target]),
    )


class WaypointConnections:
    """Reuse snapped-node paths across episodes for each terminal and slot."""

    def __init__(self, graph):
        self.graph = graph
        self.points = {
            node: graph.node_mask(node) for node in set(graph.snapped.values())
        }
        self.slots = {}
        self.matrices = {}
        self.rows = {}

    def leg(self, terminal, origin, destination, slot):
        graph = self.graph
        graph.check_cancelled()
        key = terminal, slot, origin
        if key not in self.rows:
            if slot not in self.slots:
                selected = _parallel_edges(graph, slot)
                edge_of = {(int(graph.u[e]), int(graph.v[e])): int(e) for e in selected}
                self.slots[slot] = selected, edge_of
            selected, edge_of = self.slots[slot]
            matrix_key = terminal, slot
            if matrix_key not in self.matrices:
                edges = selected[graph.u[selected] != terminal]
                self.matrices[matrix_key] = csr_matrix(
                    (graph.times[slot][edges], (graph.u[edges], graph.v[edges])),
                    shape=(len(graph.nodes), len(graph.nodes)),
                )
            # Keep only paths between snapped nodes after each full-graph search.
            distances, predecessors = dijkstra(
                self.matrices[matrix_key],
                directed=True,
                indices=origin,
                return_predecessors=True,
            )
            self.rows[key] = {
                target: _road_leg(
                    graph,
                    origin,
                    target,
                    distances,
                    predecessors,
                    edge_of,
                    self.points[origin],
                )
                for target in self.points
            }
        return self.rows[key][destination]


class RoadProblem:
    """Request-specific directed POI lookup and unique-score route evaluator."""

    def __init__(
        self,
        graph,
        start,
        end,
        budget_min,
        *,
        start_minute=480,
        all_pois=True,
        waypoint_connections=None,
    ):
        if start == end or start not in graph.points or end not in graph.points:
            raise ValueError("Choose distinct known endpoints")
        for point in (start, end):
            if point not in graph.snapped:
                raise ValueError(f"{point} is more than 300 m from the road graph")
        if not math.isfinite(budget_min) or budget_min <= 0:
            raise ValueError("Time budget must be positive and finite")
        if not isinstance(start_minute, int) or not 0 <= start_minute < 1440:
            raise ValueError("Departure must be a minute within the day")
        self.graph, self.start, self.end = graph, start, end
        self.budget_s = budget_min * 60
        self.start_minute = start_minute
        target = graph.snapped[end]
        self.candidates = sorted(
            p
            for p in graph.pois
            if p in graph.snapped
            and p not in (start, end)
            and graph.snapped[p] != target
        )
        points = [start, end] + (self.candidates if all_pois else [])
        # The destination is a sink: no leg may drive through it and continue.
        self.target = target
        self.sources = sorted({graph.snapped[p] for p in points})
        self.waypoint_connections = waypoint_connections
        self.slot_trees, self.cache, self.score_cache = {}, {}, {}
        self.dijkstra_calls = 0
        direct = self.evaluate((start, end))
        if direct is None:
            raise NoRoadRouteError("No directed route found within the time budget")
        self.direct = direct

    def trees(self, slot):
        if slot not in self.slot_trees:
            graph = self.graph
            chosen = _parallel_edges(graph, slot)
            chosen = chosen[graph.u[chosen] != self.target]
            matrix = csr_matrix(
                (graph.times[slot][chosen], (graph.u[chosen], graph.v[chosen])),
                shape=(len(graph.nodes), len(graph.nodes)),
            )
            edge_of = {(int(graph.u[e]), int(graph.v[e])): int(e) for e in chosen}
            trees = {}
            for source in self.sources:
                graph.check_cancelled()
                trees[source] = dijkstra(
                    matrix, directed=True, indices=source, return_predecessors=True
                )
                self.dijkstra_calls += 1
            self.slot_trees[slot] = trees, edge_of
        return self.slot_trees[slot]

    def leg(self, origin, destination, slot=None):
        slot = slot or slot_of(self.start_minute)
        key = origin, destination, slot
        if key not in self.cache:
            graph = self.graph
            source, target = graph.snapped[origin], graph.snapped[destination]
            if self.waypoint_connections is not None:
                self.cache[key] = self.waypoint_connections.leg(
                    self.target, source, target, slot
                )
            else:
                trees, edge_of = self.trees(slot)
                distances, predecessors = trees[source]
                self.cache[key] = _road_leg(
                    graph,
                    source,
                    target,
                    distances,
                    predecessors,
                    edge_of,
                    graph.node_mask(source),
                )
        return self.cache[key]

    def _best_connection(
        self, origin, destination, elapsed, departure_slot, compare_slots
    ):
        """Replay the departure-slot path and alternatives from reachable slots."""
        leg = self.leg(origin, destination, departure_slot)
        if leg is None:
            return None
        arrival = self.graph.advance_leg(leg, self.start_minute, elapsed)
        if not compare_slots:
            return leg, arrival, departure_slot
        selected_slot = departure_slot
        clock = (self.start_minute * 60 + elapsed) % 86400
        upcoming = sorted(
            ((boundary - clock) % 86400, slot)
            for boundary, slot in SLOT_STARTS
            if slot != departure_slot
        )
        for offset, slot in upcoming:
            # An arrival before the next boundary can already use the exact
            # shortest path for its entire journey under the departure rates.
            if elapsed + offset >= min(arrival, self.budget_s):
                break
            alternative = self.leg(origin, destination, slot)
            if alternative is None or alternative.edges == leg.edges:
                continue
            replayed = self.graph.advance_leg(alternative, self.start_minute, elapsed)
            if replayed < arrival - 1e-7:
                leg, arrival, selected_slot = alternative, replayed, slot
        return leg, arrival, selected_slot

    def timed_connection(self, origin, destination, elapsed, *, compare_slots=True):
        """Plan a leg from the supplied point at the trip's running clock."""
        slot = slot_of((self.start_minute + int(elapsed // 60)) % 1440)
        return self._best_connection(origin, destination, elapsed, slot, compare_slots)

    def evaluate(self, route):
        self.graph.check_cancelled()
        if (
            len(route) < 2
            or route[0] != self.start
            or route[-1] != self.end
            or len(set(route)) != len(route)
            or any(p not in self.candidates for p in route[1:-1])
        ):
            return None
        result, changed = self._evaluate_connections(route, compare_slots=True)
        if changed and len(route) > 2:
            # Arriving earlier at a waypoint can increase the next leg's time
            # when the next slot is faster. Keep the previous complete option.
            original, _ = self._evaluate_connections(route, compare_slots=False)
            if original is not None and (
                result is None or original.quality > result.quality
            ):
                return original
        return result

    def _evaluate_connections(self, route, *, compare_slots):
        mask, distance, weighted, elapsed, slots = 0, 0.0, 0.0, 0.0, []
        connection_slots = []
        changed = False
        for a, b in pairwise(route):
            slot = slot_of((self.start_minute + int(elapsed // 60)) % 1440)
            connection = self._best_connection(a, b, elapsed, slot, compare_slots)
            if connection is None:
                return None, changed
            leg, elapsed, selected_slot = connection
            changed |= selected_slot != slot
            if elapsed > self.budget_s + 1e-7:
                return None, changed
            slots.append(slot)
            connection_slots.append(selected_slot)
            distance += leg.distance_m
            weighted += leg.weighted_m
            mask |= leg.collected
        if mask not in self.score_cache:
            self.score_cache[mask] = sum(
                score
                for bit, score in enumerate(self.graph.scores)
                if mask & (1 << bit)
            )
        result = RoadResult(
            tuple(route),
            self.score_cache[mask],
            distance,
            weighted,
            mask,
            elapsed,
            tuple(slots),
            tuple(connection_slots),
        )
        return result, changed


def configured_points(root=None):
    """Load project POIs and hubs without importing the optional web package."""
    root = Path(root) if root else PARAMS_FILE.parents[1]
    points = []
    for filename, variable, key in (
        ("hubs.py", "HUBS", "hub_id"),
        ("pois.py", "POIS", "poi_id"),
    ):
        for p in runpy.run_path(str(root / "configs" / filename))[variable]:
            poi = key == "poi_id"
            points.append(
                {
                    "id": f"P{int(p[key]):03d}" if poi else p[key],
                    "name": p["name"],
                    "lat": p["coordinates"][0],
                    "lon": p["coordinates"][1],
                    "kind": "poi" if poi else "hub",
                    "score": p.get("score", 0),
                }
            )
    return points
