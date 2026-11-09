"""Coordinate loading and cached road geometry shared by both visualizers."""

from __future__ import annotations

import csv
import math
from itertools import pairwise
from pathlib import Path

import networkx as nx

from Web.types import PointCoordinates, RoadSegments, RouteReference


def load_points(path: str | Path) -> PointCoordinates:
    """Read consistent geographic coordinates from both endpoints of each row.

    Args:
        path: UTF-8 or UTF-8-with-BOM OD matrix CSV path.

    Returns:
        Point IDs mapped to latitude/longitude pairs in degrees.

    Raises:
        OSError: The file cannot be read.
        KeyError: A required coordinate or endpoint column is absent.
        ValueError: Coordinates are invalid, inconsistent, or absent.
    """
    points: PointCoordinates = {}
    with open(path, encoding="utf-8-sig", newline="") as file:
        for row in csv.DictReader(file):
            for prefix in ("from", "to"):
                point = row[f"{prefix}_id"]
                lat, lon = float(row[f"{prefix}_lat"]), float(row[f"{prefix}_lon"])
                if not (
                    math.isfinite(lat)
                    and math.isfinite(lon)
                    and -90 <= lat <= 90
                    and -180 <= lon <= 180
                ):
                    raise ValueError(f"Invalid coordinates for {point}")
                if point in points and points[point] != (lat, lon):
                    raise ValueError(f"Inconsistent coordinates for {point}")
                points[point] = (lat, lon)
    if not points:
        raise ValueError("OD matrix is empty")
    return points


class RoadPaths:
    """Reconstruct and cache directed paths on an unprojected OSM graph.

    Attributes:
        graph: Directed multigraph with WGS84 node and edge geometries.
        nodes: Point IDs snapped within the configured distance threshold.
        cache: Directed point pairs mapped to road segments or missing paths.
    """

    def __init__(
        self,
        graph: nx.MultiDiGraph,
        points: PointCoordinates,
        max_snap_distance: float = 300,
    ) -> None:
        """Snap matrix points to road nodes without inventing long connectors.

        Args:
            graph: Unprojected directed road graph whose distances use metres.
            points: Point coordinates as latitude/longitude pairs.
            max_snap_distance: Maximum allowed snap distance in metres.
        """
        import osmnx as ox

        self.graph: nx.MultiDiGraph = graph
        ids = list(points)
        nodes, distances = ox.distance.nearest_nodes(
            graph,
            X=[points[p][1] for p in ids],
            Y=[points[p][0] for p in ids],
            return_dist=True,
        )
        self.nodes: dict[str, int] = {
            p: int(n)
            for p, n, d in zip(ids, nodes, distances)
            if d <= max_snap_distance
        }
        self.cache: dict[tuple[str, str], RoadSegments | None] = {}

    def leg(self, origin: str, destination: str) -> RoadSegments | None:
        """Return cached road geometry for a directed origin/destination pair.

        Args:
            origin: Matrix point ID at the beginning of the leg.
            destination: Matrix point ID at the end of the leg.

        Returns:
            Longitude/latitude edge segments preserving road curves, an empty
            list for a shared snapped node, or None for an unavailable path.
        """

        key = (origin, destination)
        if key not in self.cache:
            segments: RoadSegments | None = None
            if origin in self.nodes and destination in self.nodes:
                try:
                    _, route = nx.bidirectional_dijkstra(
                        self.graph,
                        self.nodes[origin],
                        self.nodes[destination],
                        weight="length",
                    )
                    segments = []
                    for u, v in pairwise(route):
                        edge = min(
                            self.graph[u][v].values(), key=lambda e: float(e["length"])
                        )
                        a, b = self.graph.nodes[u], self.graph.nodes[v]
                        geometry = edge.get("geometry")
                        coords = (
                            [(float(c[0]), float(c[1])) for c in geometry.coords]
                            if geometry is not None
                            else [
                                (float(a["x"]), float(a["y"])),
                                (float(b["x"]), float(b["y"])),
                            ]
                        )
                        segments.append(coords)
                except nx.NetworkXNoPath:
                    pass
            self.cache[key] = segments
        return self.cache[key]

    def route(self, result: RouteReference) -> tuple[RoadSegments, list[str]]:
        """Combine cached legs while retaining explicit gaps for missing paths.

        Args:
            result: Ordered route IDs from a solver snapshot.

        Returns:
            A pair containing available road segments and missing-leg labels.
        """
        segments: RoadSegments = []
        missing: list[str] = []
        for a, b in pairwise(result["route"]):
            leg = self.leg(a, b)
            if leg is None:
                missing.append(f"{a} → {b}")
            else:
                segments.extend(leg)
        return segments, missing
