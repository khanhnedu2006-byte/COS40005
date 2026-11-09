"""Typed JSON contracts shared by the web API and solver-process bridge.

Coordinate tuples become JSON arrays during serialization. Only completed
routes cross the solver-process boundary; intermediate search states stay local.
"""

from __future__ import annotations

from typing import Literal, TypeAlias

from typing_extensions import TypeAliasType, TypedDict

AlgorithmName: TypeAlias = Literal["alns", "vns", "aco", "dijkstra"]
RunStatus: TypeAlias = Literal["running", "result", "error", "stopped"]
GraphStatus: TypeAlias = Literal["loading", "ready", "error"]
Coordinate: TypeAlias = tuple[float, float]
PointCoordinates: TypeAlias = dict[str, Coordinate]
RoadSegments: TypeAlias = list[list[Coordinate]]
JSONValue = TypeAliasType(
    "JSONValue",
    "str | int | float | bool | None | list[JSONValue] | dict[str, JSONValue]",
)
JSONObject: TypeAlias = dict[str, JSONValue]


class RouteReference(TypedDict):
    """Ordered point IDs needed to reconstruct a complete or partial route."""

    route: list[str]


class RouteSnapshot(RouteReference):
    """Serialized route and its matrix-derived objective values.

    Attributes:
        route: Ordered point IDs from start through destination.
        score: Score of unique selected and passed POIs.
        duration_min: Elapsed travel time in minutes.
        collected: Sorted unique POI IDs credited to the route.
    """

    score: float
    duration_min: float
    collected: list[str]


class SearchParameters(TypedDict):
    """Validated parameters passed to an isolated solver process.

    Attributes:
        algorithm: Solver key accepted by the web application.
        start: Starting point ID in the selected matrix.
        end: Distinct destination ID in the selected matrix.
        departure: Local departure time in HH:MM format.
        budget_min: Positive travel-time budget in minutes.
        iterations: Search iterations; ignored by Dijkstra.
        seed: Reproducible random seed; ignored by Dijkstra.
    """

    algorithm: AlgorithmName
    start: str
    end: str
    departure: str
    budget_min: float
    iterations: int
    seed: int


class RunStatusPayload(TypedDict):
    """Current computation state and its final route, without search history.

    Attributes:
        status: Running, successful, failed, or cancelled computation.
        result: Complete matrix-evaluated route after success, otherwise None.
        message: Failure explanation, otherwise None.
    """

    status: RunStatus
    result: RouteSnapshot | None
    message: str | None


class PointMetadata(TypedDict):
    """Location metadata used for markers and endpoint selectors."""

    id: str
    name: str
    lat: float
    lon: float
    kind: str
    score: float


class GraphStatusPayload(TypedDict):
    """Graph readiness and cache diagnostics returned by the status endpoint."""

    graph: GraphStatus
    message: str | None
    graph_loads: int
    graph_seconds: float | None
    cached_legs: int


class BootstrapPayload(TypedDict):
    """Initial location, boundary, default-input, and readiness payload."""

    points: list[PointMetadata]
    matrix: str
    ring: JSONObject
    defaults: SearchParameters
    status: GraphStatusPayload


class MultiLineGeometry(TypedDict):
    """GeoJSON road geometry preserving each directed edge's shape."""

    type: Literal["MultiLineString"]
    coordinates: RoadSegments


# Functional syntax preserves the GeoJSON/API field named ``from``.
GeometryLeg = TypedDict(
    "GeometryLeg",
    {
        "from": str,
        "to": str,
        "geometry": MultiLineGeometry | None,
        "message": str | None,
    },
)


class GeometryPayload(TypedDict):
    """Batch of available road geometries and explicit missing-path results."""

    legs: list[GeometryLeg]


class RunCreatedPayload(TypedDict):
    """Identifier used to check, stop, or download the new run."""

    id: str


class RunStoppedPayload(TypedDict):
    """Terminal or already-completed status after a stop request."""

    status: RunStatus
