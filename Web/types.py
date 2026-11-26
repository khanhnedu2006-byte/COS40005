"""Typed JSON contracts shared by the web API and solver-process bridge.

Coordinate tuples become JSON arrays during serialization. Only completed
routes cross the solver-process boundary; intermediate search states stay local.
"""

from __future__ import annotations

from typing import Literal, TypeAlias

from typing_extensions import NotRequired, TypeAliasType, TypedDict

AlgorithmName: TypeAlias = Literal[
    "alns", "vns", "aco", "dijkstra", "ppo"
]
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


class LineGeometry(TypedDict):
    """GeoJSON geometry for the road route."""

    type: Literal["LineString"]
    coordinates: list[Coordinate]


class RoadInstruction(TypedDict):
    """Estimated manoeuvre and distance along a road."""

    action: str
    road: str
    distance_m: float
    coordinate: Coordinate
    coordinates: list[Coordinate]


class RouteSnapshot(RouteReference):
    """Serialized road route and its measured values.

    Attributes:
        route: Ordered point IDs from start through destination.
        score: Score of unique selected and passed POIs.
        distance_km: Total unpenalized road distance in kilometres.
        collected: Sorted unique POI IDs credited to the route.
        geometry: Approximate road geometry between graph nodes.
        narrow_m: Length of roads with a known width below 5.5 metres.
        unknown_width_m: Length of roads without an unambiguous width.
    """

    score: float
    distance_km: float
    duration_min: float
    collected: list[str]
    geometry: LineGeometry
    narrow_m: float
    unknown_width_m: float
    instructions: list[RoadInstruction]


class SearchParameters(TypedDict):
    """Validated parameters passed to an isolated solver process.

    Attributes:
        algorithm: Solver key accepted by the web application.
        start: Starting point ID in the selected matrix.
        end: Distinct destination ID in the selected matrix.
        budget_min: Positive estimated travel-time budget in minutes.
        departure: Local departure time in HH:MM format.
        iterations: Search iterations used by ALNS, VNS and ACO.
        seed: Reproducible random seed used by ALNS, VNS and ACO.
        checkpoint: Saved ZIP checkpoint for the selected PPO solver.
    """

    algorithm: AlgorithmName
    start: str
    end: str
    budget_min: float
    departure: str
    iterations: int
    seed: int
    checkpoint: NotRequired[str | None]


class RunStatusPayload(TypedDict):
    """Current computation state and its final route, without search history.

    Attributes:
        status: Running, successful, failed, or cancelled computation.
        result: Completed road route after success, otherwise None.
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
    routing_available: NotRequired[bool]
    snap_distance_m: NotRequired[float]


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
