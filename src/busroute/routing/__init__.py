"""Routing solvers, base contracts, and evaluation utilities.

``base`` defines ``BaseRouteSolver``, ``RoadDatasetSpec``, ``RouteRequest``,
and ``RouteOutput``. ``road_metaheuristics`` / ``ppo_waypoint`` implement
``PoiBaseSolver`` subclasses for POI-level routing, while
``edge_metaheuristics`` / ``ppo_train`` implement ``StreetBaseSolver`` subclasses
for intersection-by-intersection street routing.
"""

from busroute.routing.base import (
    BaseRouteSolver,
    RoadDatasetSpec,
    RouteOutput,
    RouteRequest,
)

__all__ = [
    "BaseRouteSolver",
    "RoadDatasetSpec",
    "RouteOutput",
    "RouteRequest",
]
