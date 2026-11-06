"""Chỉ số đánh giá một lộ trình."""

from collections.abc import Hashable, Sequence
from typing import TypedDict

import networkx as nx


class RouteMetrics(TypedDict):
    """Metrics for a route through a weighted NetworkX multigraph."""

    total_min: float
    within_budget: bool
    poi_count: int
    poi_score: float
    unverified_edges: list[tuple[Hashable, Hashable]]


def route_metrics(
    G: nx.MultiDiGraph, route: Sequence[Hashable], budget_s: float,
    weight: str = "travel_time_morning",
) -> RouteMetrics:
    """Summarize travel time, unique POI scores, and unverified route edges.

    This legacy evaluator uses one fixed weight for the entire route; it is
    not equivalent to the slot-aware ODMatrix evaluator.

    Args:
        G: Directed multigraph with weighted edges and optional POI attributes.
        route: Ordered node IDs; consecutive nodes must have connecting edges.
        budget_s: Travel-time budget in seconds.
        weight: Edge attribute containing travel time in seconds.

    Returns:
        A mapping with total minutes, budget status, POI count and score, and
        node pairs whose selected edges lack GPS verification. For parallel
        edges the lowest-weight edge is used. The starting node is not scored.
    """
    total_t, score = 0.0, 0.0
    seen: set[Hashable] = set()
    unverified: list[tuple[Hashable, Hashable]] = []
    for u, v in zip(route[:-1], route[1:]):
        data = min(G[u][v].values(), key=lambda d: d[weight])
        total_t += data[weight]
        if not data.get("verified_by_gps"):
            unverified.append((u, v))  # danh sách cần kiểm tra Street View / tài xế
        if G.nodes[v].get("is_poi") and v not in seen:
            seen.add(v)
            score += G.nodes[v].get("poi_score") or 0
    return {
        "total_min": round(total_t / 60, 1),
        "within_budget": total_t <= budget_s,
        "poi_count": len(seen),
        "poi_score": score,
        "unverified_edges": unverified,
    }
