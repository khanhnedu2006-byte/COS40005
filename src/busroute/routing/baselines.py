"""Baseline để so sánh với PPO: đường ngắn nhất và tham lam theo điểm POI / thời gian."""

from collections.abc import Hashable, Iterable

import networkx as nx


def shortest_path_route(
    G: nx.Graph, start: Hashable, end: Hashable, weight: str = "travel_time_morning"
) -> list[Hashable]:
    """Find the shortest path using a single edge-weight attribute.

    Args:
        G: NetworkX graph.
        start: Starting node ID.
        end: Destination node ID.
        weight: Edge attribute used as the path cost.

    Returns:
        An ordered list of node IDs including start and end.

    Raises:
        networkx.NetworkXNoPath: If the destination is unreachable.
        networkx.NodeNotFound: If a requested node is absent.
    """
    return nx.shortest_path(G, start, end, weight=weight)


def greedy_poi_route(
    G: nx.Graph, start: Hashable, end: Hashable, poi_nodes: Iterable[Hashable],
    budget_s: float, weight: str = "travel_time_morning",
) -> list[Hashable]:
    """Greedily visit POIs by score per travel time before reaching the end.

    Uses fixed edge weights, not the time-dependent OD matrix. It does not
    account for incidental POIs passed along shortest paths.

    Args:
        G: NetworkX graph whose POI nodes may have a `poi_score` attribute.
        start: Starting node ID.
        end: Destination node ID.
        poi_nodes: Candidate POI node IDs.
        budget_s: Maximum total travel time in seconds for selecting a POI.
        weight: Edge travel-time attribute, in seconds.

    Returns:
        An ordered list of node IDs, including start and end. If no POI is
        selected, the direct shortest path may itself exceed the budget.

    Raises:
        networkx.NetworkXNoPath: If the final destination is unreachable.
        networkx.NodeNotFound: If a requested node is absent.
    """
    route, current, used, remaining = [start], start, 0.0, set(poi_nodes)
    while remaining:
        best, best_ratio, best_path, best_t = None, 0.0, None, 0.0
        for p in remaining:
            try:
                t_go, path = nx.single_source_dijkstra(G, current, p, weight=weight)
                t_back = nx.shortest_path_length(G, p, end, weight=weight)
            except nx.NetworkXNoPath:
                continue
            if used + t_go + t_back <= budget_s:
                ratio = (G.nodes[p].get("poi_score") or 0) / max(t_go, 1)
                if ratio > best_ratio:
                    best, best_ratio, best_path, best_t = p, ratio, path, t_go
        if best is None or best_path is None:
            break
        route += best_path[1:]
        used += best_t
        current = best
        remaining.discard(best)
    route += nx.shortest_path(G, current, end, weight=weight)[1:]
    return route
