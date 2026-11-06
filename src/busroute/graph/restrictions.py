"""Bước 5 – Áp ràng buộc viết cứng: whitelist đường chính + đoạn tiếp cận + cấm theo giờ."""

from typing import Any

import networkx as nx

from busroute.config.restrictions import TIME_BANNED_AREAS
from busroute.config.roads import ACCESS_SPURS, BLOCKED_WAYS, MAIN_ROADS
from busroute.utils.geo import normalize_road_name, point_in_polygon
from busroute.utils.timeslots import TIME_SLOTS, in_windows, to_minutes

MAIN_ROAD_NAMES = {normalize_road_name(n) for n in MAIN_ROADS}
SPUR_WAY_IDS = {s["osm_way_id"] for s in ACCESS_SPURS}
BLOCKED_WAY_IDS = {b["osm_way_id"] for b in BLOCKED_WAYS}


def _as_list(x):
    return x if isinstance(x, list) else [x]


def is_statically_allowed(data: dict[str, Any]) -> bool:
    """Allow whitelisted roads or access spurs unless their way ID is blocked."""
    way_ids = set(_as_list(data.get("osmid")))
    if way_ids & BLOCKED_WAY_IDS:
        return False
    on_main = any(
        normalize_road_name(n) in MAIN_ROAD_NAMES for n in _as_list(data.get("name"))
    )
    on_spur = bool(way_ids & SPUR_WAY_IDS)
    return on_main or on_spur


def is_time_banned(mid_lat: float, mid_lon: float, minute_of_day: int) -> bool:
    """Check whether a midpoint lies in a currently active restricted area."""
    for area in TIME_BANNED_AREAS:
        if (
            area["polygon"]
            and point_in_polygon(mid_lat, mid_lon, area["polygon"])
            and in_windows(minute_of_day, area["windows"])
        ):
            return True
    return False


def is_edge_allowed(
    data: dict[str, Any], mid_lat: float, mid_lon: float, minute_of_day: int
) -> bool:
    """Apply static and time-dependent restrictions to an edge midpoint."""
    return is_statically_allowed(data) and not is_time_banned(
        mid_lat, mid_lon, minute_of_day
    )


def apply_restrictions(G: nx.MultiDiGraph) -> tuple[nx.MultiDiGraph, int]:
    """Mutate G by removing disallowed edges and setting slot restriction flags.

    Flags use the edge midpoint and the start of each slot. Return G and the
    number of edges removed.

    Args:
        G: Directed multigraph with node coordinates and OSM edge attributes.

    Returns:
        A tuple containing the modified graph and number of removed edges.
    """
    drop = [
        (u, v, k)
        for u, v, k, d in G.edges(keys=True, data=True)
        if not is_statically_allowed(d)
    ]
    G.remove_edges_from(drop)
    for u, v, k, d in G.edges(keys=True, data=True):
        mid_lat = (G.nodes[u]["y"] + G.nodes[v]["y"]) / 2
        mid_lon = (G.nodes[u]["x"] + G.nodes[v]["x"]) / 2
        d["mid_lat"], d["mid_lon"] = mid_lat, mid_lon
        for slot, (start, _) in TIME_SLOTS.items():
            d[f"blocked_{slot}"] = is_time_banned(mid_lat, mid_lon, to_minutes(start))
    return G, len(drop)
