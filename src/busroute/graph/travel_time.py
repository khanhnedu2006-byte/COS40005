"""Bước 4 – Thời gian đi mỗi đoạn theo khung giờ: ưu tiên GPS, không có thì giới hạn luật × hệ số."""

from statistics import median
from typing import Any

import networkx as nx

from busroute.config.speeds import (
    CALIBRATION,
    DEFAULT_CALIBRATION,
    DEFAULT_LIMIT_CLASS,
    LEGAL_LIMIT_KMH,
    MIN_GPS_SAMPLES,
    ROAD_LIMIT_CLASS,
)
from busroute.utils.geo import normalize_road_name
from busroute.utils.timeslots import TIME_SLOTS

_LIMIT_CLASS_BY_NAME = {normalize_road_name(k): v for k, v in ROAD_LIMIT_CLASS.items()}


def edge_names(data: dict[str, Any]) -> list[Any]:
    """Return edge names as a list, including a missing name as None."""
    name = data.get("name")
    return name if isinstance(name, list) else [name]


def limit_class_of(data: dict[str, Any]) -> str:
    """Find the first configured speed-limit class among an edge’s names."""
    for name in edge_names(data):
        cls = _LIMIT_CLASS_BY_NAME.get(normalize_road_name(name))
        if cls:
            return cls
    return DEFAULT_LIMIT_CLASS


def add_travel_times(
    G: nx.MultiDiGraph,
    gps_samples: dict[tuple[Any, Any, Any, str], list[float]] | None = None,
) -> nx.MultiDiGraph:
    """Add speed and travel-time attributes for each slot to G in place.

    Use the median GPS speed when enough samples exist, capped at the legal
    limit; otherwise use the configured calibration. Travel times are seconds.

    Args:
        G: Directed multigraph with edge lengths in metres and road names.
        gps_samples: Optional speed lists in km/h keyed by (u, v, key, slot).

    Returns:
        The same graph, with limit, speed, travel-time, and verification
        attributes added to its edges.
    """
    gps_samples = gps_samples or {}
    for u, v, k, data in G.edges(keys=True, data=True):
        cls = limit_class_of(data)
        limit = LEGAL_LIMIT_KMH[cls]
        data.update(limit_class=cls, limit_kmh=limit, verified_by_gps=False)
        for slot in TIME_SLOTS:
            samples = gps_samples.get((u, v, k, slot), [])
            if len(samples) >= MIN_GPS_SAMPLES:
                speed = min(median(samples), limit)  # ưu tiên 1: GPS thật
                data["verified_by_gps"] = True
            else:
                speed = limit * CALIBRATION.get(
                    (cls, slot), DEFAULT_CALIBRATION
                )  # ưu tiên 2
            data[f"speed_{slot}"] = speed
            data[f"travel_time_{slot}"] = data["length"] / (speed / 3.6)
    return G
