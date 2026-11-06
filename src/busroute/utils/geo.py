"""Hàm hình học dùng chung."""

import re
import unicodedata

import numpy as np


def haversine_m(lat1, lon1, lat2, lon2):
    """Calculate great-circle distance using a spherical Earth.

    Args:
        lat1: First latitude in degrees, scalar, NumPy array, or pandas Series.
        lon1: First longitude in degrees, scalar, NumPy array, or pandas Series.
        lat2: Second latitude in degrees, scalar, NumPy array, or pandas Series.
        lon2: Second longitude in degrees, scalar, NumPy array, or pandas Series.

    Returns:
        Distance in metres, with shape determined by NumPy broadcasting.
    """
    p1, p2 = np.radians(lat1), np.radians(lat2)
    a = (
        np.sin((p2 - p1) / 2) ** 2
        + np.cos(p1) * np.cos(p2) * np.sin(np.radians(lon2 - lon1) / 2) ** 2
    )
    return 2 * 6371000 * np.arcsin(np.sqrt(a))


def point_in_polygon(lat: float, lon: float, polygon) -> bool:
    """Test a point against a polygon with the ray-casting algorithm.

    Args:
        lat: Point latitude in degrees.
        lon: Point longitude in degrees.
        polygon: Sequence of (latitude, longitude) vertices.

    Returns:
        Whether the ray-casting test places the point inside. An empty polygon
        returns False; boundary points are not handled consistently.
    """
    inside, n = False, len(polygon)
    for i in range(n):
        y1, x1 = polygon[i]
        y2, x2 = polygon[(i + 1) % n]
        if (y1 > lat) != (y2 > lat):
            x_cross = x1 + (lat - y1) * (x2 - x1) / (y2 - y1)
            if lon < x_cross:
                inside = not inside
    return inside


def normalize_road_name(name) -> str:
    """Normalize a road name for matching.

    Args:
        name: A value to convert to text; false values produce an empty string.

    Returns:
        Lowercase NFC text with collapsed whitespace and a leading "đường" or
        "duong" removed, except for the configured road-type prefixes.
    """
    if not name:
        return ""
    name = unicodedata.normalize("NFC", str(name)).strip().lower()
    # Bỏ tiền tố "đường" trừ khi nó là một phần của tên (đường tỉnh 609, đường cao tốc…)
    name = re.sub(
        r"^(đường|duong)\s+(?!(tỉnh|huyện|xã|cao tốc|sắt|tinh|huyen)\b)", "", name
    )
    return re.sub(r"\s+", " ", name)
