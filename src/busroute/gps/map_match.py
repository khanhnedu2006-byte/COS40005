"""Gắn điểm GPS vào đoạn đường gần nhất trên đồ thị OSM."""

import osmnx as ox

MAX_MATCH_DISTANCE_M = 30  # điểm cách đường quá xa thì bỏ (nhiễu hoặc trong bãi đỗ)


def match_points(G, gps_df):
    """Match moving GPS points to nearby graph edges using nearest-edge search.

    Args:
        G: Graph accepted by OSMnx; its distance units must be metres for the
            configured match threshold to apply as intended.
        gps_df: Points with longitude, latitude, and `gps_speed_kmh` columns.

    Returns:
        Moving points matched within the distance threshold, with edge IDs and
        `match_dist_m` added. Points at or below 5 km/h are excluded.
    """
    # Bản đơn giản: nearest edge cho từng điểm.
    # TODO: nâng cấp lên HMM map matching (ví dụ thư viện leuvenmapmatching) khi có nhiều dữ liệu hơn
    moving = gps_df[gps_df.gps_speed_kmh > 5].copy()
    edges, dists = ox.distance.nearest_edges(
        G, moving.lon.values, moving.lat.values, return_dist=True
    )
    moving["u"], moving["v"], moving["key"] = zip(*edges)
    moving["match_dist_m"] = (
        dists  # lưu ý: đơn vị phụ thuộc hệ toạ độ của G – chiếu đồ thị trước khi dùng
    )
    return moving[moving.match_dist_m <= MAX_MATCH_DISTANCE_M]
