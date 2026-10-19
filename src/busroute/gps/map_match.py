"""Gắn điểm GPS vào đoạn đường gần nhất trên đồ thị OSM."""
import osmnx as ox

MAX_MATCH_DISTANCE_M = 30   # điểm cách đường quá xa thì bỏ (nhiễu hoặc trong bãi đỗ)


def match_points(G, gps_df):
    # Bản đơn giản: nearest edge cho từng điểm.
    # TODO: nâng cấp lên HMM map matching (ví dụ thư viện leuvenmapmatching) khi có nhiều dữ liệu hơn
    moving = gps_df[gps_df.gps_speed_kmh > 5].copy()
    edges, dists = ox.distance.nearest_edges(G, moving.lon.values, moving.lat.values, return_dist=True)
    moving["u"], moving["v"], moving["key"] = zip(*edges)
    moving["match_dist_m"] = dists   # lưu ý: đơn vị phụ thuộc hệ toạ độ của G – chiếu đồ thị trước khi dùng
    return moving[moving.match_dist_m <= MAX_MATCH_DISTANCE_M]
