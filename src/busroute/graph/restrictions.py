"""Bước 5 – Áp ràng buộc viết cứng: whitelist đường chính + đoạn tiếp cận + cấm theo giờ."""
from busroute.config.roads import MAIN_ROADS, ACCESS_SPURS, BLOCKED_WAYS
from busroute.config.restrictions import TIME_BANNED_AREAS
from busroute.utils.geo import normalize_road_name, point_in_polygon
from busroute.utils.timeslots import TIME_SLOTS, to_minutes, in_windows

MAIN_ROAD_NAMES = {normalize_road_name(n) for n in MAIN_ROADS}
SPUR_WAY_IDS = {s["osm_way_id"] for s in ACCESS_SPURS}
BLOCKED_WAY_IDS = {b["osm_way_id"] for b in BLOCKED_WAYS}


def _as_list(x):
    return x if isinstance(x, list) else [x]


def is_statically_allowed(data):
    # Không phụ thuộc giờ: thuộc whitelist và không nằm trong danh sách cấm tuyệt đối
    way_ids = set(_as_list(data.get("osmid")))
    if way_ids & BLOCKED_WAY_IDS:
        return False
    on_main = any(normalize_road_name(n) in MAIN_ROAD_NAMES for n in _as_list(data.get("name")))
    on_spur = bool(way_ids & SPUR_WAY_IDS)
    return on_main or on_spur


def is_time_banned(mid_lat, mid_lon, minute_of_day):
    # True nếu trung điểm cạnh nằm trong vùng cấm và đang trong khung giờ cấm
    for area in TIME_BANNED_AREAS:
        if area["polygon"] and point_in_polygon(mid_lat, mid_lon, area["polygon"]) \
                and in_windows(minute_of_day, area["windows"]):
            return True
    return False


def is_edge_allowed(data, mid_lat, mid_lon, minute_of_day):
    # Hàm dùng cho action mask trong môi trường RL
    return is_statically_allowed(data) and not is_time_banned(mid_lat, mid_lon, minute_of_day)


def apply_restrictions(G):
    # Xoá cạnh ngoài whitelist; gắn cờ blocked_<slot> theo giờ cấm (kiểm tra tại đầu khung giờ)
    drop = [(u, v, k) for u, v, k, d in G.edges(keys=True, data=True) if not is_statically_allowed(d)]
    G.remove_edges_from(drop)
    for u, v, k, d in G.edges(keys=True, data=True):
        mid_lat = (G.nodes[u]["y"] + G.nodes[v]["y"]) / 2
        mid_lon = (G.nodes[u]["x"] + G.nodes[v]["x"]) / 2
        d["mid_lat"], d["mid_lon"] = mid_lat, mid_lon
        for slot, (start, _) in TIME_SLOTS.items():
            d[f"blocked_{slot}"] = is_time_banned(mid_lat, mid_lon, to_minutes(start))
    return G, len(drop)
