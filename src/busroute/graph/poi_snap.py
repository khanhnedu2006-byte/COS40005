"""Bước 3 – Gắn POI viết cứng vào node đường gần nhất."""
import osmnx as ox

from busroute.config.pois import POIS
from busroute.utils.geo import haversine_m


def snap_pois(G, max_distance_m=300):
    # Trả về (danh sách POI đã gắn, danh sách POI cần kiểm tra tay)
    snapped, needs_review = [], []
    for poi in POIS:
        node = ox.distance.nearest_nodes(G, poi["lon"], poi["lat"])
        dist = haversine_m(poi["lat"], poi["lon"], G.nodes[node]["y"], G.nodes[node]["x"])
        item = {**poi, "node_id": node, "snap_distance_m": round(float(dist), 1)}
        G.nodes[node].update(is_poi=True, poi_id=poi["poi_id"], poi_name=poi["name"], poi_score=poi["score"])
        (needs_review if dist > max_distance_m else snapped).append(item)
    return snapped, needs_review
