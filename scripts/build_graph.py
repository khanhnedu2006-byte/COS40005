"""Chạy Bước 2–6: OSM → POI → thời gian → ràng buộc → kiểm tra liên thông → xuất dataset."""
import argparse, glob, os

import pandas as pd
import yaml

from busroute.graph.osm_fetch import fetch_road_graph
from busroute.graph.poi_snap import snap_pois
from busroute.graph.travel_time import add_travel_times
from busroute.graph.restrictions import apply_restrictions
from busroute.graph.connectivity import check_connectivity
from busroute.graph.export import export_graph

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/settings.yaml")
    ap.add_argument("--gps", default="data/interim/gps")
    ap.add_argument("--out", default="data/processed")
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config, encoding="utf-8"))

    G = fetch_road_graph(cfg["osm"].get("place"), cfg["osm"]["network_type"], cfg["osm"]["cache_dir"],
                         bbox=cfg["osm"].get("bbox"))
    print("OSM graph:", G.number_of_nodes(), "nodes,", G.number_of_edges(), "edges")

    snapped, review = snap_pois(G, cfg["poi"]["max_snap_distance_m"])
    for p in review:
        print(f"Kiểm tra tay POI {p['name']}: cách đường {p['snap_distance_m']} m")

    # TODO: map matching GPS → gps_samples (busroute.gps.map_match + speed_calibration)
    G = add_travel_times(G, gps_samples=None)

    G, dropped = apply_restrictions(G)
    print("Đã bỏ", dropped, "cạnh ngoài whitelist")

    missing = check_connectivity(G, [p["node_id"] for p in snapped])
    if missing:
        print("POI bị tách khỏi mạng đường chính – thêm ACCESS_SPURS cho:", missing)

    nodes, edges = export_graph(G, args.out)
    print("Exported", len(nodes), "nodes,", len(edges), "edges →", args.out)
