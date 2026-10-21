"""Chặng 1 – Bản đồ nền: tải đồ thị OSM → nodes_base, edges_base, osm_road_names.

Người phụ trách: Nguyen Nam Khanh
Chạy:  python scripts/stage1_build_base.py
Sau đó: python scripts/check_stage1.py
"""
import argparse
import os
import time

import yaml

from busroute.graph.osm_fetch import fetch_road_graph
from busroute.graph.base_tables import build_base_tables, road_name_summary

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/settings.yaml")
    ap.add_argument("--out", default="data/stage1")
    ap.add_argument("--reports", default="reports")
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config, encoding="utf-8"))["osm"]

    t0 = time.time()
    # Lần đầu gọi Overpass (có thể lâu); các lần sau đọc từ data/cache/road_graph_drive.graphml
    G = fetch_road_graph(cfg.get("place"), cfg["network_type"], cfg["cache_dir"], bbox=cfg.get("bbox"))
    print(f"Đồ thị OSM: {G.number_of_nodes():,} node, {G.number_of_edges():,} cạnh ({time.time() - t0:.0f}s)")

    nodes, edges = build_base_tables(G)
    names = road_name_summary(edges)

    os.makedirs(args.out, exist_ok=True)
    os.makedirs(args.reports, exist_ok=True)
    nodes.to_parquet(os.path.join(args.out, "nodes_base.parquet"), index=False)
    edges.to_parquet(os.path.join(args.out, "edges_base.parquet"), index=False)
    names.to_csv(os.path.join(args.reports, "osm_road_names.csv"), index=False, encoding="utf-8-sig")

    print(f"Đã ghi {args.out}/nodes_base.parquet ({len(nodes):,} dòng)")
    print(f"Đã ghi {args.out}/edges_base.parquet ({len(edges):,} dòng)")
    print(f"Đã ghi {args.reports}/osm_road_names.csv ({len(names):,} tên đường)")
    print("Số node theo vùng:", nodes.subgraph_id.value_counts().to_dict())
    print("Cạnh không có tên:", int((edges.road_name == "").sum()))
