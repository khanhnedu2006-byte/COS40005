"""Bước 6 – Xuất dataset đồ thị: nodes.parquet, edges.parquet, edge_index.npy (cho PyTorch Geometric)."""
import os
import numpy as np
import pandas as pd


def export_graph(G, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    node_ids = list(G.nodes)
    index_of = {n: i for i, n in enumerate(node_ids)}
    nodes = pd.DataFrame([{
        "node_idx": index_of[n], "node_id": n, "lat": d["y"], "lon": d["x"],
        "is_poi": d.get("is_poi", False), "poi_id": d.get("poi_id"),
        "poi_name": d.get("poi_name"), "poi_score": d.get("poi_score"),
    } for n, d in G.nodes(data=True)])
    rows = []
    for u, v, k, d in G.edges(keys=True, data=True):
        row = {"from_idx": index_of[u], "to_idx": index_of[v], "key": k,
               "osm_way_id": str(d.get("osmid")), "name": str(d.get("name")),
               "road_type": str(d.get("highway")), "length_m": d["length"]}
        # Giữ mọi thuộc tính theo khung giờ và cờ kiểm chứng
        row.update({c: d[c] for c in d if c.startswith(("travel_time_", "blocked_", "speed_"))})
        row.update(limit_kmh=d.get("limit_kmh"), verified_by_gps=d.get("verified_by_gps", False))
        rows.append(row)
    edges = pd.DataFrame(rows)
    nodes.to_parquet(os.path.join(out_dir, "nodes.parquet"), index=False)
    edges.to_parquet(os.path.join(out_dir, "edges.parquet"), index=False)
    np.save(os.path.join(out_dir, "edge_index.npy"), edges[["from_idx", "to_idx"]].to_numpy().T)
    return nodes, edges
