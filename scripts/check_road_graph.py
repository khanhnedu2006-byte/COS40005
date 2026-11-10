"""Kiểm tra dataset đồ thị đường (road_nodes / road_edges / pois) → reports/road_graph_validation.csv

Người phụ trách: Nguyen Nam Khanh
Chạy: python scripts/check_road_graph.py
- Đường nhanh nhất giữa mọi cặp điểm trên đồ thị (Dijkstra, giờ cấm bỏ đoạn ring_deep) so với TomTom
  (od_matrix.csv + test_legs.csv): sai số, khớp "đi được / không".
- Khoảng cách gắn POI vào nút; liên thông trong giờ cấm.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra

ROOT = Path(__file__).resolve().parents[1]
F = ROOT / "data" / "final"
SLOTS = ["morning", "ban_noon", "afternoon", "ban_evening", "night"]
BAN = {"ban_noon", "ban_evening"}

if __name__ == "__main__":
    nodes = pd.read_csv(F / "road_nodes.csv", encoding="utf-8-sig", keep_default_na=False)
    edges = pd.read_csv(F / "road_edges.csv", encoding="utf-8-sig", keep_default_na=False)
    pts = pd.read_csv(ROOT / "data" / "interim" / "points_snap.csv", encoding="utf-8-sig", keep_default_na=False).set_index("point_id")
    tt = pd.concat([pd.read_csv(F / f, encoding="utf-8-sig", keep_default_na=False) for f in ("od_matrix.csv", "test_legs.csv")])
    n = len(nodes)
    print(f"Đồ thị: {n} nút, {len(edges)} đoạn, {edges.length_m.sum() / 1000:,.0f} km; "
          f"một chiều {edges.oneway.mean() * 100:.0f} %, sâu trong vành đai {edges.ring_deep.sum()} đoạn")
    print("Gắn điểm vào nút (m): trung vị", pts.snap_dist_m.median(), "| > 300 m:",
          pts[pts.snap_dist_m > 300][["name", "snap_dist_m"]].to_dict("index"))
    rows = []
    for s in SLOTS:
        e = edges[edges.ring_deep == 0] if s in BAN else edges
        g = csr_matrix((e[f"time_{s}"].to_numpy() + 1e-3, (e.from_node, e.to_node)), shape=(n, n))
        ids = list(pts.index)
        col = "node_id_ban" if s in BAN else "node_id"           # giờ cấm: nút còn đường ra vào hợp lệ
        src = pts[col].to_numpy()
        dist = dijkstra(g, indices=src)
        sub = tt[tt.slot == s]
        for r in sub.itertuples():
            i, j = ids.index(r.from_id), ids.index(r.to_id)
            d = dist[i, pts[col].iloc[j]]
            banned = s in BAN and (pts.ring_deep.iloc[i] or pts.ring_deep.iloc[j])
            ok = np.isfinite(d) and not banned
            rows.append({"slot": s, "from_id": r.from_id, "to_id": r.to_id, "tomtom_feasible": r.feasible,
                         "graph_feasible": int(ok), "tomtom_s": r.duration_s, "graph_s": round(d) if ok else 0,
                         "group": "point" if "point" in (r.from_type, r.to_type) else ("hub" if "hub" in (r.from_type, r.to_type) else "poi")})
    res = pd.DataFrame(rows)
    both = res[(res.tomtom_feasible == 1) & (res.graph_feasible == 1)].copy()
    both["ape"] = (both.graph_s - both.tomtom_s).abs() / both.tomtom_s
    both["ratio"] = both.graph_s / both.tomtom_s
    summ = both.groupby(["slot", "group"]).agg(n=("ape", "size"), mape=("ape", "mean"),
                                               median_ratio=("ratio", "median")).round(3)
    agree = res.groupby("slot").apply(lambda d: (d.tomtom_feasible == d.graph_feasible).mean(), include_groups=False).round(3)
    summ.to_csv(ROOT / "reports" / "road_graph_validation.csv", encoding="utf-8-sig")
    print(summ.to_string())
    print("Khớp đi được / không với TomTom:", agree.to_dict())
