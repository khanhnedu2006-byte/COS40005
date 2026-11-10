"""Dựng dataset ĐỒ THỊ ĐƯỜNG cho RL chọn đường từng ngã rẽ → data/final/{pois,road_nodes,road_edges}.csv

Người phụ trách: Nguyen Nam Khanh
Chạy: python scripts/build_road_graph.py      (~1–2 phút)
Nguồn:
  - Nút / đoạn đường: OpenStreetMap (OSMnx, chặng 1: data/cache/road_graph_drive.graphml).
  - Đường to / bé: thẻ OSM `highway` (+ `lanes` ≥ 4 thì nâng một bậc).
  - Thời gian từng đoạn × 5 khung giờ: chiều dài × giây/mét của nhóm đường, hiệu chỉnh cho khớp
    14.850 thời gian TomTom (configs/estimator_params.json, scripts/calibrate_estimator.py).
  - Vành đai: data/references/central_ring.geojson (CV 4883/2019) + trục Duy Tân (CV 1710/2022).
Đường dân sinh / kiệt hẻm (residential, living_street) KHÔNG có trong đồ thị: xe 45 chỗ không vào.
POI, A, B gắn vào nút gần nhất trên đường còn lại (xe chạy ngang trên đường lớn gần đó).
"""
import csv
import json
import math
import sys
import time
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components, dijkstra

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "configs"))
from busroute.graph.ab_estimator import ABEstimator, GROUP_OF, GROUPS, SLOTS  # noqa: E402
from fetch_routes import (TIME_BANNED_AREAS, corridor_segments, load_points,  # noqa: E402
                          RING_EDGE_BAND_M, CORRIDOR_BAND_M, ENDPOINT_EDGE_OK_M, ENDPOINT_CORRIDOR_OK_M)

GRAPHML = ROOT / "data" / "cache" / "road_graph_drive.graphml"
OUT = ROOT / "data" / "final"
NS = "{http://graphml.graphdrawing.org/xmlns}"
PASS_RADIUS_M = 100
# Mức đường cho xe 45 chỗ (residential / living_street bị loại hẳn)
SIZE_OF = {"motorway": "big", "motorway_link": "big", "trunk": "big", "trunk_link": "big",
           "primary": "big", "primary_link": "big",
           "secondary": "medium", "secondary_link": "medium", "tertiary": "medium", "tertiary_link": "medium",
           "unclassified": "small", "road": "small"}
UPGRADE = {"small": "medium", "medium": "big", "big": "big"}
LANES_UPGRADE = 4


def first(v):
    # Thuộc tính OSMnx gộp có dạng "['a', 'b']" → lấy giá trị đầu
    if v is None:
        return None
    s = str(v).strip()
    if s.startswith("["):
        s = s.strip("[]").split(",")[0]
    return s.strip(" '\"") or None


def read_graph():
    keys, nodes, edges = {}, {}, []
    for _, el in ET.iterparse(GRAPHML, events=("end",)):
        tag = el.tag.replace(NS, "")
        if tag == "key":
            keys[el.get("id")] = el.get("attr.name")
        elif tag == "node":
            d = {keys[c.get("key")]: c.text for c in el}
            nodes[el.get("id")] = (float(d["y"]), float(d["x"]))
            el.clear()
        elif tag == "edge":
            d = {keys[c.get("key")]: c.text for c in el}
            hw = first(d.get("highway"))
            if hw in SIZE_OF:
                u, v = el.get("source"), el.get("target")
                if "geometry" in d:
                    g = d["geometry"]
                    coords = [(float(b), float(a)) for a, b in (p.split() for p in g[g.index("(") + 1:-1].split(","))]
                else:
                    coords = [nodes[u], nodes[v]]
                lanes = first(d.get("lanes"))
                try:
                    lanes = int(float(lanes)) if lanes else None
                except ValueError:
                    lanes = None
                maxspeed = first(d.get("maxspeed"))
                edges.append({"u": u, "v": v, "highway": hw, "name": first(d.get("name")) or "",
                              "ref": first(d.get("ref")) or "", "length_m": float(d.get("length") or 0),
                              "lanes": lanes, "maxspeed": maxspeed if maxspeed and maxspeed.replace(".", "").isdigit() else "",
                              "bridge": int(first(d.get("bridge")) not in (None, "False", "no")),
                              "osm_way_id": first(d.get("osmid")) or "", "coords": coords})
            el.clear()
    return nodes, edges


def seg_dist(pts, a, b):
    # Khoảng cách (m) từ mỗi điểm pts (k×2) tới đoạn a–b gần nhất (a, b: m×2), chiếu phẳng
    kx = (111320 * np.cos(np.radians(pts[:, 0])))[:, None]
    ax, ay = (a[:, 1] - pts[:, 1:2]) * kx, (a[:, 0] - pts[:, 0:1]) * 110540
    bx, by = (b[:, 1] - pts[:, 1:2]) * kx, (b[:, 0] - pts[:, 0:1]) * 110540
    dx, dy = bx - ax, by - ay
    L = dx * dx + dy * dy
    with np.errstate(divide="ignore", invalid="ignore"):
        t = np.clip(np.where(L > 0, -(ax * dx + ay * dy) / L, 0), 0, 1)
    return np.hypot(ax + t * dx, ay + t * dy)


if __name__ == "__main__":
    t0 = time.time()
    osm_nodes, edges = read_graph()
    print(f"Đọc OSM: {len(edges)} đoạn đường xe 45 chỗ đi được ({time.time() - t0:.0f}s)")

    # Giữ thành phần liên thông mạnh lớn nhất (không có ngõ cụt một chiều)
    ids = sorted({e["u"] for e in edges} | {e["v"] for e in edges})
    idx = {k: i for i, k in enumerate(ids)}
    u = np.array([idx[e["u"]] for e in edges]); v = np.array([idx[e["v"]] for e in edges])
    _, lab = connected_components(csr_matrix((np.ones(len(u)), (u, v)), shape=(len(ids), len(ids))),
                                  directed=True, connection="strong")
    big = np.bincount(lab).argmax()
    keep_node = lab == big
    edges = [e for e, a, b in zip(edges, u, v) if keep_node[a] and keep_node[b]]
    ids = [k for k, ok in zip(ids, keep_node) if ok]
    idx = {k: i for i, k in enumerate(ids)}
    lat = np.array([osm_nodes[k][0] for k in ids]); lon = np.array([osm_nodes[k][1] for k in ids])
    print(f"Thành phần liên thông lớn nhất: {len(ids)} nút, {len(edges)} đoạn")

    # Vành đai: điểm sâu trong vành đai (ngoài dải 25 m sát mép và ngoài trục Duy Tân)
    ring = TIME_BANNED_AREAS[0]["polygon"]
    corr = corridor_segments()
    pts_all = [{"id": p["id"], "lat": p["lat"], "lon": p["lon"]} for p in load_points() if p["type"] == "poi"]
    helper = ABEstimator.__new__(ABEstimator)          # chỉ dùng các hàm hình học
    helper.ring = np.array(ring); helper.corr = np.array(corr) if corr else np.zeros((0, 2, 2))
    mids = np.array([[(e["coords"][0][0] + e["coords"][-1][0]) / 2, (e["coords"][0][1] + e["coords"][-1][1]) / 2]
                     for e in edges])
    ring_deep = helper.deep_in_ring(mids, RING_EDGE_BAND_M, CORRIDOR_BAND_M)
    corr_d = helper.dist_to_segs(mids, helper.corr[:, 0], helper.corr[:, 1])
    node_deep = helper.deep_in_ring(np.column_stack([lat, lon]), ENDPOINT_EDGE_OK_M, ENDPOINT_CORRIDOR_OK_M)

    # POI nằm sát đoạn (≤ 100 m theo hình dạng thật của đoạn)
    pois = [p for p in load_points() if p["type"] == "poi"]
    poi_xy = np.array([[p["lat"], p["lon"]] for p in pois])
    near = [[] for _ in edges]
    for i, e in enumerate(edges):
        c = np.array(e["coords"])
        box = (np.abs(poi_xy[:, 0] - c[:, 0].mean()) < 0.02) & (np.abs(poi_xy[:, 1] - c[:, 1].mean()) < 0.02)
        if box.any() and len(c) > 1:
            d = seg_dist(poi_xy[box], c[:-1], c[1:]).min(axis=1)
            near[i] = [p["id"] for p, dd in zip(np.array(pois)[box], d) if dd <= PASS_RADIUS_M]

    # Thời gian từng đoạn theo khung giờ (giây) = chiều dài × giây/mét của nhóm đường
    params = json.loads((ROOT / "configs" / "estimator_params.json").read_text(encoding="utf-8"))
    OUT.mkdir(parents=True, exist_ok=True)
    e_fields = ["edge_id", "from_node", "to_node", "length_m", "road_class", "size_level", "lanes", "maxspeed_kmh",
                "name", "ref", "bridge", "oneway"] + [f"time_{s}" for s in SLOTS] + \
               ["ring_deep", "corridor", "pois_passed", "osm_way_id"]
    pair = {(idx[e["u"]], idx[e["v"]]) for e in edges}
    rows = []
    for i, e in enumerate(edges):
        size = SIZE_OF[e["highway"]]
        if e["lanes"] and e["lanes"] >= LANES_UPGRADE:
            size = UPGRADE[size]
        a, b = idx[e["u"]], idx[e["v"]]
        row = {"edge_id": i, "from_node": a, "to_node": b, "length_m": round(e["length_m"], 1),
               "road_class": e["highway"], "size_level": size, "lanes": e["lanes"] or "",
               "maxspeed_kmh": e["maxspeed"], "name": e["name"], "ref": e["ref"], "bridge": e["bridge"],
               "oneway": int((b, a) not in pair),
               "ring_deep": int(ring_deep[i]), "corridor": int(corr_d[i] <= CORRIDOR_BAND_M),
               "pois_passed": ";".join(near[i]), "osm_way_id": e["osm_way_id"]}
        for s in SLOTS:
            row[f"time_{s}"] = round(e["length_m"] * params["s_per_m"][s][GROUP_OF[e["highway"]]], 1)
        rows.append(row)
    with open(OUT / "road_edges.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=e_fields); w.writeheader(); w.writerows(rows)

    # Nút
    near_node = [[] for _ in ids]
    for r in rows:
        for pid in filter(None, r["pois_passed"].split(";")):
            for n in (r["from_node"], r["to_node"]):
                if pid not in near_node[n]:
                    near_node[n].append(pid)
    deg_out = np.bincount([r["from_node"] for r in rows], minlength=len(ids))
    deg_in = np.bincount([r["to_node"] for r in rows], minlength=len(ids))
    with open(OUT / "road_nodes.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["node_id", "osm_node_id", "lat", "lon", "out_degree", "in_degree", "ring_deep", "near_pois"])
        for i, k in enumerate(ids):
            w.writerow([i, k, round(lat[i], 7), round(lon[i], 7), deg_out[i], deg_in[i], int(node_deep[i]),
                        ";".join(near_node[i])])

    # POI + ví dụ A/B (hub, điểm kiểm tra): gắn vào nút gần nhất
    allpts = load_points(include_test=True)
    from pois import POIS
    meta = {f"P{p['poi_id']:03d}": p for p in POIS}
    with open(OUT / "pois.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        # Giờ cấm: chỉ gắn vào nút còn cả đoạn VÀO lẫn đoạn RA không nằm sâu trong vành đai
        ok_out = np.zeros(len(ids), bool); ok_in = np.zeros(len(ids), bool)
        for r in rows:
            if not r["ring_deep"]:
                ok_out[r["from_node"]] = True; ok_in[r["to_node"]] = True
        ban_ok = ok_out & ok_in
        w.writerow(["point_id", "type", "name", "category", "lat", "lon", "score", "node_id", "snap_dist_m",
                    "node_id_ban", "snap_dist_ban_m", "ring_deep"])
        for p in allpts:
            kx = 111320 * math.cos(math.radians(p["lat"]))
            d = np.hypot((lon - p["lon"]) * kx, (lat - p["lat"]) * 110540)
            n = int(np.argmin(d))
            nb = int(np.argmin(np.where(ban_ok, d, np.inf)))
            deep = helper.deep_in_ring(np.array([[p["lat"], p["lon"]]]), ENDPOINT_EDGE_OK_M, ENDPOINT_CORRIDOR_OK_M)[0]
            w.writerow([p["id"], p["type"], p["name"], meta.get(p["id"], {}).get("category", ""),
                        p["lat"], p["lon"], p["score"], n, round(float(d[n]), 1),
                        nb, round(float(d[nb]), 1), int(deep)])
    print(f"Đã ghi road_edges.csv ({len(rows)}), road_nodes.csv ({len(ids)}), pois.csv ({len(allpts)}) "
          f"– {time.time() - t0:.0f}s")
    print("Mức đường (km):", {s: round(sum(r['length_m'] for r in rows if r['size_level'] == s) / 1000)
                             for s in ("big", "medium", "small")})
