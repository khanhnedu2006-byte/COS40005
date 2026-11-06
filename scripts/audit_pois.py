"""Rà toạ độ POI so với mạng đường OSM và vành đai cấm → reports/poi_audit.csv

Người phụ trách: Nguyen Nam Khanh
Chạy: python scripts/audit_pois.py
Mục đích: tìm POI đặt sai chỗ trước khi gọi TomTom (kết quả TomTom gắn với toạ độ,
đổi toạ độ sau khi chạy phải lấy lại toàn bộ cặp điểm của POI đó).
Cờ cảnh báo:
- far_from_road   : cách đường xe ô tô gần nhất > 150 m (TomTom sẽ tự "kéo" điểm về đường gần nhất)
- far_from_major  : cách đường chính (trunk/primary/secondary/tertiary) > 300 m – xe 45 chỗ khó vào
- on_bridge       : đường gần nhất là cầu → dễ bị quay đầu vòng (như Cầu Rồng)
- inside_ring     : nằm sâu trong vành đai (> 25 m từ mép) → giờ cấm không tới được
- near_ring_edge  : cách mép vành đai < 60 m → nên đặt hẳn ra phía đường bao
"""

import csv
import math
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "configs"))
sys.path.insert(0, str(ROOT / "scripts"))

from busroute.config.restrictions import TIME_BANNED_AREAS  # noqa: E402
from busroute.utils.geo import point_in_polygon  # noqa: E402
from fetch_routes import load_points, dist_to_polygon_edge_m, RING_EDGE_BAND_M  # noqa: E402

GRAPHML = ROOT / "data" / "cache" / "road_graph_drive.graphml"
OUT = ROOT / "reports" / "poi_audit.csv"
MAJOR = {
    "motorway",
    "trunk",
    "primary",
    "secondary",
    "tertiary",
    "motorway_link",
    "trunk_link",
    "primary_link",
    "secondary_link",
    "tertiary_link",
}
NS = "{http://graphml.graphdrawing.org/xmlns}"


def load_edges(points, radius_deg=0.01):
    # Đọc graphml theo luồng, chỉ giữ cạnh nằm gần một POI (≈1 km) để tiết kiệm bộ nhớ
    keys, nodes, edges = {}, {}, []
    near = lambda la, lo: any(
        abs(la - p["lat"]) < radius_deg and abs(lo - p["lon"]) < radius_deg
        for p in points
    )
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
            if "geometry" in d:
                # LINESTRING (lon lat, lon lat, ...)
                coords = [
                    (float(b), float(a))
                    for a, b in (
                        xy.split()
                        for xy in d["geometry"][
                            d["geometry"].index("(") + 1 : -1
                        ].split(",")
                    )
                ]
            else:
                coords = [nodes[el.get("source")], nodes[el.get("target")]]
            if any(near(*c) for c in coords[:: max(1, len(coords) // 4)] + coords[-1:]):
                edges.append(
                    {
                        "coords": coords,
                        "name": d.get("name") or "",
                        "highway": d.get("highway") or "",
                        "bridge": d.get("bridge") or "",
                        "oneway": d.get("oneway") or "",
                    }
                )
            el.clear()
    return edges


def point_seg(p, a, b):
    # Khoảng cách điểm–đoạn (m) và điểm chiếu trên đoạn (lat, lon), chiếu phẳng quanh p
    kx, ky = 111320 * math.cos(math.radians(p[0])), 110540
    ax, ay = (a[1] - p[1]) * kx, (a[0] - p[0]) * ky
    bx, by = (b[1] - p[1]) * kx, (b[0] - p[0]) * ky
    dx, dy = bx - ax, by - ay
    L = dx * dx + dy * dy
    t = 0.0 if L == 0 else max(0.0, min(1.0, -(ax * dx + ay * dy) / L))
    proj = (a[0] + t * (b[0] - a[0]), a[1] + t * (b[1] - a[1]))
    return math.hypot(ax + t * dx, ay + t * dy), proj


def nearest(p, edges, only_major=False):
    best = (float("inf"), None, None)
    for e in edges:
        if only_major and not any(
            h in MAJOR for h in e["highway"].strip("[]").replace("'", "").split(", ")
        ):
            continue
        for a, b in zip(e["coords"], e["coords"][1:]):
            d, proj = point_seg(p, a, b)
            if d < best[0]:
                best = (d, e, proj)
    return best


if __name__ == "__main__":
    points = load_points()
    print(f"Đọc đồ thị OSM quanh {len(points)} điểm...")
    edges = load_edges(points)
    ring = TIME_BANNED_AREAS[0]["polygon"]
    rows = []
    for p in points:
        loc = (p["lat"], p["lon"])
        d_any, e_any, _ = nearest(loc, edges)
        d_major, e_major, snap = nearest(loc, edges, only_major=True)
        in_poly = point_in_polygon(*loc, ring)
        d_edge = dist_to_polygon_edge_m(loc, ring)
        flags = []
        if d_any > 150:
            flags.append("far_from_road")
        if d_major > 300:
            flags.append("far_from_major")
        if e_any and e_any["bridge"] not in ("", "False", "no"):
            flags.append("on_bridge")
        if in_poly and d_edge > RING_EDGE_BAND_M:
            flags.append("inside_ring")
        elif d_edge < 60:
            flags.append("near_ring_edge")
        rows.append(
            {
                "id": p["id"],
                "name": p["name"],
                "lat": p["lat"],
                "lon": p["lon"],
                "nearest_road": e_any["name"] if e_any else "",
                "nearest_road_class": e_any["highway"] if e_any else "",
                "dist_road_m": round(d_any),
                "nearest_major": e_major["name"] if e_major else "",
                "dist_major_m": round(d_major),
                "ring": "inside" if in_poly else "outside",
                "dist_ring_edge_m": round(d_edge),
                "flags": ";".join(flags),
                # Toạ độ đề xuất = điểm gần nhất trên đường chính (chỉ gợi ý, nhóm duyệt trước khi thay)
                "suggest_lat": round(snap[0], 5)
                if snap and "far_from_road" in flags
                else "",
                "suggest_lon": round(snap[1], 5)
                if snap and "far_from_road" in flags
                else "",
            }
        )
    OUT.parent.mkdir(exist_ok=True)
    with open(OUT, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    flagged = [r for r in rows if r["flags"]]
    print(f"Đã ghi {OUT}: {len(rows)} điểm, {len(flagged)} điểm có cảnh báo")
    for r in flagged:
        print(
            f"  {r['id']} {r['name'][:32]:32} {r['flags']:40} đường gần nhất: {r['nearest_road'][:25]} "
            f"({r['dist_road_m']} m), đường chính: {r['nearest_major'][:25]} ({r['dist_major_m']} m)"
        )
