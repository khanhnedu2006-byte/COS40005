"""Ước lượng thời gian chặng cho điểm A/B BẤT KỲ trên đồ thị đường OSM (chặng 1).

Người phụ trách: Nguyen Nam Khanh
Dùng khi A, B không nằm trong 55 điểm của od_matrix.csv (người dùng gõ địa chỉ bất kỳ, hoặc sinh
A/B ngẫu nhiên lúc train). Tốc độ theo nhóm đường × khung giờ được hiệu chỉnh cho khớp thời gian
TomTom (scripts/calibrate_estimator.py → configs/estimator_params.json).
Luật giống od_matrix: giờ cấm bỏ đoạn đường nằm sâu trong vành đai (trừ trục Duy Tân);
điểm đầu/cuối nằm sâu trong vành đai → giờ cấm không đi được.
"""
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra

ROOT = Path(__file__).resolve().parents[3]
PARAMS_FILE = ROOT / "configs" / "estimator_params.json"
SLOTS = ["morning", "ban_noon", "afternoon", "ban_evening", "night"]
BAN_SLOTS = {"ban_noon", "ban_evening"}
# Gom loại đường OSM thành 6 nhóm để hiệu chỉnh tốc độ
GROUP_OF = {
    "motorway": "fast", "motorway_link": "fast", "trunk": "fast", "trunk_link": "fast",
    "primary": "primary", "primary_link": "primary",
    "secondary": "secondary", "secondary_link": "secondary",
    "tertiary": "tertiary", "tertiary_link": "tertiary",
    "unclassified": "minor", "road": "minor",
    "residential": "local", "living_street": "local",
}
GROUPS = ["fast", "primary", "secondary", "tertiary", "minor", "local"]
# Tốc độ khởi tạo (km/h) trước khi hiệu chỉnh
INIT_KMH = {"fast": 45, "primary": 30, "secondary": 25, "tertiary": 22, "minor": 18, "local": 12}
SNAP_GROUPS = {"fast", "primary", "secondary", "tertiary", "minor"}   # điểm đón phải ở đường xe lớn vào được
PASS_RADIUS_M = 100


def _m_per_deg(lat):
    return 111320 * math.cos(math.radians(lat)), 110540


class ABEstimator:
    def __init__(self, nodes, edges, ring_polygon, corridor_segs, pois, params=None,
                 edge_band_m=25, corridor_band_m=30, endpoint_edge_ok_m=100, endpoint_corridor_ok_m=50):
        self.lat = nodes.sort_values("node_idx").lat.to_numpy()
        self.lon = nodes.sort_values("node_idx").lon.to_numpy()
        e = edges[edges.road_class.isin(GROUP_OF)].copy()
        self.u = e.from_idx.to_numpy()
        self.v = e.to_idx.to_numpy()
        self.length = e.length_m.to_numpy()
        self.group = np.array([GROUPS.index(GROUP_OF[c]) for c in e.road_class])
        self.n = len(self.lat)
        self.ring = np.array(ring_polygon)
        self.corr = np.array(corridor_segs) if corridor_segs else np.zeros((0, 2, 2))
        self.edge_band_m, self.corridor_band_m = edge_band_m, corridor_band_m
        self.endpoint_edge_ok_m, self.endpoint_corridor_ok_m = endpoint_edge_ok_m, endpoint_corridor_ok_m
        # Đoạn nằm sâu trong vành đai (theo trung điểm) → bị bỏ trong giờ cấm
        mid = np.column_stack([(self.lat[self.u] + self.lat[self.v]) / 2, (self.lon[self.u] + self.lon[self.v]) / 2])
        self.edge_deep = self.deep_in_ring(mid, edge_band_m, corridor_band_m)
        self.poi_ids = [p["id"] for p in pois]
        self.poi_xy = np.array([[p["lat"], p["lon"]] for p in pois])
        snap_ok = np.zeros(self.n, bool)
        ok_edges = np.isin(self.group, [GROUPS.index(g) for g in SNAP_GROUPS])
        snap_ok[self.u[ok_edges]] = True
        snap_ok[self.v[ok_edges]] = True
        self.snap_nodes = np.where(snap_ok)[0]
        # Nút mà mọi đoạn nối vào đều nằm sâu trong vành đai → giờ cấm không gắn điểm vào nút này
        touch_ok = np.zeros(self.n, bool)
        touch_ok[self.u[~self.edge_deep]] = True
        touch_ok[self.v[~self.edge_deep]] = True
        self.snap_nodes_ban = self.snap_nodes[touch_ok[self.snap_nodes]]
        self.params = params or self.load_params()
        self._graphs = {}

    # ------------------------------------------------------------------ hình học
    def in_ring(self, pts):
        lat, lon = pts[:, 0:1], pts[:, 1:2]
        a, b = self.ring, np.roll(self.ring, -1, axis=0)
        cond = (a[:, 0] > lat) != (b[:, 0] > lat)
        with np.errstate(divide="ignore", invalid="ignore"):
            xc = a[:, 1] + (lat - a[:, 0]) * (b[:, 1] - a[:, 1]) / (b[:, 0] - a[:, 0])
        return ((cond & (lon < xc)).sum(axis=1) % 2) == 1

    @staticmethod
    def dist_to_segs(pts, a, b, chunk=20000):
        if len(a) == 0:
            return np.full(len(pts), np.inf)
        out = np.empty(len(pts))
        for s in range(0, len(pts), chunk):
            p = pts[s:s + chunk]
            kx = (111320 * np.cos(np.radians(p[:, 0])))[:, None]
            ax, ay = (a[:, 1] - p[:, 1:2]) * kx, (a[:, 0] - p[:, 0:1]) * 110540
            bx, by = (b[:, 1] - p[:, 1:2]) * kx, (b[:, 0] - p[:, 0:1]) * 110540
            dx, dy = bx - ax, by - ay
            L = dx * dx + dy * dy
            with np.errstate(divide="ignore", invalid="ignore"):
                t = np.clip(np.where(L > 0, -(ax * dx + ay * dy) / L, 0), 0, 1)
            out[s:s + chunk] = np.hypot(ax + t * dx, ay + t * dy).min(axis=1)
        return out

    def deep_in_ring(self, pts, edge_band, corridor_band):
        pts = np.atleast_2d(pts)
        inside = self.in_ring(pts)
        idx = np.where(inside)[0]
        if len(idx):
            sub = pts[idx]
            ok = (self.dist_to_segs(sub, self.ring, np.roll(self.ring, -1, axis=0)) > edge_band) & \
                 (self.dist_to_segs(sub, self.corr[:, 0], self.corr[:, 1]) > corridor_band)
            inside[idx] = ok
        return inside

    # ------------------------------------------------------------------ tham số + đồ thị theo khung giờ
    @staticmethod
    def load_params():
        if PARAMS_FILE.exists():
            return json.loads(PARAMS_FILE.read_text(encoding="utf-8"))
        return {"s_per_m": {s: {g: 3.6 / INIT_KMH[g] for g in GROUPS} for s in SLOTS},
                "overhead_s": {s: 0.0 for s in SLOTS}}

    def set_params(self, params):
        self.params = params
        self._graphs = {}

    def graph(self, slot):
        if slot not in self._graphs:
            spm = np.array([self.params["s_per_m"][slot][g] for g in GROUPS])
            w = self.length * spm[self.group] + 1e-3
            keep = ~self.edge_deep if slot in BAN_SLOTS else np.ones(len(w), bool)
            u, v, w, eid = self.u[keep], self.v[keep], w[keep], np.where(keep)[0]
            # Đa cạnh giữa cùng (u, v): giữ cạnh nhẹ nhất
            order = np.lexsort((w, v, u))
            u, v, w, eid = u[order], v[order], w[order], eid[order]
            first = np.ones(len(u), bool)
            first[1:] = (u[1:] != u[:-1]) | (v[1:] != v[:-1])
            u, v, w, eid = u[first], v[first], w[first], eid[first]
            g = csr_matrix((w, (u, v)), shape=(self.n, self.n))
            gt = csr_matrix((w, (v, u)), shape=(self.n, self.n))
            keys = u.astype(np.int64) * self.n + v                            # đã sắp theo (u, v)
            self._graphs[slot] = (g, gt, (keys, eid))
        return self._graphs[slot]

    # ------------------------------------------------------------------ truy vấn
    def snap(self, lat, lon, slot=None):
        # Nút gần nhất nằm trên đường xe lớn vào được (đường cấp 3 trở lên hoặc unclassified).
        # Giờ cấm: chỉ gắn vào nút còn đường ra ngoài vành đai (điểm sát đường bao vẫn đi được)
        cand = self.snap_nodes_ban if slot in BAN_SLOTS else self.snap_nodes
        kx, ky = _m_per_deg(lat)
        d = np.hypot((self.lon[cand] - lon) * kx, (self.lat[cand] - lat) * ky)
        i = int(np.argmin(d))
        return int(cand[i]), float(d[i])

    def endpoint_banned(self, lat, lon):
        return bool(self.deep_in_ring(np.array([[lat, lon]]), self.endpoint_edge_ok_m, self.endpoint_corridor_ok_m)[0])

    def _path_edges(self, pred, src, dst, edge_of, reverse=False):
        # Danh sách mã cạnh trên đường đi ngắn nhất (pred từ dijkstra)
        nodes, cur = [dst], dst
        while cur != src:
            cur = pred[cur]
            if cur < 0:
                return None
            nodes.append(cur)
        nodes = np.array(nodes[::-1], dtype=np.int64)        # src … dst theo chiều dijkstra
        a, b = (nodes[1:], nodes[:-1]) if reverse else (nodes[:-1], nodes[1:])
        keys, eid = edge_of
        return eid[np.searchsorted(keys, a * self.n + b)].tolist()

    def legs(self, src_nodes, dst_nodes, slot, reverse=False):
        """Thời gian (s), quãng đường (m), danh sách cạnh cho mọi cặp src→dst.
        reverse=True: chạy Dijkstra ngược từ mỗi dst (nhanh khi ít dst, nhiều src)."""
        g, gt, edge_of = self.graph(slot)
        res = {}
        if not reverse:
            dist, pred = dijkstra(g, indices=src_nodes, return_predecessors=True)
            for i, s in enumerate(src_nodes):
                for t in dst_nodes:
                    if s == t or not np.isfinite(dist[i, t]):
                        continue
                    pe = self._path_edges(pred[i], s, t, edge_of)
                    if pe is not None:
                        res[(s, t)] = (dist[i, t], pe)
        else:
            dist, pred = dijkstra(gt, indices=dst_nodes, return_predecessors=True)
            for j, t in enumerate(dst_nodes):
                for s in src_nodes:
                    if s == t or not np.isfinite(dist[j, s]):
                        continue
                    pe = self._path_edges(pred[j], t, s, edge_of, reverse=True)
                    if pe is not None:
                        res[(s, t)] = (dist[j, s], pe[::-1])
        return res

    def leg_record(self, cost_s, edges, slot):
        # Gói kết quả một chặng: thời gian (có cộng phí cố định), quãng đường, POI chạy ngang
        L = float(self.length[edges].sum()) if edges else 0.0
        dur = cost_s + self.params["overhead_s"][slot]
        pts = np.column_stack([self.lat[self.u[edges]], self.lon[self.u[edges]]]) if edges else np.zeros((0, 2))
        passed = []
        if len(pts):
            kx = 111320 * np.cos(np.radians(pts[:, 0]))
            d = np.hypot((self.poi_xy[:, None, 1] - pts[None, :, 1]) * kx[None, :],
                         (self.poi_xy[:, None, 0] - pts[None, :, 0]) * 110540)
            passed = [pid for pid, hit in zip(self.poi_ids, (d <= PASS_RADIUS_M).any(axis=1)) if hit]
        return {"duration_s": round(dur), "distance_m": round(L), "pois_passed": passed}

    def group_lengths(self, edges):
        # Tổng chiều dài theo nhóm đường của một đường đi (dùng khi hiệu chỉnh)
        return np.bincount(self.group[edges], weights=self.length[edges], minlength=len(GROUPS))


def load_default(pois):
    """Tạo bộ ước lượng từ dữ liệu chặng 1 + vành đai + trục Duy Tân của dự án."""
    import sys
    sys.path.insert(0, str(ROOT / "scripts"))
    from fetch_routes import TIME_BANNED_AREAS, corridor_segments
    nodes = pd.read_parquet(ROOT / "data" / "stage1" / "nodes_base.parquet")
    edges = pd.read_parquet(ROOT / "data" / "stage1" / "edges_base.parquet")
    return ABEstimator(nodes, edges, TIME_BANNED_AREAS[0]["polygon"], corridor_segments(), pois)


def ab_rows(est, a, b, slots=SLOTS, scores=None):
    """Các chặng còn thiếu khi A, B bất kỳ: A→mỗi POI, mỗi POI→B, A→B, theo từng khung giờ.

    a, b: {"id": "A", "lat": …, "lon": …}. Trả về list dict CÙNG CỘT với od_matrix.csv
    (from_type/to_type = "point"), để ghép với phần POI→POI rồi đưa vào môi trường train.
    """
    scores = scores or {}
    poi_xy = dict(zip(est.poi_ids, map(tuple, est.poi_xy)))
    rows = []
    for slot in slots:
        na, _ = est.snap(a["lat"], a["lon"], slot)
        nb, _ = est.snap(b["lat"], b["lon"], slot)
        ban = slot in BAN_SLOTS
        a_bad = ban and est.endpoint_banned(a["lat"], a["lon"])
        b_bad = ban and est.endpoint_banned(b["lat"], b["lon"])
        poi_nodes = {pid: est.snap(*poi_xy[pid], slot)[0] for pid in est.poi_ids
                     if not (ban and est.endpoint_banned(*poi_xy[pid]))}
        out_legs = est.legs([na], list(poi_nodes.values()) + [nb], slot) if not a_bad else {}
        in_legs = est.legs(list(poi_nodes.values()), [nb], slot, reverse=True) if not b_bad else {}

        def row(src, dst, s_type, d_type, leg):
            rec = est.leg_record(leg[0], leg[1], slot) if leg else None
            skip = {src["id"], dst["id"]}
            return {"from_id": src["id"], "to_id": dst["id"], "from_type": s_type, "to_type": d_type,
                    "from_lat": src["lat"], "from_lon": src["lon"], "to_lat": dst["lat"], "to_lon": dst["lon"],
                    "to_poi_score": scores.get(dst["id"], 0.0), "slot": slot,
                    "duration_s": rec["duration_s"] if rec else 0, "distance_m": rec["distance_m"] if rec else 0,
                    "feasible": int(rec is not None),
                    "pois_passed": ";".join(q for q in rec["pois_passed"] if q not in skip) if rec else "",
                    "through_ring": 0}
        for pid in est.poi_ids:                       # POI nằm sâu trong vành đai lúc cấm → dòng feasible = 0
            node = poi_nodes.get(pid)
            poi = {"id": pid, "lat": poi_xy[pid][0], "lon": poi_xy[pid][1]}
            rows.append(row(a, poi, "point", "poi", out_legs.get((na, node)) if node is not None else None))
            rows.append(row(poi, b, "poi", "point", in_legs.get((node, nb)) if node is not None else None))
        rows.append(row(a, b, "point", "point", out_legs.get((na, nb))))
    return rows


def sample_points(est, n, rng, bbox=(15.84, 107.95, 16.16, 108.36)):
    """Lấy n điểm ngẫu nhiên trên đường xe lớn vào được, trong vùng phục vụ (Đà Nẵng – Hội An – Bà Nà).
    Dùng làm A, B ngẫu nhiên khi train."""
    s, w, nlat, e = bbox
    cand = est.snap_nodes[(est.lat[est.snap_nodes] >= s) & (est.lat[est.snap_nodes] <= nlat)
                          & (est.lon[est.snap_nodes] >= w) & (est.lon[est.snap_nodes] <= e)]
    pick = rng.choice(cand, size=n, replace=False)
    return [{"id": f"X{i:04d}", "lat": float(est.lat[k]), "lon": float(est.lon[k])} for i, k in enumerate(pick)]
