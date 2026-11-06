"""Tạo od_matrix GIẢ (cùng cột với bản thật) để người làm routing viết env.py trong lúc chờ TomTom.

Người phụ trách: Nguyen Nam Khanh
Chạy: python scripts/make_mock_matrix.py  → data/final/od_matrix_mock.csv
Thời gian = khoảng cách đường chim bay × 1,35 ÷ tốc độ giả định theo khung giờ.
KHÔNG dùng để báo cáo kết quả – chỉ để chạy thử code.
"""

import csv
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from fetch_routes import (
    load_points,
    TIME_SLOTS,
    TIME_BANNED_AREAS,
    banned_in,
    dist_m,  # noqa: E402
    point_in_polygon,
    dist_to_polygon_edge_m,
    RING_EDGE_BAND_M,
)

OUT = ROOT / "data" / "final" / "od_matrix_mock.csv"
DETOUR = 1.35  # đường thật dài hơn chim bay ~35 %
SPEED_KMH = {
    "morning": 25,
    "ban_noon": 22,
    "afternoon": 24,
    "ban_evening": 20,
    "night": 30,
}

if __name__ == "__main__":
    points = load_points()
    ring = TIME_BANNED_AREAS[0]["polygon"]
    deep = {
        p["id"]
        for p in points
        if point_in_polygon(p["lat"], p["lon"], ring)
        and dist_to_polygon_edge_m((p["lat"], p["lon"]), ring) > RING_EDGE_BAND_M
    }
    rows = []
    for a in points:
        for b in points:
            if a["id"] == b["id"]:
                continue
            d = dist_m((a["lat"], a["lon"]), (b["lat"], b["lon"])) * DETOUR
            for slot in TIME_SLOTS:
                ok = not (banned_in(slot) and (a["id"] in deep or b["id"] in deep))
                rows.append(
                    {
                        "from_id": a["id"],
                        "to_id": b["id"],
                        "from_type": a["type"],
                        "to_type": b["type"],
                        "from_lat": a["lat"],
                        "from_lon": a["lon"],
                        "to_lat": b["lat"],
                        "to_lon": b["lon"],
                        "to_poi_score": b["score"],
                        "slot": slot,
                        "duration_s": round(d / (SPEED_KMH[slot] / 3.6)) if ok else 0,
                        "distance_m": round(d) if ok else 0,
                        "feasible": int(ok),
                        "pois_passed": "",
                        "through_ring": 0,
                    }
                )
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(
        f"Đã ghi {OUT}: {len(rows)} dòng ({len(points)} điểm), {sum(r['feasible'] == 0 for r in rows)} dòng feasible=0"
    )
