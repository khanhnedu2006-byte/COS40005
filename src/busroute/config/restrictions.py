"""Vùng cấm theo giờ. Quy định 2019: cấm xe khách trên 30 chỗ vào vành đai trung tâm Đà Nẵng."""

import json
from pathlib import Path

RING_FILE = Path(__file__).resolve().parents[3] / "data" / "references" / "central_ring.geojson"

def load_ring_polygon(path):
    data =  json.loads(Path(path).read_text(encoding="utf-8"))["features"][0]
    polygon = data["geometry"]["coordinates"][0]
    return [(lat, lon) for lon, lat in polygon], data["properties"]

_ring,_props = load_ring_polygon(RING_FILE)

TIME_BANNED_AREAS = [
    {
        "name": "Central ring",
        # TODO: vẽ đa giác theo các tuyến bao quanh vành đai (lat, lon), theo chiều kim đồng hồ
        "polygon": _ring,
        "windows": [("10:30", "12:00"), ("16:30", "19:00")],
        "reason": "No passenger vehicles over 30 seats in the central ring (Da Nang, 2019)",
        "source": _props.get("source"),
    },
]

# Ranh giới thành phố Đà Nẵng mới (sau sáp nhập 01/07/2025) nếu Nominatim chưa cập nhật
BOUNDARY_POLYGON = []   # TODO: danh sách (lat, lon)
