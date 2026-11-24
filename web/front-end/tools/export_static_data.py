"""Xuất dữ liệu tĩnh cho front-end từ repo COS40005 (POI, hub, đa giác vành đai, ma trận OD dùng cho chế độ mock).

Cách dùng:  python tools/export_static_data.py <đường dẫn tới thư mục gốc repo COS40005>
Ghi vào:    public/data/pois.json, hubs.json, central_ring.geojson, od_matrix.csv
"""
import json, shutil, sys
from pathlib import Path

repo = Path(sys.argv[1]); out = Path(__file__).resolve().parent.parent / "public" / "data"
out.mkdir(parents=True, exist_ok=True)

ns = {}
exec((repo / "configs" / "pois.py").read_text(encoding="utf-8"), ns)
pois = [{"id": "P%03d" % p["poi_id"], "name": p["name"], "lat": p["coordinates"][0], "lng": p["coordinates"][1],
         "category": p.get("category", ""), "score": p["score"]} for p in ns["POIS"]]
ns = {}
exec((repo / "configs" / "hubs.py").read_text(encoding="utf-8"), ns)
hubs = [{"id": h["hub_id"], "name": h["name"], "lat": h["coordinates"][0], "lng": h["coordinates"][1]} for h in ns["HUBS"]]

(out / "pois.json").write_text(json.dumps(pois, ensure_ascii=False, indent=1), encoding="utf-8")
(out / "hubs.json").write_text(json.dumps(hubs, ensure_ascii=False, indent=1), encoding="utf-8")
shutil.copy(repo / "data" / "references" / "central_ring.geojson", out / "central_ring.geojson")
shutil.copy(repo / "data" / "final" / "od_matrix.csv", out / "od_matrix.csv")
print("pois:", len(pois), "hubs:", len(hubs))
