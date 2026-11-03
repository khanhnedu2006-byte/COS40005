"""Lấy thời gian di chuyển giữa mọi cặp điểm (POI + hub) từ Goong Directions API → od_matrix.csv

Người phụ trách: Nguyen Nam Khanh
Bước 1 (gọi API, lưu kết quả thô, chạy tiếp được nếu bị ngắt):
    python scripts/fetch_goong_routes.py fetch --dry-run      # chỉ đếm số request
    python scripts/fetch_goong_routes.py fetch --limit 10     # thử 10 cặp
    python scripts/fetch_goong_routes.py fetch                # chạy hết
Bước 2 (dựng ma trận từ kết quả thô, không gọi API):
    python scripts/fetch_goong_routes.py build

Lưu ý:
- Goong KHÔNG có tham số giờ xuất phát / kẹt xe → duration giống nhau mọi giờ.
  Khác biệt theo khung giờ đến từ (1) vành đai cấm trong giờ cấm, (2) hệ số k(slot) học từ GPS
  (data/stage3/gps_calibration.csv; chưa có thì k = 1.0 và ghi cảnh báo).
- Dùng vehicle=truck (gần xe 45 chỗ nhất trong các lựa chọn car/bike/taxi/truck/hd).
- API key đọc từ .env (GOONG_API_KEY), không bao giờ in ra màn hình.
"""
import argparse
import csv
import json
import math
import os
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "configs"))

from busroute.config.restrictions import TIME_BANNED_AREAS   # noqa: E402
from busroute.utils.geo import point_in_polygon               # noqa: E402
from busroute.utils.timeslots import TIME_SLOTS, to_minutes   # noqa: E402

API_URL = "https://rsapi.goong.io/Direction"
VEHICLE = "truck"
RAW_FILE = ROOT / "data" / "cache" / "goong_routes.jsonl"     # data/cache/ đã nằm trong .gitignore
CALIB_FILE = ROOT / "data" / "stage3" / "gps_calibration.csv"
OUT_FILE = ROOT / "data" / "final" / "od_matrix.csv"
REQUESTS_PER_SEC = 4          # gói free của Goong giới hạn 5 request/giây
PASS_RADIUS_M = 100           # POI cách tuyến ≤ 100 m thì tính là "chạy ngang qua"
DENSIFY_STEP_M = 20           # chèn điểm mỗi 20 m để kiểm tra vành đai không bị lọt


# ------------------------------------------------------------------ điểm
def load_points():
    # 50 POI từ configs/pois.py + hub (nếu đã có configs/hubs.py), mã hoá P001 / H01
    from pois import POIS
    points = [{"id": f"P{int(p['poi_id']):03d}", "type": "poi", "name": p["name"],
               "lat": float(p["coordinates"][0]), "lon": float(p["coordinates"][1]),
               "score": float(p.get("score") or 0)} for p in POIS]
    try:
        from hubs import HUBS
        points += [{"id": h["hub_id"], "type": "hub", "name": h["name"],
                    "lat": float(h["coordinates"][0]), "lon": float(h["coordinates"][1]),
                    "score": 0.0} for h in HUBS]
    except ImportError:
        print("Chưa có configs/hubs.py → chỉ dùng POI")
    return points


def read_api_key():
    # Đọc GOONG_API_KEY từ biến môi trường hoặc .env (không dùng thư viện ngoài)
    key = os.environ.get("GOONG_API_KEY")
    env = ROOT / ".env"
    if not key and env.exists():
        for line in env.read_text(encoding="utf-8").splitlines():
            if line.strip().startswith("GOONG_API_KEY="):
                key = line.split("=", 1)[1].strip().strip('"').strip("'")
    if not key:
        sys.exit("Thiếu GOONG_API_KEY trong .env")
    return key


# ------------------------------------------------------------------ hình học
def decode_polyline(encoded, precision=5):
    # Giải mã polyline dạng Google (Goong dùng cùng định dạng) → [(lat, lon), ...]
    coords, index, lat, lon, factor = [], 0, 0, 0, 10 ** precision
    while index < len(encoded):
        for is_lon in (False, True):
            shift = result = 0
            while True:
                b = ord(encoded[index]) - 63
                index += 1
                result |= (b & 0x1F) << shift
                shift += 5
                if b < 0x20:
                    break
            delta = ~(result >> 1) if result & 1 else result >> 1
            if is_lon:
                lon += delta
            else:
                lat += delta
        coords.append((lat / factor, lon / factor))
    return coords


def dist_m(a, b):
    # Khoảng cách xấp xỉ (equirectangular), đủ chính xác cho vài km
    x = math.radians(b[1] - a[1]) * math.cos(math.radians((a[0] + b[0]) / 2))
    y = math.radians(b[0] - a[0])
    return 6371000 * math.hypot(x, y)


def densify(coords, step=DENSIFY_STEP_M):
    # Chèn thêm điểm trên các đoạn dài để không "nhảy qua" vành đai
    out = coords[:1]
    for a, b in zip(coords, coords[1:]):
        n = max(1, int(dist_m(a, b) // step))
        out += [(a[0] + (b[0] - a[0]) * k / n, a[1] + (b[1] - a[1]) * k / n) for k in range(1, n + 1)]
    return out


def crosses_ring(coords, polygon):
    return any(point_in_polygon(lat, lon, polygon) for lat, lon in coords)


def pois_passed(coords, points, skip_ids):
    # POI nằm gần tuyến (trừ điểm đầu/cuối); thô nhưng đủ cho 50 POI
    hits = []
    for p in points:
        if p["type"] != "poi" or p["id"] in skip_ids:
            continue
        if any(dist_m((p["lat"], p["lon"]), c) <= PASS_RADIUS_M for c in coords[::2]):
            hits.append(p["id"])
    return hits


# ------------------------------------------------------------------ gọi API
def call_direction(key, a, b):
    params = {"origin": f"{a['lat']},{a['lon']}", "destination": f"{b['lat']},{b['lon']}",
              "vehicle": VEHICLE, "alternatives": "true", "api_key": key}
    url = API_URL + "?" + urllib.parse.urlencode(params)
    for attempt in range(3):
        try:
            with urllib.request.urlopen(url, timeout=30) as r:
                return json.loads(r.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            # Không in URL vì có chứa api_key
            print(f"   HTTP {e.code} cho {a['id']}→{b['id']} (lần {attempt + 1})")
            if e.code in (401, 403):
                sys.exit("Key sai hoặc hết hạn mức")
        except (urllib.error.URLError, TimeoutError) as e:
            print(f"   Lỗi mạng {a['id']}→{b['id']}: {type(e).__name__} (lần {attempt + 1})")
        time.sleep(2 * (attempt + 1))
    return None


def fetch(args):
    points = load_points()
    pairs = [(a, b) for a in points for b in points if a["id"] != b["id"]]
    done = set()
    if RAW_FILE.exists():
        with open(RAW_FILE, encoding="utf-8") as f:
            done = {(r["from_id"], r["to_id"]) for r in map(json.loads, f)}
    todo = [(a, b) for a, b in pairs if (a["id"], b["id"]) not in done]
    if args.limit:
        todo = todo[: args.limit]
    print(f"{len(points)} điểm → {len(pairs)} cặp; đã có {len(done)}; cần gọi {len(todo)} request "
          f"(~{len(todo) / REQUESTS_PER_SEC / 60:.1f} phút, ~{len(todo) / 1000 * 2.5:.2f} USD nếu hết free)")
    if args.dry_run or not todo:
        return
    key = read_api_key()
    RAW_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(RAW_FILE, "a", encoding="utf-8") as f:
        for i, (a, b) in enumerate(todo, 1):
            data = call_direction(key, a, b)
            if data is None:
                continue
            routes = [{"duration_s": sum(l["duration"]["value"] for l in r["legs"]),
                       "distance_m": sum(l["distance"]["value"] for l in r["legs"]),
                       "polyline": r["overview_polyline"]["points"]} for r in data.get("routes", [])]
            f.write(json.dumps({"from_id": a["id"], "to_id": b["id"], "routes": routes},
                               ensure_ascii=False) + "\n")
            f.flush()
            if i % 50 == 0:
                print(f"  {i}/{len(todo)}")
            time.sleep(1 / REQUESTS_PER_SEC)
    print(f"Đã lưu {RAW_FILE}")


# ------------------------------------------------------------------ dựng ma trận
def load_calibration():
    if not CALIB_FILE.exists():
        print("CẢNH BÁO: chưa có gps_calibration.csv → k = 1.0 cho mọi khung giờ")
        return {s: 1.0 for s in TIME_SLOTS}
    with open(CALIB_FILE, encoding="utf-8-sig") as f:
        return {r["slot"]: float(r["k"]) for r in csv.DictReader(f)}


def banned_in(slot):
    # Khung giờ có nằm trong giờ cấm của vành đai không
    s, e = (to_minutes(x) for x in TIME_SLOTS[slot])
    for area in TIME_BANNED_AREAS:
        for ws, we in area["windows"]:
            if to_minutes(ws) <= s and e <= to_minutes(we):
                return True
    return False


def build(args):
    points = {p["id"]: p for p in load_points()}
    plist = list(points.values())
    ring = TIME_BANNED_AREAS[0]["polygon"]
    k = load_calibration()
    rows, n_infeasible = [], 0
    with open(RAW_FILE, encoding="utf-8") as f:
        raw = [json.loads(line) for line in f]
    for r in raw:
        a, b = points[r["from_id"]], points[r["to_id"]]
        # Giải mã mọi tuyến thay thế một lần, đánh dấu tuyến nào đi vào vành đai
        routes = []
        for rt in r["routes"]:
            coords = densify(decode_polyline(rt["polyline"]))
            routes.append({**rt, "coords": coords, "in_ring": crosses_ring(coords, ring)})
        routes.sort(key=lambda x: x["duration_s"])
        for slot in TIME_SLOTS:
            ok = [x for x in routes if not (banned_in(slot) and x["in_ring"])]
            best = ok[0] if ok else None
            n_infeasible += best is None
            rows.append({
                "from_id": a["id"], "to_id": b["id"], "from_type": a["type"], "to_type": b["type"],
                "from_lat": a["lat"], "from_lon": a["lon"], "to_lat": b["lat"], "to_lon": b["lon"],
                "to_poi_score": b["score"], "slot": slot,
                "duration_s": round(best["duration_s"] * k.get(slot, 1.0)) if best else 0,
                "distance_m": best["distance_m"] if best else 0,
                "feasible": int(best is not None),
                "pois_passed": ";".join(pois_passed(best["coords"], plist, {a["id"], b["id"]})) if best else "",
                "through_ring": int(best["in_ring"]) if best else 0,
            })
    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_FILE, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"Đã ghi {OUT_FILE}: {len(rows)} dòng, {n_infeasible} dòng không có tuyến hợp lệ")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("fetch")
    f.add_argument("--dry-run", action="store_true")
    f.add_argument("--limit", type=int, default=0)
    sub.add_parser("build")
    args = ap.parse_args()
    fetch(args) if args.cmd == "fetch" else build(args)
