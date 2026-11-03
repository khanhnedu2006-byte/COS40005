"""Lấy thời gian di chuyển giữa mọi cặp điểm (POI + hub) từ TomTom hoặc Goong → od_matrix.csv

Người phụ trách: Nguyen Nam Khanh
Thử nhanh 1 cặp (in thời gian từng khung giờ, không lưu):
    python scripts/fetch_routes.py test --provider tomtom
Bước 1 (gọi API, lưu kết quả thô, chạy tiếp được nếu bị ngắt hoặc hết hạn mức ngày):
    python scripts/fetch_routes.py fetch --provider tomtom --dry-run   # chỉ đếm số request
    python scripts/fetch_routes.py fetch --provider tomtom --limit 20  # thử 20 request
    python scripts/fetch_routes.py fetch --provider tomtom             # mỗi ngày tối đa 2400 request
Bước 2 (dựng ma trận từ kết quả thô, không gọi API):
    python scripts/fetch_routes.py build --provider tomtom

Lưu ý:
- TomTom: travelMode=bus, có giờ xuất phát (departAt) + kẹt xe lịch sử → mỗi khung giờ 1 request.
  Gói miễn phí ~2.500 request/ngày → 2.450 cặp × 5 khung giờ cần khoảng 5 ngày chạy.
- Goong: không có giờ xuất phát → 1 request/cặp, khác biệt theo giờ chỉ đến từ vành đai + k(slot).
- k(slot) đọc từ data/stage3/gps_calibration.csv; chưa có thì k = 1.0 và in cảnh báo.
- API key đọc từ .env (TOMTOM_API_KEY / GOONG_API_KEY), không bao giờ in ra màn hình.
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

import datetime as dt

RAW_DIR = ROOT / "data" / "cache"                               # data/cache/ đã nằm trong .gitignore
CALIB_FILE = ROOT / "data" / "stage3" / "gps_calibration.csv"
OUT_FILE = ROOT / "data" / "final" / "od_matrix.csv"
PROVIDERS = {
    "tomtom": {"key": "TOMTOM_API_KEY", "per_slot": True, "rps": 4, "daily": 2400},
    "goong": {"key": "GOONG_API_KEY", "per_slot": False, "rps": 4, "daily": 25000},
}
# Giờ đại diện cho mỗi khung giờ (giờ Việt Nam), dùng làm departAt cho TomTom
SLOT_DEPART = {"morning": "08:00", "ban_noon": "11:15", "afternoon": "14:30",
               "ban_evening": "17:45", "night": "20:30"}
# Kích thước xe 45 chỗ (ước lượng) – TomTom dùng để tránh đường quá hẹp / quá thấp.
# KHÔNG khai vehicleWeight: TomTom sẽ áp giờ cấm xe tải nặng ở Đà Nẵng (bắt đi vòng 31 km
# thay vì 12 km lúc 11:15), không phải luật cho xe khách. Luật cấm xe khách = đa giác vành đai.
BUS_DIMENSIONS = {"vehicleLength": 12.0, "vehicleWidth": 2.5, "vehicleHeight": 3.6}
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


def read_api_key(name):
    # Đọc key từ biến môi trường hoặc .env (không dùng thư viện ngoài)
    key = os.environ.get(name)
    env = ROOT / ".env"
    if not key and env.exists():
        for line in env.read_text(encoding="utf-8").splitlines():
            if line.strip().startswith(name + "="):
                key = line.split("=", 1)[1].strip().strip('"').strip("'")
    if not key:
        sys.exit(f"Thiếu {name} trong .env")
    return key


def depart_time(slot):
    # Thứ Tư gần nhất cách hôm nay ≥ 2 ngày (TomTom cần thời điểm trong tương lai)
    today = dt.date.today()
    day = today + dt.timedelta(days=2)
    while day.weekday() != 2:
        day += dt.timedelta(days=1)
    return f"{day.isoformat()}T{SLOT_DEPART[slot]}:00+07:00"


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


RING_END_SKIP_M = 300      # bỏ qua 300 m đầu/cuối: POI nằm sát đường bao thì đoạn tiếp cận có thể lọt vào trong
RING_TOLERANCE_M = 200     # tổng chiều dài nằm trong vành đai > 200 m mới coi là "đi vào vành đai"


def ring_inside_m(coords, polygon, skip=RING_END_SKIP_M):
    # Tổng số mét của tuyến nằm trong đa giác, không tính đoạn đầu/cuối
    cum = [0.0]
    for a, b in zip(coords, coords[1:]):
        cum.append(cum[-1] + dist_m(a, b))
    total, inside = cum[-1], 0.0
    for i in range(1, len(coords)):
        if skip <= cum[i] <= total - skip and point_in_polygon(*coords[i], polygon):
            inside += cum[i] - cum[i - 1]
    return inside


def crosses_ring(coords, polygon):
    return ring_inside_m(coords, polygon) > RING_TOLERANCE_M


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
def http_json(url, key, label):
    # GET có thử lại; mọi thông báo lỗi đều che key
    for attempt in range(3):
        try:
            with urllib.request.urlopen(url, timeout=30) as r:
                return json.loads(r.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            body = e.read()[:200].decode("utf-8", "ignore").replace(key, "***")
            print(f"   HTTP {e.code} cho {label} (lần {attempt + 1}): {body}")
            if e.code in (400, 401, 403):
                return None if e.code == 400 else sys.exit("Key sai, chưa bật dịch vụ hoặc hết hạn mức")
            if e.code == 429:
                time.sleep(10)
        except (urllib.error.URLError, TimeoutError) as e:
            print(f"   Lỗi mạng {label}: {str(e).replace(key, '***')[:120]} (lần {attempt + 1})")
        time.sleep(2 * (attempt + 1))
    return None


def call_tomtom(key, a, b, slot, travel_mode="bus", dims=None):
    dims = BUS_DIMENSIONS if dims is None else dims
    params = {"key": key, "travelMode": travel_mode, "traffic": "true", "routeType": "fastest",
              "departAt": depart_time(slot), "maxAlternatives": "3", **dims}
    url = (f"https://api.tomtom.com/routing/1/calculateRoute/"
           f"{a['lat']},{a['lon']}:{b['lat']},{b['lon']}/json?" + urllib.parse.urlencode(params))
    data = http_json(url, key, f"{a['id']}→{b['id']} {slot}")
    if data is None:
        return None
    return [{"duration_s": r["summary"]["travelTimeInSeconds"],
             "distance_m": r["summary"]["lengthInMeters"],
             "traffic_delay_s": r["summary"].get("trafficDelayInSeconds", 0),
             "points": [[round(p["latitude"], 5), round(p["longitude"], 5)]
                        for leg in r["legs"] for p in leg["points"]]} for r in data.get("routes", [])]


def call_goong(key, a, b, slot=None):
    params = {"origin": f"{a['lat']},{a['lon']}", "destination": f"{b['lat']},{b['lon']}",
              "vehicle": "truck", "alternatives": "true", "api_key": key}
    data = http_json("https://rsapi.goong.io/Direction?" + urllib.parse.urlencode(params), key,
                     f"{a['id']}→{b['id']}")
    if data is None:
        return None
    return [{"duration_s": sum(l["duration"]["value"] for l in r["legs"]),
             "distance_m": sum(l["distance"]["value"] for l in r["legs"]),
             "points": decode_polyline(r["overview_polyline"]["points"])} for r in data.get("routes", [])]


CALLERS = {"tomtom": call_tomtom, "goong": call_goong}


def raw_file(provider):
    return RAW_DIR / f"{provider}_routes.jsonl"


def jobs(points, provider):
    # Danh sách việc cần gọi: (điểm đi, điểm đến, khung giờ hoặc None)
    slots = list(TIME_SLOTS) if PROVIDERS[provider]["per_slot"] else [None]
    return [(a, b, s) for a in points for b in points if a["id"] != b["id"] for s in slots]


def test(args):
    # Gọi thử 1 cặp Cầu Rồng → Ngũ Hành Sơn cho mọi khung giờ, in kết quả, không lưu
    cfg = PROVIDERS[args.provider]
    key = read_api_key(cfg["key"])
    pts = {p["id"]: p for p in load_points()}
    a, b = pts["P001"], pts["P007"]
    ring = TIME_BANNED_AREAS[0]["polygon"]
    print(f"{a['name']} → {b['name']} ({args.provider})")
    saved = []   # lưu toạ độ tuyến để vẽ kiểm tra (data/cache, không đưa lên git)
    for slot in (TIME_SLOTS if cfg["per_slot"] else [None]):
        routes = CALLERS[args.provider](key, a, b, slot)
        saved += [{"slot": slot, "idx": i, **r} for i, r in enumerate(routes or [])]
        if not routes:
            print(f"  {slot}: không có kết quả")
            continue
        for i, r in enumerate(routes):
            inside = ring_inside_m(densify([tuple(p) for p in r["points"]]), ring)
            print(f"  {slot or 'mọi giờ':12} tuyến {i}: {r['duration_s'] / 60:5.1f} phút, "
                  f"{r['distance_m'] / 1000:5.1f} km, trong vành đai {inside / 1000:4.1f} km "
                  f"→ vào vành đai: {inside > RING_TOLERANCE_M}")
        time.sleep(1 / cfg["rps"])
    out = RAW_DIR / f"{args.provider}_test_routes.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(saved, ensure_ascii=False), encoding="utf-8")
    print(f"Đã lưu toạ độ các tuyến vào {out}")


def diag(args):
    # Chẩn đoán key TomTom: gọi 1 tuyến ngắn qua API bản cũ và API Orbis, in mã HTTP (không in key)
    key = read_api_key("TOMTOM_API_KEY")
    print(f"Độ dài key: {len(key)} ký tự")
    loc = "16.06140,108.22763:16.00340,108.26300"
    tests = {
        "Routing API (bản cũ)": f"https://api.tomtom.com/routing/1/calculateRoute/{loc}/json?key={key}",
        "Orbis Routing (apiVersion=2)":
            f"https://api.tomtom.com/maps/orbis/routing/calculateRoute/{loc}/json?apiVersion=2&key={key}",
        "Search API (kiểm tra key nói chung)":
            f"https://api.tomtom.com/search/2/geocode/Da%20Nang.json?limit=1&key={key}",
    }
    for name, url in tests.items():
        try:
            with urllib.request.urlopen(url, timeout=30) as r:
                body = r.read()[:120].decode("utf-8", "ignore")
                print(f"  {name}: HTTP {r.status} OK")
        except urllib.error.HTTPError as e:
            body = e.read()[:160].decode("utf-8", "ignore").replace(key, "***")
            print(f"  {name}: HTTP {e.code} – {body}")
        except Exception as e:
            print(f"  {name}: {type(e).__name__} {str(e).replace(key, '***')[:120]}")


def compare(args):
    # So sánh tuyến nhanh nhất giữa các cấu hình xe để tìm nguyên nhân tuyến đi vòng
    key = read_api_key("TOMTOM_API_KEY")
    pts = {p["id"]: p for p in load_points()}
    a, b = pts[args.src], pts[args.dst]
    ring = TIME_BANNED_AREAS[0]["polygon"]
    variants = {
        "bus + dài/rộng/cao (hiện tại)": ("bus", BUS_DIMENSIONS),
        "bus, không khai kích thước": ("bus", {}),
        "bus + tải trọng 16 tấn": ("bus", {**BUS_DIMENSIONS, "vehicleWeight": 16000}),
        "car": ("car", {}),
    }
    print(f"{a['name']} → {b['name']}, khung giờ {args.slot} ({depart_time(args.slot)})")
    for name, (mode, dims) in variants.items():
        routes = call_tomtom(key, a, b, args.slot, mode, dims)
        if not routes:
            print(f"  {name}: không có kết quả")
            continue
        r = min(routes, key=lambda x: x["duration_s"])
        inside = ring_inside_m(densify([tuple(p) for p in r["points"]]), ring)
        print(f"  {name:45} {r['duration_s'] / 60:5.1f} phút, {r['distance_m'] / 1000:5.1f} km, "
              f"trong vành đai {inside / 1000:4.1f} km")
        time.sleep(0.3)


def fetch(args):
    cfg = PROVIDERS[args.provider]
    points = load_points()
    todo_all = jobs(points, args.provider)
    path = raw_file(args.provider)
    done = set()
    if path.exists():
        with open(path, encoding="utf-8") as f:
            done = {(r["from_id"], r["to_id"], r.get("slot")) for r in map(json.loads, f)}
    todo = [j for j in todo_all if (j[0]["id"], j[1]["id"], j[2]) not in done]
    limit = args.limit or cfg["daily"]
    batch = todo[:limit]
    print(f"{len(points)} điểm → {len(todo_all)} request tổng; đã có {len(done)}; còn {len(todo)}; "
          f"lần này gọi {len(batch)} (~{len(batch) / cfg['rps'] / 60:.1f} phút)")
    if len(todo) > len(batch):
        print(f"→ Cần chạy thêm khoảng {math.ceil((len(todo) - len(batch)) / cfg['daily'])} ngày nữa")
    if args.dry_run or not batch:
        return
    key = read_api_key(cfg["key"])
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        for i, (a, b, slot) in enumerate(batch, 1):
            routes = CALLERS[args.provider](key, a, b, slot)
            if routes is None:
                continue
            f.write(json.dumps({"from_id": a["id"], "to_id": b["id"], "slot": slot, "routes": routes},
                               ensure_ascii=False) + "\n")
            f.flush()
            if i % 100 == 0:
                print(f"  {i}/{len(batch)}")
            time.sleep(1 / cfg["rps"])
    print(f"Đã lưu {path}")


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
    k = load_calibration() if not PROVIDERS[args.provider]["per_slot"] else {}
    rows, n_infeasible = [], 0
    # Gom kết quả thô theo (từ, đến) → {slot hoặc None: danh sách tuyến}
    raw = {}
    with open(raw_file(args.provider), encoding="utf-8") as f:
        for r in map(json.loads, f):
            raw.setdefault((r["from_id"], r["to_id"]), {})[r.get("slot")] = r["routes"]
    # TomTom đã có kẹt xe theo giờ → không nhân thêm k(slot), chỉ dùng k để kiểm tra
    use_k = not PROVIDERS[args.provider]["per_slot"]
    for (fid, tid), by_slot in raw.items():
        a, b = points[fid], points[tid]
        cache = {}
        for slot in TIME_SLOTS:
            key_slot = slot if slot in by_slot else None
            if key_slot not in by_slot:
                continue
            if key_slot not in cache:
                # Giải mã mọi tuyến thay thế một lần, đánh dấu tuyến nào đi vào vành đai
                routes = []
                for rt in by_slot[key_slot]:
                    coords = densify([tuple(p) for p in rt["points"]])
                    routes.append({**rt, "coords": coords, "in_ring": crosses_ring(coords, ring)})
                cache[key_slot] = sorted(routes, key=lambda x: x["duration_s"])
            routes = cache[key_slot]
            ok = [x for x in routes if not (banned_in(slot) and x["in_ring"])]
            best = ok[0] if ok else None
            n_infeasible += best is None
            rows.append({
                "from_id": a["id"], "to_id": b["id"], "from_type": a["type"], "to_type": b["type"],
                "from_lat": a["lat"], "from_lon": a["lon"], "to_lat": b["lat"], "to_lon": b["lon"],
                "to_poi_score": b["score"], "slot": slot,
                "duration_s": round(best["duration_s"] * (k.get(slot, 1.0) if use_k else 1.0)) if best else 0,
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
    for name in ("diag", "test", "compare", "fetch", "build"):
        sp = sub.add_parser(name)
        if name == "compare":
            sp.add_argument("--slot", default="ban_noon", choices=list(TIME_SLOTS))
            sp.add_argument("--src", default="P001")
            sp.add_argument("--dst", default="P007")
        sp.add_argument("--provider", choices=list(PROVIDERS), default="tomtom")
        if name == "fetch":
            sp.add_argument("--dry-run", action="store_true")
            sp.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    {"diag": diag, "test": test, "compare": compare, "fetch": fetch, "build": build}[args.cmd](args)
