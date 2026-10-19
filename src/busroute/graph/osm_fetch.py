"""Bước 2 – Lấy đồ thị đường từ OSM qua Overpass API. OSM CHỈ là bản đồ nền, không lọc đường ở đây."""
import os
import time

import osmnx as ox
from shapely.geometry import Polygon

from busroute.config.restrictions import BOUNDARY_POLYGON

# Các máy chủ Overpass công khai – máy chính hay quá tải, lần lượt thử máy dự phòng
OVERPASS_ENDPOINTS = [
    "https://overpass-api.de/api",
    "https://overpass.kumi.systems/api",
    "https://overpass.private.coffee/api",
    "https://maps.mail.ru/osm/tools/overpass/api",
]
MAX_ROUNDS = 4          # số vòng thử lại toàn bộ danh sách máy chủ
FIRST_WAIT_S = 30       # thời gian chờ ban đầu, gấp đôi sau mỗi vòng (exponential backoff)


def _download(network_type, bbox, place):
    # Thứ tự ưu tiên vùng lấy bản đồ: BOUNDARY_POLYGON → bbox → place (place cần Nominatim)
    if BOUNDARY_POLYGON:
        poly = Polygon([(lon, lat) for lat, lon in BOUNDARY_POLYGON])   # shapely dùng (lon, lat)
        return ox.graph_from_polygon(poly, network_type=network_type, simplify=True)
    if bbox:
        # bbox = [west, south, east, north] (osmnx 2.x)
        return ox.graph_from_bbox(tuple(bbox), network_type=network_type, simplify=True)
    return ox.graph_from_place(place, network_type=network_type, simplify=True)


def fetch_road_graph(place=None, network_type="drive", cache_dir="data/cache", bbox=None):
    os.makedirs(cache_dir, exist_ok=True)
    graph_file = os.path.join(cache_dir, f"road_graph_{network_type}.graphml")
    if os.path.exists(graph_file):
        # Đã tải trước đó → đọc file, không gọi API nữa
        return ox.load_graphml(graph_file)
    # Cache từng truy vấn con: lỗi giữa chừng thì lần sau chỉ tải tiếp phần còn thiếu
    ox.settings.use_cache = True
    ox.settings.cache_folder = cache_dir
    ox.settings.requests_timeout = 300
    wait, last_error = FIRST_WAIT_S, None
    for round_no in range(1, MAX_ROUNDS + 1):
        for url in OVERPASS_ENDPOINTS:
            ox.settings.overpass_url = url
            try:
                print(f"[vòng {round_no}] tải bản đồ từ {url} ...")
                G = _download(network_type, bbox, place)
                # Giữ thành phần liên thông mạnh lớn nhất để mọi node đều đi tới được
                G = ox.truncate.largest_component(G, strongly=True)
                ox.save_graphml(G, graph_file)
                return G
            except Exception as err:   # quá tải (502/504), hết thời gian, mất kết nối
                last_error = err
                print(f"  lỗi: {str(err).splitlines()[0][:120]}")
        print(f"Mọi máy chủ đều lỗi, chờ {wait}s rồi thử lại...")
        time.sleep(wait)
        wait *= 2
    raise RuntimeError("Không tải được bản đồ OSM sau nhiều lần thử") from last_error
