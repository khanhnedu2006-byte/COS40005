"""Hàm hình học dùng chung."""
import re
import unicodedata
import numpy as np


def haversine_m(lat1, lon1, lat2, lon2):
    # Khoảng cách mặt cầu (mét); nhận số hoặc mảng numpy
    p1, p2 = np.radians(lat1), np.radians(lat2)
    a = np.sin((p2 - p1) / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(np.radians(lon2 - lon1) / 2) ** 2
    return 2 * 6371000 * np.arcsin(np.sqrt(a))


def point_in_polygon(lat, lon, polygon):
    # Thuật toán ray casting; polygon là danh sách (lat, lon). Đa giác rỗng → False
    inside, n = False, len(polygon)
    for i in range(n):
        y1, x1 = polygon[i]
        y2, x2 = polygon[(i + 1) % n]
        if (y1 > lat) != (y2 > lat):
            x_cross = x1 + (lat - y1) * (x2 - x1) / (y2 - y1)
            if lon < x_cross:
                inside = not inside
    return inside


def normalize_road_name(name):
    # Chuẩn hoá tên đường để so khớp: bỏ tiền tố "Đường", gộp khoảng trắng, viết thường, chuẩn NFC
    if not name:
        return ""
    name = unicodedata.normalize("NFC", str(name)).strip().lower()
    # Bỏ tiền tố "đường" trừ khi nó là một phần của tên (đường tỉnh 609, đường cao tốc…)
    name = re.sub(r"^(đường|duong)\s+(?!(tỉnh|huyện|xã|cao tốc|sắt|tinh|huyen)\b)", "", name)
    return re.sub(r"\s+", " ", name)
