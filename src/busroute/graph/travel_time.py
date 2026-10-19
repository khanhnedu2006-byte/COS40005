"""Bước 4 – Thời gian đi mỗi đoạn theo khung giờ: ưu tiên GPS, không có thì giới hạn luật × hệ số."""
from statistics import median

from busroute.config.speeds import (LEGAL_LIMIT_KMH, DEFAULT_LIMIT_CLASS, ROAD_LIMIT_CLASS,
                                    CALIBRATION, DEFAULT_CALIBRATION, MIN_GPS_SAMPLES)
from busroute.utils.geo import normalize_road_name
from busroute.utils.timeslots import TIME_SLOTS

_LIMIT_CLASS_BY_NAME = {normalize_road_name(k): v for k, v in ROAD_LIMIT_CLASS.items()}


def edge_names(data):
    # Sau simplify, OSMnx có thể gộp nhiều tên thành list
    name = data.get("name")
    return name if isinstance(name, list) else [name]


def limit_class_of(data):
    for name in edge_names(data):
        cls = _LIMIT_CLASS_BY_NAME.get(normalize_road_name(name))
        if cls:
            return cls
    return DEFAULT_LIMIT_CLASS


def add_travel_times(G, gps_samples=None):
    # Ghi vào mỗi cạnh: limit_kmh, speed_<slot>, travel_time_<slot> (giây), verified_by_gps
    gps_samples = gps_samples or {}
    for u, v, k, data in G.edges(keys=True, data=True):
        cls = limit_class_of(data)
        limit = LEGAL_LIMIT_KMH[cls]
        data.update(limit_class=cls, limit_kmh=limit, verified_by_gps=False)
        for slot in TIME_SLOTS:
            samples = gps_samples.get((u, v, k, slot), [])
            if len(samples) >= MIN_GPS_SAMPLES:
                speed = min(median(samples), limit)          # ưu tiên 1: GPS thật
                data["verified_by_gps"] = True
            else:
                speed = limit * CALIBRATION.get((cls, slot), DEFAULT_CALIBRATION)   # ưu tiên 2
            data[f"speed_{slot}"] = speed
            data[f"travel_time_{slot}"] = data["length"] / (speed / 3.6)
    return G
