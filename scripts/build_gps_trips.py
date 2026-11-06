"""Tách log GPS thành các chặng chạy giữa hai điểm dừng → data/interim/gps_trips.csv

Người phụ trách: Nguyen Nam Khanh
Chạy: python scripts/build_gps_trips.py
Dùng để kiểm chứng thời gian TomTom: cùng điểm đi/đến, cùng thứ + giờ xuất phát,
so thời gian thật của xe 45 chỗ với thời gian TomTom dự báo (lệnh fetch_routes.py validate).
Kết quả chứa toạ độ GPS của xe → để trong data/interim/ (đã .gitignore), KHÔNG commit.
"""

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from busroute.gps.clean import clean_file  # noqa: E402
from busroute.utils.geo import haversine_m  # noqa: E402
from busroute.utils.timeslots import slot_of  # noqa: E402

DATASET = ROOT / "dataset"
OUT = ROOT / "data" / "interim" / "gps_trips.csv"
MIN_KM = (
    2.0  # chặng ngắn hơn 2 km (đường chim bay) bỏ qua: sai số phút làm lệch quá nhiều
)
MIN_MINUTES = 5


def legs_of(df, stops, plate, date):
    # Chặng = từ lúc rời điểm dừng k tới lúc tới điểm dừng k+1 (cùng một chuyến, không mất tín hiệu)
    legs = []
    for k in range(len(stops) - 1):
        s0, s1 = stops.iloc[k], stops.iloc[k + 1]
        seg = df[(df.time >= s0.end) & (df.time <= s1.start)]
        if seg.empty or seg.trip_id.nunique() > 1:
            continue
        minutes = (s1.start - s0.end).total_seconds() / 60
        crow_km = haversine_m(s0.lat, s0.lon, s1.lat, s1.lon) / 1000
        path_km = (
            haversine_m(seg.lat.shift(), seg.lon.shift(), seg.lat, seg.lon)
            .fillna(0)
            .sum()
            / 1000
        )
        if crow_km < MIN_KM or minutes < MIN_MINUTES:
            continue
        legs.append(
            {
                "plate": plate,
                "date": date,
                "depart": s0.end.strftime("%Y-%m-%d %H:%M"),
                "weekday": s0.end.weekday(),
                "slot": slot_of(s0.end.hour * 60 + s0.end.minute),
                "o_lat": round(s0.lat, 5),
                "o_lon": round(s0.lon, 5),
                "d_lat": round(s1.lat, 5),
                "d_lon": round(s1.lon, 5),
                "gps_minutes": round(minutes, 1),
                "gps_path_km": round(path_km, 1),
                "crow_km": round(crow_km, 1),
                "n_points": len(seg),
            }
        )
    return legs


if __name__ == "__main__":
    rows = []
    for f in sorted(DATASET.glob("*.csv")):
        plate, date = f.stem.split("_")
        df, stops, stats = clean_file(f, plate)
        legs = legs_of(df, stops, plate, date) if len(stops) > 1 else []
        print(
            f"{f.name}: {stats['after_noise']} điểm, {stats['stops']} điểm dừng → {len(legs)} chặng"
        )
        rows += legs
    OUT.parent.mkdir(parents=True, exist_ok=True)
    out = pd.DataFrame(rows)
    out.to_csv(OUT, index=False, encoding="utf-8-sig")
    print(f"\nĐã ghi {OUT}: {len(out)} chặng")
    if len(out):
        print(
            out.groupby("slot")
            .agg(
                n=("gps_minutes", "size"),
                median_min=("gps_minutes", "median"),
                median_km=("gps_path_km", "median"),
            )
            .to_string()
        )
