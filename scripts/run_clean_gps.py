"""Chạy Bước 1 cho mọi file log GPS và ẩn danh biển số."""
import argparse, glob, json, os, re

import pandas as pd

from busroute.gps.clean import clean_file

# Biển số → mã ẩn danh (khớp với slide). Biển số đã chuẩn hoá: bỏ khoảng trắng, dấu "-" và "."
PLATE_TO_BUS = {"43F00409": "bus_01", "43H11687": "bus_02", "43H14909": "bus_03",
                "43B05169": "bus_04", "43B05367": "bus_05"}


def bus_id_from_filename(path):
    key = re.sub(r"[\s\-.]", "", os.path.basename(path)).upper()
    for plate, bus in PLATE_TO_BUS.items():
        if plate in key:
            return bus
    return None


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    all_stats = []
    for path in sorted(glob.glob(os.path.join(args.src, "**", "*.csv"), recursive=True)):
        bus = bus_id_from_filename(path)
        if bus is None:
            print("Bỏ qua (không nhận ra biển số):", path)
            continue
        df, stops, stats = clean_file(path, bus)
        # Tên output = mã xe + tên file gốc đã làm sạch, để 1 xe có nhiều file không ghi đè nhau
        stem = re.sub(r"[^0-9A-Za-z]+", "_", os.path.splitext(os.path.basename(path))[0]).strip("_")
        tag = f"{bus}__{stem}"
        df.to_csv(os.path.join(args.out, f"{tag}_clean.csv"), index=False)
        stops.to_csv(os.path.join(args.out, f"{tag}_stops.csv"), index=False)
        all_stats.append({**stats, "file": os.path.basename(path)})
    print(pd.DataFrame(all_stats).to_string(index=False))
    json.dump(all_stats, open(os.path.join(args.out, "stats.json"), "w"), indent=1, default=str)
