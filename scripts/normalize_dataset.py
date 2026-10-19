"""Chuẩn hoá mọi file log GPS (bản export cũ và mới) về cùng một định dạng, mỗi cột một thông tin."""
import argparse, glob, os, re

import pandas as pd

# Tên cột của hai định dạng export → tên chuẩn
COLUMN_ALIASES = {"Ngày tháng": "timestamp", "Vĩ độ": "latitude", "Kinh độ": "longitude"}
# Biển số chuẩn hoá → mã ẩn danh (khớp với slide)
PLATE_TO_BUS = {"43F00409": "bus_01", "43H11687": "bus_02", "43H14909": "bus_03",
                "43B05169": "bus_04", "43B05367": "bus_05"}
OUTPUT_COLUMNS = ["bus_id", "plate", "timestamp", "latitude", "longitude",
                  "device_speed_kmh", "device_km", "time_precision"]


def plate_from_filename(path):
    # "xe 43B - 05169  .csv", "43H-149.09.csv" → "43B-05169", "43H-14909"
    key = re.sub(r"[^0-9A-Za-z]", "", os.path.basename(path)).upper()
    for plate in PLATE_TO_BUS:
        if plate in key:
            return plate
    return None


def normalize_file(path):
    df = pd.read_csv(path, encoding="utf-8-sig")
    df.columns = [c.strip() for c in df.columns]
    df = df.rename(columns=COLUMN_ALIASES)
    # Bản mới có giây; bản cũ chỉ tới phút (MM/DD/YY H:MM)
    ts = pd.to_datetime(df["timestamp"], format="%Y-%m-%d %H:%M:%S", errors="coerce")
    precision = "second"
    if ts.isna().any():
        ts = pd.to_datetime(df["timestamp"], format="%m/%d/%y %H:%M")
        precision = "minute"
    # Tách cột gộp "V / KM / L": V = tốc độ thiết bị (km/h), KM = quãng đường thiết bị báo, L = nhiên liệu (luôn trống → bỏ)
    parts = df["V/KM/L"].astype(str).str.split("/", expand=True)
    plate = plate_from_filename(path)
    out = pd.DataFrame({
        "bus_id": PLATE_TO_BUS[plate],
        "plate": f"{plate[:3]}-{plate[3:]}",
        "timestamp": ts.dt.strftime("%Y-%m-%d %H:%M:%S"),
        "latitude": pd.to_numeric(df["latitude"], errors="coerce"),
        "longitude": pd.to_numeric(df["longitude"], errors="coerce"),
        "device_speed_kmh": pd.to_numeric(parts[0].str.strip(), errors="coerce"),
        "device_km": pd.to_numeric(parts[1].str.strip(), errors="coerce"),
        "time_precision": precision,
    })[OUTPUT_COLUMNS]
    # Bỏ dòng trùng hoàn toàn và dòng thiếu toạ độ – không mang thông tin
    out = out.drop_duplicates().dropna(subset=["latitude", "longitude"]).reset_index(drop=True)
    return out, len(df)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="dataset")
    ap.add_argument("--out", default="dataset")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    for path in sorted(glob.glob(os.path.join(args.src, "*.csv"))):
        if plate_from_filename(path) is None:
            print("Bỏ qua (không nhận ra biển số):", path)
            continue
        out, n_raw = normalize_file(path)
        day = out["timestamp"].str[:10].iloc[0]
        name = f"{out['plate'].iloc[0]}_{day}.csv"
        out.to_csv(os.path.join(args.out, name), index=False, encoding="utf-8")
        print(f"{os.path.basename(path):28s} → {name:26s} {n_raw:5d} → {len(out):5d} dòng")
