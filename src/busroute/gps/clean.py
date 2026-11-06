"""Bước 1 – Làm sạch log GPS hộp đen: bỏ trùng, khôi phục giây, lọc nhiễu, tách điểm dừng và chuyến."""

from os import PathLike

import numpy as np
import pandas as pd

from busroute.utils.geo import haversine_m

MAX_SPEED_KMH = 120  # tốc độ suy ra tối đa hợp lý cho xe 45 chỗ
FAR_KM = 100  # điểm cách trung vị quá xa thì loại
STOP_RADIUS_M = 50  # bán kính coi là đứng yên
STOP_MIN_MINUTES = 3  # đứng yên tối thiểu bao lâu thì tính là điểm dừng
TRIP_GAP_MINUTES = 10  # mất tín hiệu quá lâu thì tách chuyến


def read_log(path: str | PathLike[str]) -> pd.DataFrame:
    """Read a normalized GPS log, accepting a UTF-8 byte order mark."""
    return pd.read_csv(path, encoding="utf-8-sig")


def parse_time(df: pd.DataFrame) -> pd.Series:
    """Parse timestamps, distributing points within minutes if any lack seconds."""
    if (df["time_precision"] == "minute").any():
        return restore_seconds(df)
    return pd.to_datetime(df["timestamp"], format="%Y-%m-%d %H:%M:%S")


def restore_seconds(df: pd.DataFrame) -> pd.Series:
    """Spread points sharing a timestamp evenly over a minute in input order."""
    minute = pd.to_datetime(df["timestamp"], format="%Y-%m-%d %H:%M:%S")
    rank = df.groupby(minute).cumcount()
    count = df.groupby(minute)["timestamp"].transform("size")
    return minute + pd.to_timedelta(rank * 60 / count, unit="s")


def drop_speed_outliers(df: pd.DataFrame) -> pd.DataFrame:
    """Iteratively remove points whose speed from the preceding point is excessive."""
    while True:
        dist = haversine_m(df.lat.shift(), df.lon.shift(), df.lat, df.lon)
        dt = df.time.diff().dt.total_seconds()
        implied = dist / dt.replace(0, np.nan) * 3.6
        bad = implied > MAX_SPEED_KMH
        if not bad.any():
            return df
        df = df[~bad].reset_index(drop=True)


def detect_stops(df: pd.DataFrame) -> pd.DataFrame:
    """Summarize consecutive points near their first point as stops when long enough."""
    stops, i, n = [], 0, len(df)
    while i < n:
        j = i
        while (
            j + 1 < n
            and haversine_m(df.lat[i], df.lon[i], df.lat[j + 1], df.lon[j + 1])
            <= STOP_RADIUS_M
        ):
            j += 1
        minutes = (df.time[j] - df.time[i]).total_seconds() / 60
        if j > i and minutes >= STOP_MIN_MINUTES:
            stops.append(
                dict(
                    start=df.time[i],
                    end=df.time[j],
                    minutes=round(minutes, 1),
                    lat=df.lat[i : j + 1].mean(),
                    lon=df.lon[i : j + 1].mean(),
                )
            )
        i = j + 1
    return pd.DataFrame(stops)


def clean_file(
    path: str | PathLike[str], bus_id: str
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
    """Clean one normalized log and return points, stops, and summary statistics.

    Timestamps are sorted before spatial and speed filtering; trip IDs split at
    signal gaps longer than TRIP_GAP_MINUTES.

    Args:
        path: Path to a normalized GPS CSV.
        bus_id: Identifier to include in the cleaned points and statistics.

    Returns:
        A tuple of cleaned points, detected stops, and summary statistics.
    """
    raw = read_log(path)
    stats = {"bus_id": bus_id, "raw_rows": len(raw)}
    df = raw.drop_duplicates().reset_index(drop=True)
    stats["after_dedup"] = len(df)
    df = pd.DataFrame(
        {
            "time": parse_time(df),
            "lat": df.latitude,
            "lon": df.longitude,
            "device_speed": df.device_speed_kmh,
        }
    )
    df = df.sort_values("time", kind="stable").reset_index(drop=True)
    # Loại điểm quá xa vị trí trung vị (nhiễu GPS)
    far = haversine_m(df.lat, df.lon, df.lat.median(), df.lon.median()) > FAR_KM * 1000
    df = df[~far].reset_index(drop=True)
    df = drop_speed_outliers(df)
    stats["after_noise"] = len(df)
    # Tính lại quãng đường và tốc độ từ toạ độ (không tin cột của thiết bị)
    step = haversine_m(df.lat.shift(), df.lon.shift(), df.lat, df.lon).fillna(0)
    dt = df.time.diff().dt.total_seconds()
    df["gps_speed_kmh"] = (step / dt.replace(0, np.nan) * 3.6).round(1)
    df["trip_id"] = (dt / 60 > TRIP_GAP_MINUTES).cumsum()
    df["bus_id"] = bus_id
    stops = detect_stops(df)
    stats.update(
        distance_km=round(step.sum() / 1000, 1),
        trips=int(df.trip_id.nunique()),
        stops=len(stops),
        device_speed_errors=int((df.device_speed > MAX_SPEED_KMH).sum()),
    )
    return df, stops, stats
