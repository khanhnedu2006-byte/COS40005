"""Tính tốc độ thực tế theo (đoạn đường, khung giờ) và hệ số hiệu chỉnh so với giới hạn luật."""
from collections import defaultdict

from busroute.utils.timeslots import slot_of


def collect_edge_speeds(matched_df):
    # {(u, v, key, slot): [tốc độ km/h, ...]}
    samples = defaultdict(list)
    for row in matched_df.itertuples():
        minute = row.time.hour * 60 + row.time.minute
        samples[(row.u, row.v, row.key, slot_of(minute))].append(row.gps_speed_kmh)
    return samples


def fit_calibration(samples, edge_limit_kmh, edge_limit_class):
    # Hệ số = trung vị (tốc độ thực / giới hạn luật), nhóm theo (limit_class, slot)
    ratios = defaultdict(list)
    for (u, v, k, slot), speeds in samples.items():
        limit = edge_limit_kmh[(u, v, k)]
        ratios[(edge_limit_class[(u, v, k)], slot)] += [s / limit for s in speeds]
    return {key: sorted(r)[len(r) // 2] for key, r in ratios.items() if r}
