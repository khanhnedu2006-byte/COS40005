"""Tính tốc độ thực tế theo (đoạn đường, khung giờ) và hệ số hiệu chỉnh so với giới hạn luật."""

from collections import defaultdict

from busroute.utils.timeslots import slot_of


def collect_edge_speeds(matched_df):
    """Group matched GPS speeds by directed edge and time slot.

    Args:
        matched_df: Matched points with time, u, v, key and gps_speed_kmh.

    Returns:
        A mapping from (u, v, key, slot) to observed speeds in km/h.
    """
    samples = defaultdict(list)
    for row in matched_df.itertuples():
        minute = row.time.hour * 60 + row.time.minute
        samples[(row.u, row.v, row.key, slot_of(minute))].append(row.gps_speed_kmh)
    return samples


def fit_calibration(samples, edge_limit_kmh, edge_limit_class):
    """Estimate speed-to-limit ratios by road class and time slot.

    Args:
        samples: Speed lists keyed by (u, v, key, slot).
        edge_limit_kmh: Legal speed limits keyed by (u, v, key).
        edge_limit_class: Limit class names keyed by (u, v, key).

    Returns:
        Ratios keyed by (limit class, slot). For an even number of samples,
        the upper middle sorted ratio is used.
    """
    ratios = defaultdict(list)
    for (u, v, k, slot), speeds in samples.items():
        limit = edge_limit_kmh[(u, v, k)]
        ratios[(edge_limit_class[(u, v, k)], slot)] += [s / limit for s in speeds]
    return {key: sorted(r)[len(r) // 2] for key, r in ratios.items() if r}
