"""Hiệu chỉnh tốc độ của ABEstimator cho khớp thời gian TomTom trong od_matrix.csv.

Người phụ trách: Nguyen Nam Khanh
Chạy: python scripts/calibrate_estimator.py            (cả 5 khung giờ)
      python scripts/calibrate_estimator.py --slot morning
Cách làm (mỗi khung giờ):
  1. Gắn 55 điểm vào đồ thị OSM; tìm đường ngắn nhất theo tốc độ hiện tại.
  2. Với mỗi chặng, cộng chiều dài theo 6 nhóm đường → thời gian = Σ (chiều dài nhóm × giây/mét nhóm) + phí cố định.
  3. Giải bình phương tối thiểu KHÔNG ÂM theo SAI SỐ TƯƠNG ĐỐI để khớp thời gian TomTom → tốc độ mới.
  4. Lặp 3 lần (đường ngắn nhất đổi khi tốc độ đổi).
Đánh giá: chặng POI→POI giữ lại 20 % + MỌI chặng có hub (hub không dùng để hiệu chỉnh = mô phỏng điểm A/B lạ).
Ghi: configs/estimator_params.json, reports/estimator_validation.csv
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import nnls

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from busroute.graph.ab_estimator import ABEstimator, GROUPS, SLOTS, BAN_SLOTS, PARAMS_FILE, load_default  # noqa: E402
from fetch_routes import load_points  # noqa: E402

ITERATIONS = 3
TEST_SHARE = 0.2
SEED = 42


def fit_slot(est, pts, od, slot, train_keys):
    sub = od[od.slot == slot]
    y_of = {(r.from_id, r.to_id): r.duration_s for r in sub.itertuples() if r.feasible == 1}
    for it in range(ITERATIONS):
        node = {p["id"]: est.snap(p["lat"], p["lon"], slot)[0] for p in pts}
        ids = [p["id"] for p in pts]
        legs = est.legs([node[i] for i in ids], [node[i] for i in ids], slot)
        rows, ys = [], []
        for a in ids:
            for b in ids:
                k = (a, b)
                if k not in train_keys or k not in y_of:
                    continue
                leg = legs.get((node[a], node[b]))
                if leg is None:
                    continue
                rows.append(np.append(est.group_lengths(leg[1]), 1.0))
                ys.append(y_of[k])
        A, y = np.array(rows), np.array(ys, float)
        coef, _ = nnls(A / y[:, None], np.ones(len(y)))     # sai số tương đối
        # Chặn dưới: không nhanh hơn 80 km/h trên mọi nhóm (tránh nghiệm vô lý cho nhóm ít dữ liệu)
        spm = np.maximum(coef[:-1], 3.6 / 80)
        params = est.params
        params["s_per_m"][slot] = {g: float(v) for g, v in zip(GROUPS, spm)}
        params["overhead_s"][slot] = float(coef[-1])
        est.set_params(params)
        kmh = {g: round(3.6 / v, 1) for g, v in zip(GROUPS, spm)}
        print(f"  {slot} lần {it + 1}: {len(y)} chặng, tốc độ km/h {kmh}, phí cố định {coef[-1]:.0f}s")


def evaluate(est, pts, od, slot, train_keys):
    sub = od[od.slot == slot]
    node = {p["id"]: est.snap(p["lat"], p["lon"], slot)[0] for p in pts}
    ids = [p["id"] for p in pts]
    legs = est.legs([node[i] for i in ids], [node[i] for i in ids], slot)
    banned = {p["id"] for p in pts if slot in BAN_SLOTS and est.endpoint_banned(p["lat"], p["lon"])}
    out = []
    for r in sub.itertuples():
        k = (r.from_id, r.to_id)
        leg = None if (r.from_id in banned or r.to_id in banned) else legs.get((node[r.from_id], node[r.to_id]))
        pred = est.leg_record(leg[0], leg[1], slot) if leg else None
        group = "train" if k in train_keys else ("hub" if "H" in (r.from_id[0] + r.to_id[0]) else "test_poi")
        out.append({"slot": slot, "group": group, "from_id": r.from_id, "to_id": r.to_id,
                    "tomtom_feasible": r.feasible, "est_feasible": int(pred is not None),
                    "tomtom_s": r.duration_s, "est_s": pred["duration_s"] if pred else 0})
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--slot", choices=SLOTS)
    args = ap.parse_args()
    t0 = time.time()
    pts = load_points()
    est = load_default([p for p in pts if p["type"] == "poi"])
    od = pd.read_csv(ROOT / "data" / "final" / "od_matrix.csv", encoding="utf-8-sig", keep_default_na=False)
    rng = np.random.default_rng(SEED)
    poi_ids = [p["id"] for p in pts if p["type"] == "poi"]
    pairs = [(a, b) for a in poi_ids for b in poi_ids if a != b]
    train_keys = {pairs[i] for i in rng.permutation(len(pairs))[int(TEST_SHARE * len(pairs)):]}
    slots = [args.slot] if args.slot else SLOTS
    rows = []
    for slot in slots:
        fit_slot(est, pts, od, slot, train_keys)
        rows += evaluate(est, pts, od, slot, train_keys)
        PARAMS_FILE.write_text(json.dumps(est.params, indent=2), encoding="utf-8")
    res = pd.DataFrame(rows)
    out = ROOT / "reports" / f"estimator_validation{'_' + args.slot if args.slot else ''}.csv"
    both = res[(res.tomtom_feasible == 1) & (res.est_feasible == 1)].copy()
    both["ape"] = (both.est_s - both.tomtom_s).abs() / both.tomtom_s
    summ = both.groupby(["slot", "group"]).agg(n=("ape", "size"), mape=("ape", "mean"),
                                               median_ape=("ape", "median")).round(3)
    agree = res.groupby(["slot"]).apply(lambda d: (d.tomtom_feasible == d.est_feasible).mean(),
                                        include_groups=False).round(3)
    summ.to_csv(out, encoding="utf-8-sig")
    print(summ.to_string())
    print("Khớp feasible với TomTom:", agree.to_dict())
    print(f"Đã ghi {PARAMS_FILE} và {out} ({time.time() - t0:.0f}s)")
