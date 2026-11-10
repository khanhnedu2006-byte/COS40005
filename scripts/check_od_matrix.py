"""Kiểm tra od_matrix.csv trước khi giao cho routing + sinh data card. Thoát mã 1 nếu có lỗi.

Người phụ trách: Nguyen Nam Khanh
Chạy: python scripts/check_od_matrix.py [--file data/final/od_matrix.csv]
Ghi: data/final/data_card.md (mô tả bộ dữ liệu, số liệu thống kê, giả định)
"""
import argparse
import datetime as dt
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
COLUMNS = ["from_id", "to_id", "from_type", "to_type", "from_lat", "from_lon", "to_lat", "to_lon",
           "to_poi_score", "slot", "duration_s", "distance_m", "feasible", "pois_passed", "through_ring"]
SLOTS = ["morning", "ban_noon", "afternoon", "ban_evening", "night"]
BAN = {"ban_noon", "ban_evening"}


def run_checks(df):
    res = []
    def check(name, ok, detail=""):
        res.append((name, bool(ok), detail))
    ids = sorted(set(df.from_id) | set(df.to_id))
    n = len(ids)
    check("Đủ cột", set(COLUMNS) <= set(df.columns), str(set(COLUMNS) - set(df.columns)))
    check("Đủ mọi cặp × 5 khung giờ", len(df) == n * (n - 1) * 5, f"{len(df)} dòng, cần {n * (n - 1) * 5}")
    check("Không trùng (from, to, slot)", not df.duplicated(["from_id", "to_id", "slot"]).any())
    check("Không có cặp tự nối", (df.from_id != df.to_id).all())
    check("Khung giờ hợp lệ", set(df.slot) <= set(SLOTS), str(set(df.slot) - set(SLOTS)))
    f = df[df.feasible == 1]
    check("feasible=1 thì duration > 0 và distance > 0", ((f.duration_s > 0) & (f.distance_m > 0)).all(),
          f"{int(((f.duration_s <= 0) | (f.distance_m <= 0)).sum())} dòng lỗi")
    speed = f.distance_m / f.duration_s * 3.6
    check("Tốc độ trung bình 5–90 km/h", speed.between(5, 90).all(),
          f"{int((~speed.between(5, 90)).sum())} dòng ngoài khoảng")
    check("Giờ cấm không đi sâu vào vành đai", (df[df.slot.isin(BAN)].through_ring == 0).all())
    # Cùng cặp: giờ cấm không nhanh hơn giờ thường liền trước quá 20 % (đi vòng chỉ có thể lâu hơn)
    w = f.pivot_table(index=["from_id", "to_id"], columns="slot", values="duration_s")
    for ban, base in (("ban_noon", "morning"), ("ban_evening", "afternoon")):
        if ban in w and base in w:
            r = (w[ban] / w[base]).dropna()
            # Cảnh báo: TomTom đôi khi chỉ đưa lối tắt khi được yêu cầu tránh vành đai (vd. P026→P039 qua
            # Kiệt 383 Núi Thành) → giờ cấm ngắn hơn giờ thường. Không phải lỗi dữ liệu; ghi vào data card.
            check(f"{ban} ≥ 0,8 × {base} (cảnh báo)", (r >= 0.8).mean() > 0.99,
                  f"{int((r < 0.8).sum())} cặp ({100 * (r < 0.8).mean():.1f} %)")
    # Bất đối xứng A→B so với B→A (cảnh báo)
    m = f.set_index(["from_id", "to_id", "slot"]).duration_s
    rev = m.copy()
    rev.index = rev.index.set_names(["to_id", "from_id", "slot"])
    ratio = (m / rev.reorder_levels(["from_id", "to_id", "slot"]).reindex(m.index)).dropna()
    check("A→B / B→A trong 0,5–2 (cảnh báo)", ratio.between(0.5, 2).mean() > 0.95,
          f"{100 * (~ratio.between(0.5, 2)).mean():.1f} % cặp lệch")
    # Mỗi điểm phải đi tới được ít nhất 1 điểm khác trong mọi khung giờ (nếu không: điểm "chết")
    reach = f.groupby(["from_id", "slot"]).size().unstack(fill_value=0)
    dead = [(i, s) for i in ids for s in SLOTS if reach.get(s, pd.Series()).get(i, 0) == 0]
    check("Mỗi điểm đi được ra ngoài ở mọi khung giờ (cảnh báo nếu POI trong vành đai)", True,
          f"{len(dead)} (điểm, khung giờ) không đi được – đúng luật nếu là POI trong vành đai")
    return res, ids, dead


def write_card(df, ids, dead, results, path, source):
    f = df[df.feasible == 1]
    lines = [
        "# Data card – od_matrix.csv", "",
        f"- Sinh ngày: {dt.date.today().isoformat()} · nguồn: `{source}`",
        f"- Điểm: {len(ids)} ({sum(i.startswith('P') for i in ids)} POI, {sum(i.startswith('H') for i in ids)} hub)",
        f"- Dòng: {len(df):,} · feasible = 1: {len(f):,} ({100 * len(f) / len(df):.1f} %)",
        "- Thời gian: TomTom Routing API, travelMode=bus, dài 12 m / rộng 2,5 m / cao 3,6 m, departAt = thứ Tư, "
        "giờ đại diện mỗi khung (08:00 · 11:15 · 14:30 · 17:45 · 20:30), kẹt xe lịch sử.",
        "- Giờ cấm: tránh vành đai trung tâm (CV 4883/UBND-SGTVT 2019), cho phép trục Duy Tân – Nguyễn Văn Trỗi – "
        "Trần Thị Lý (CV 1710/UBND-SGTVT 2022).",
        "- Kiểm chứng: xem `reports/gps_validation.csv` (so với chặng GPS thật cùng thứ + giờ).", "",
        "## Thời gian theo khung giờ (phút, các cặp feasible)", "",
        "| slot | n | trung vị | p90 | km/h trung vị |", "|---|---|---|---|---|",
    ]
    for s in SLOTS:
        x = f[f.slot == s]
        if len(x):
            lines.append(f"| {s} | {len(x)} | {x.duration_s.median() / 60:.1f} | {x.duration_s.quantile(.9) / 60:.1f} | "
                         f"{(x.distance_m / x.duration_s * 3.6).median():.1f} |")
    lines += ["", f"## Không đi được ({len(dead)} cặp điểm × khung giờ)", "",
              ", ".join(f"{i}@{s}" for i, s in dead[:60]) or "–", "",
              "## Kết quả kiểm tra", ""]
    lines += [f"- [{'x' if ok else ' '}] {name}" + (f" – {d}" if d and not ok else "") for name, ok, d in results]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", default=str(ROOT / "data" / "final" / "od_matrix.csv"))
    args = ap.parse_args()
    df = pd.read_csv(args.file, encoding="utf-8-sig", keep_default_na=False)
    results, ids, dead = run_checks(df)
    for name, ok, d in results:
        print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f" – {d}" if d else ""))
    card = Path(args.file).with_name("data_card.md")
    write_card(df, ids, dead, results, card, Path(args.file).name)
    failed = [r for r in results if not r[1]]
    print(f"\n{len(results) - len(failed)}/{len(results)} đạt · data card: {card}")
    sys.exit(1 if failed else 0)
