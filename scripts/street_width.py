"""Gắn chiều rộng mặt đường thật (CSDL đường phố Đà Nẵng) vào các đoạn đường OSM.

Người phụ trách: Nguyen Nam Khanh
Nguồn: Cổng Thông tin điện tử thành phố Đà Nẵng – CSDL đường phố
       (https://danang.gov.vn/ttdt/csdl/duong-pho), lưu tại data/references/danang_duong_pho.csv.
       Bắt buộc ghi nguồn khi dùng lại.
Cách ghép:
  1. Chuẩn hoá tên đường (bỏ "Đường", ghi chú trong ngoặc, thống nhất dấu thanh hoà/hòa…).
  2. Chỉ ghép đoạn OSM nằm trong vùng Đà Nẵng cũ (danh sách chỉ có phố Đà Nẵng cũ, tránh trùng tên
     với Quảng Nam cũ / Hội An).
  3. Ô ghi nhiều bề rộng ("7,5 & 5,5": tuyến có đoạn rộng đoạn hẹp) → lấy số NHỎ NHẤT cho chắc xe 45 chỗ
     qua được; "2 x 10,5" (hai chiều có dải phân cách) → cộng lại 21 m.
  4. Tên có nhiều đoạn với bề rộng khác nhau: tìm 2 ngã giao với phố "điểm đầu" và "điểm cuối" trên OSM,
     đoạn OSM nằm giữa 2 ngã giao nhận bề rộng của đoạn đó; phần còn lại nhận bề rộng của đoạn dài nhất.
"""
import csv
import re
import unicodedata
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
CSV_PATH = ROOT / "data" / "references" / "danang_duong_pho.csv"
# Vùng Đà Nẵng cũ (6 quận + Hòa Vang), xấp xỉ bằng khung toạ độ
OLD_DN_BBOX = (15.955, 16.25, 107.80, 108.36)       # lat_min, lat_max, lon_min, lon_max
TONE = {"oà": "òa", "oá": "óa", "oả": "ỏa", "oã": "õa", "oạ": "ọa", "oè": "òe", "oé": "óe", "oẻ": "ỏe",
        "oẽ": "õe", "oẹ": "ọe", "uỳ": "ùy", "uý": "úy", "uỷ": "ủy", "uỹ": "ũy", "uỵ": "ụy"}


def norm_name(s):
    # Chuẩn hoá tên để so khớp: chữ thường, NFC, bỏ ngoặc, bỏ tiền tố "đường", thống nhất dấu
    s = unicodedata.normalize("NFC", s or "").lower()
    s = re.sub(r"\(.*?\)|\(.*$", " ", s)
    s = re.sub(r"\bl\s*=.*$", " ", s)
    s = re.sub(r"^(điều chỉnh|đổi tên thành)\s+", "", s.strip())
    s = re.sub(r"^đường\s+", "", s.strip())
    for a, b in TONE.items():
        s = s.replace(a, b)
    s = s.replace("–", "-")
    s = re.sub(r"\b(\d{1,2})\s*/\s*(\d{1,2})\b", r"\1 tháng \2", s)     # 2/9 → 2 tháng 9
    s = re.sub(r"^(tỉnh|tỉnh lộ|đt)\s*(\d+\w?)$", r"đt\2", s)                 # Đường tỉnh 601 = ĐT601
    s = re.sub(r"^(huyện|đh)\s*(\d+)$", r"đh\2", s)                            # Đường huyện 5 = ĐH5
    return re.sub(r"\s+", " ", s).strip(" -.")


def parse_width(s):
    # "10,5" → 10.5; "2 x 10.5" → 21 (2 làn xe riêng); "7.5 & 5.5" → (5.5, 7.5); "3,0-5,0" → trung bình
    s = (s or "").replace("×", "x")
    s = re.sub(r"(\d),(\d)", r"\1.\2", s)
    s = re.sub(r"\(.*?\)", "", s)
    vals = []
    for part in re.split(r"&|;|,|\+|\bvà\b", s):
        part = part.strip()
        if not part:
            continue
        m = re.match(r"^(\d+)\s*x\s*(\d+(?:\.\d+)?)$", part) or re.match(r"^(\d+(?:\.\d+)?)\s*x\s*(\d+)$", part)
        if m:
            a, b = float(m.group(1)), float(m.group(2))
            vals.append(a * b)
            continue
        nums = [float(x) for x in re.findall(r"\d+(?:\.\d+)?", part)]
        if len(nums) == 2 and "-" in part:
            vals.append(sum(nums) / 2)
        elif nums:
            vals.append(nums[0])
    vals = [v for v in vals if 2 <= v <= 60]
    return (min(vals), max(vals)) if vals else (None, None)


def load_records():
    rows, prev = [], ""
    with open(CSV_PATH, encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            name = (r["Tên đường"] or "").strip() or prev        # ô gộp: tên trống = phố ở dòng trên
            prev = name
            wmin, wmax = parse_width(r["Chiều rộng mặt (m)"])
            if wmax is None:
                continue
            try:
                length = float((r["Chiều dài (Km)"] or "0").replace(",", "."))
            except ValueError:
                length = 0.0
            rows.append({"name": name, "key": norm_name(name), "start": norm_name(r["Điểm đầu"]),
                         "end": norm_name(r["Điểm cuối"]), "w_min": wmin, "w_max": wmax, "len_km": length})
    return rows


def assign_widths(edges, lat, lon, idx):
    """edges: list dict có 'name','u','v' (mã OSM); lat/lon theo chỉ số nút; idx: mã OSM → chỉ số.
    Trả về list (bề rộng nhỏ nhất hoặc None) theo thứ tự edges và thống kê."""
    recs = load_records()
    by_key = defaultdict(list)
    for r in recs:
        by_key[r["key"]].append(r)
    la0, la1, lo0, lo1 = OLD_DN_BBOX
    ekey = []
    for e in edges:
        a, b = idx[e["u"]], idx[e["v"]]
        y, x = (lat[a] + lat[b]) / 2, (lon[a] + lon[b]) / 2
        ok = la0 <= y <= la1 and lo0 <= x <= lo1
        ekey.append(norm_name(e["name"]) if ok and e["name"] else "")
    # nút thuộc từng tên phố (để tìm ngã giao)
    nodes_of = defaultdict(set)
    for e, k in zip(edges, ekey):
        if k:
            nodes_of[k].update((idx[e["u"]], idx[e["v"]]))
    out = [None] * len(edges)
    edges_of = defaultdict(list)
    for i, k in enumerate(ekey):
        if k in by_key:
            edges_of[k].append(i)
    n_seg_matched = 0
    for k, ids in edges_of.items():
        rs = by_key[k]
        if len({r["w_min"] for r in rs}) == 1:
            for i in ids:
                out[i] = rs[0]["w_min"]
            continue
        # nhiều đoạn bề rộng khác nhau → định vị theo 2 ngã giao
        default = max(rs, key=lambda r: r["len_km"])["w_min"]
        for i in ids:
            out[i] = default
        street_nodes = nodes_of[k]
        for r in rs:
            ends = []
            for cross in (r["start"], r["end"]):
                jn = street_nodes & nodes_of.get(cross, set())
                if jn:
                    ends.append(np.array([[lat[n], lon[n]] for n in jn]).mean(axis=0))
            if len(ends) < 2:
                continue
            p, q = ends
            kx = np.cos(np.radians(p[0]))
            d = np.array([(q[1] - p[1]) * kx, q[0] - p[0]])
            L2 = float(d @ d)
            if L2 < 1e-10:
                continue
            for i in ids:
                a, b = idx[edges[i]["u"]], idx[edges[i]["v"]]
                m = np.array([((lon[a] + lon[b]) / 2 - p[1]) * kx, (lat[a] + lat[b]) / 2 - p[0]])
                t = float(m @ d) / L2
                off = abs(m[0] * d[1] - m[1] * d[0]) / np.sqrt(L2) * 111000
                if 0 <= t <= 1 and off < 150:
                    out[i] = r["w_min"]
            n_seg_matched += 1
    stats = {"records": len(recs), "street_names": len(by_key),
             "names_matched": len(edges_of), "segments_located": n_seg_matched,
             "edges_in_old_dn": sum(1 for k in ekey if k), "edges_with_width": sum(w is not None for w in out)}
    return out, stats
