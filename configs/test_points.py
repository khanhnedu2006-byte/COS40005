"""Điểm A/B dùng để KIỂM TRA model với điểm "lạ" (không phải POI, không dùng để hiệu chỉnh ab_estimator).

Lấy ngẫu nhiên (seed 2026) trên đường xe lớn vào được, mỗi điểm một khu vực, đều ngoài vành đai.
Cùng với 5 hub → 10 điểm A/B kiểm tra; thời gian lấy bằng TomTom (fetch_routes.py --with-test-points)
→ data/final/test_legs.csv. Ngày chọn: 2026-11-09.
"""

TEST_POINTS = [
    {"point_id": "T01", "name": "Sơn Trà – An Hải", "coordinates": [16.09342, 108.23547]},
    {"point_id": "T02", "name": "Liên Chiểu", "coordinates": [16.06782, 108.14810]},
    {"point_id": "T03", "name": "Cẩm Lệ – Hòa Xuân", "coordinates": [16.00694, 108.19990]},
    {"point_id": "T04", "name": "Điện Ngọc – ven biển phía Nam", "coordinates": [15.96395, 108.28716]},
    {"point_id": "T05", "name": "Hòa Vang – phía Tây", "coordinates": [16.00561, 108.09314]},
]
