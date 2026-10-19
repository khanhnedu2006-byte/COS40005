"""Vùng cấm theo giờ. Quy định 2019: cấm xe khách trên 30 chỗ vào vành đai trung tâm Đà Nẵng."""

TIME_BANNED_AREAS = [
    {
        "name": "Central ring",
        # TODO: vẽ đa giác theo các tuyến bao quanh vành đai (lat, lon), theo chiều kim đồng hồ
        "polygon": [],
        "windows": [("10:30", "12:00"), ("16:30", "19:00")],
        "reason": "No passenger vehicles over 30 seats in the central ring (Da Nang, 2019)",
    },
]

# Ranh giới thành phố Đà Nẵng mới (sau sáp nhập 01/07/2025) nếu Nominatim chưa cập nhật
BOUNDARY_POLYGON = []   # TODO: danh sách (lat, lon)
