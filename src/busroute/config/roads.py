"""Whitelist đường cho xe 45 chỗ. Mặc định MỌI đường bị cấm, trừ các đường dưới đây."""

# Đường chính được phép đi – so khớp tên sau khi normalize_road_name()
# TODO: xác nhận từng tuyến với tài xế trước khi dùng
MAIN_ROADS = [
    "Võ Nguyên Giáp", "Trường Sa", "Hoàng Sa",          # trục biển
    "Nguyễn Văn Linh", "Bạch Đằng", "Trần Hưng Đạo",     # trung tâm, hai bờ sông Hàn
    "Nguyễn Tất Thành", "Điện Biên Phủ", "Ngô Quyền",
    "Lê Duẩn", "2 Tháng 9", "3 Tháng 2",
    "Lê Văn Hiến", "Võ Chí Công",                        # hướng Ngũ Hành Sơn, Hội An
    "Quốc lộ 1A", "Quốc lộ 14B", "ĐT602",                # hướng Bà Nà và phía tây
]

# Đoạn ngắn ngoài mạng chính nhưng bắt buộc để tiếp cận POI (theo OSM way ID)
# Mỗi đoạn phải được kiểm tra (tài xế hoặc Street View) và ghi ngày kiểm tra
ACCESS_SPURS = [
    # {"osm_way_id": 123456789, "poi_id": "P06", "checked_by": "driver", "checked_on": "2026-10-20"},
]

# Đường bị cấm tuyệt đối dù có tên trong MAIN_ROADS (ví dụ một đoạn đang thi công)
BLOCKED_WAYS = [
    # {"osm_way_id": 987654321, "reason": "construction", "until": "2026-12-31"},
]
