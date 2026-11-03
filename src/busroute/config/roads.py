"""Whitelist đường cho xe 45 chỗ. Mặc định MỌI đường bị cấm, trừ các đường dưới đây."""

# Đường chính được phép đi – so khớp tên sau khi normalize_road_name()
# TODO: xác nhận từng tuyến với tài xế trước khi dùng
MAIN_ROADS = [
    # --- Trục biển / Sơn Trà / Mỹ Khê ---
    "Võ Nguyên Giáp", "Trường Sa", "Hoàng Sa",
    "Phạm Văn Đồng", "Hồ Nghinh", "Nguyễn Văn Thoại",
    "Ngô Quyền", "Võ Văn Kiệt", "Như Nguyệt",
    "Dương Đình Nghệ", "Lê Văn Duyệt", "Nguyễn Công Trứ",   

    # --- Trung tâm, hai bờ sông Hàn ---
    "Nguyễn Văn Linh", "Bạch Đằng", "Trần Hưng Đạo",
    "Nguyễn Tất Thành", "Điện Biên Phủ",
    "Lê Duẩn", "2 Tháng 9", "3 Tháng 2",
    "Hùng Vương", "Lê Lợi", "Quang Trung", "Trần Phú",
    "Phan Châu Trinh", "Nguyễn Chí Thanh", "Hoàng Diệu",
    "Duy Tân", "Yên Bái", "Hải Phòng",
    "Trần Thị Lý", "Trường Chinh", "Cách Mạng Tháng 8",
    "Nguyễn Hữu Thọ", "Hoàng Văn Thái",
    "Nguyễn Tri Phương", "Ông Ích Khiêm", "Trưng Nữ Vương",  
    "Hàm Nghi", "Lê Đình Lý", "Nguyễn Hữu Cảnh", "Nguyễn Văn Cừ",  

    # --- Hướng Hải Vân / Liên Chiểu ---
    "Nguyễn Lương Bằng", "Tôn Đức Thắng",
    "Nguyễn Sinh Sắc", "Phạm Hùng",                          

    # --- Ngũ Hành Sơn, Non Nước ---
    "Lê Văn Hiến", "Võ Chí Công",
    "Huyền Trân Công Chúa", "Non Nước",
    "Núi Thành", "Mai Đăng Chơn", "Lê Thanh Nghị",           

    # --- Quốc lộ, tỉnh lộ (Bà Nà, phía tây) ---
    "Quốc lộ 1", "Quốc lộ 14B", "Đường Tỉnh 602",
    "Đường Cao tốc Đà Nẵng - Quảng Ngãi",
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
