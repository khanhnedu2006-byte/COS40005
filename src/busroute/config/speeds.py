"""Giới hạn tốc độ theo Thông tư 38/2024/TT-BGTVT và hệ số hiệu chỉnh từ GPS."""

# Tốc độ tối đa (km/h) – chặn trên cho mọi ước lượng
LEGAL_LIMIT_KMH = {
    "urban_divided": 60, "urban_undivided": 50,   # khu đông dân cư, mọi xe cơ giới
    "rural_divided": 80, "rural_undivided": 70,   # ngoài khu dân cư, xe chở người trên 28 chỗ
    "highway": 120,                               # đường cao tốc
}

# Loại giới hạn của từng đường chính – mặc định là đường nội thành không dải phân cách
DEFAULT_LIMIT_CLASS = "urban_undivided"
ROAD_LIMIT_CLASS = {
    "urban_divided": [   # nội thành, có dải phân cách giữa
        "Võ Nguyên Giáp", "Nguyễn Văn Linh", "Phạm Văn Đồng", "Võ Văn Kiệt",
        "Điện Biên Phủ", "Nguyễn Tất Thành", "2 Tháng 9", "Lê Duẩn",
        "Nguyễn Hữu Thọ", "Trường Chinh", "Cách Mạng Tháng 8", "Võ Chí Công",
        "Nguyễn Lương Bằng", "Tôn Đức Thắng", "Phạm Hùng", "Hoàng Văn Thái",
        "Lê Văn Hiến", "Trường Sa", "Bạch Đằng", "Trần Hưng Đạo",
        "Nguyễn Chí Thanh", "Nguyễn Tri Phương", "Núi Thành", "3 Tháng 2",
    ],
    "urban_undivided": [  # nội thành, không có dải phân cách
        "Hoàng Sa", "Hồ Nghinh", "Nguyễn Văn Thoại", "Ngô Quyền", "Như Nguyệt",
        "Dương Đình Nghệ", "Lê Văn Duyệt", "Nguyễn Công Trứ",
        "Hùng Vương", "Lê Lợi", "Quang Trung", "Trần Phú", "Phan Châu Trinh",
        "Hoàng Diệu", "Duy Tân", "Yên Bái", "Hải Phòng",
        "Trần Thị Lý", "Ông Ích Khiêm", "Trưng Nữ Vương", "Hàm Nghi",
        "Lê Đình Lý", "Nguyễn Hữu Cảnh", "Nguyễn Văn Cừ", "Nguyễn Sinh Sắc",
        "Huyền Trân Công Chúa", "Non Nước", "Mai Đăng Chơn", "Lê Thanh Nghị",
    ],
    "rural_divided": [    # ngoài khu dân cư, có dải phân cách
        "Quốc lộ 1",
    ],
    "rural_undivided": [  # ngoài khu dân cư, không có dải phân cách
        "Quốc lộ 14B", "Đường Tỉnh 602",
    ],
    "highway": [          # đường cao tốc
        "Đường Cao tốc Đà Nẵng - Quảng Ngãi",
    ],
}

# Hệ số = tốc độ thực / giới hạn luật. GIẢ ĐỊNH tạm thời cho tới khi có đủ GPS
# Khoá: (limit_class, slot). Thiếu khoá thì dùng DEFAULT_CALIBRATION
DEFAULT_CALIBRATION = 0.5
CALIBRATION = {}

MIN_GPS_SAMPLES = 5     # số mẫu tối thiểu để tin tốc độ GPS của một đoạn
