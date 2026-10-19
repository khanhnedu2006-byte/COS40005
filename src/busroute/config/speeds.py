"""Giới hạn tốc độ theo Thông tư 38/2024/TT-BGTVT và hệ số hiệu chỉnh từ GPS."""

# Tốc độ tối đa (km/h) – chặn trên cho mọi ước lượng
LEGAL_LIMIT_KMH = {
    "urban_divided": 60, "urban_undivided": 50,   # khu đông dân cư, mọi xe cơ giới
    "rural_divided": 80, "rural_undivided": 70,   # ngoài khu dân cư, xe chở người trên 28 chỗ
}

# Loại giới hạn của từng đường chính – mặc định là đường nội thành không dải phân cách
DEFAULT_LIMIT_CLASS = "urban_undivided"
ROAD_LIMIT_CLASS = {
    "Võ Nguyên Giáp": "urban_divided",
    "Nguyễn Văn Linh": "urban_divided",
    "Quốc lộ 1A": "rural_divided",
    # TODO: bổ sung sau khi kiểm tra thực địa
}

# Hệ số = tốc độ thực / giới hạn luật. GIẢ ĐỊNH tạm thời cho tới khi có đủ GPS
# Khoá: (limit_class, slot). Thiếu khoá thì dùng DEFAULT_CALIBRATION
DEFAULT_CALIBRATION = 0.5
CALIBRATION = {}

MIN_GPS_SAMPLES = 5     # số mẫu tối thiểu để tin tốc độ GPS của một đoạn
