"""Điểm đón/trả khách (hub) – A và B của mỗi chuyến chọn trong danh sách này.

TRẠNG THÁI: ĐỀ XUẤT, nhóm cần duyệt trước khi chạy fetch_routes.py fetch thật.
Toạ độ lấy từ OpenStreetMap (Nominatim) và đồ thị chặng 1, tra ngày 2026-11-03.
Tiêu chí chọn: nơi xe 45 chỗ thực sự đón khách đoàn, nằm NGOÀI vành đai cấm
(để giờ cấm vẫn xuất phát được), trải đều Đà Nẵng – Sơn Trà – Hội An.
"""

HUBS = [
    {"hub_id": "H01", "name": "Sân bay quốc tế Đà Nẵng (nhà ga T2)",
     "coordinates": [16.05634, 108.20235], "source": "OSM aeroway/terminal"},
    # Nhà ga tàu khách nằm trong khu cảng (đường nội bộ không có trên đồ thị) → đặt ở cổng cảng
    {"hub_id": "H02", "name": "Cảng Tiên Sa – cổng cảng (khách tàu biển)",
     "coordinates": [16.12196, 108.21945], "source": "OSM ferry_terminal 16.11874,108.21321 → điểm gần nhất trên đường chính"},
    {"hub_id": "H03", "name": "Bến xe Trung tâm Đà Nẵng",
     "coordinates": [16.05501, 108.17336], "source": "OSM amenity/bus_station"},
    {"hub_id": "H04", "name": "Khu khách sạn biển Mỹ Khê (Võ Nguyên Giáp × Phạm Văn Đồng)",
     "coordinates": [16.07040, 108.24507], "source": "nút giao trên đồ thị chặng 1"},
    {"hub_id": "H05", "name": "Bến xe Hội An (cửa ngõ phố cổ)",
     "coordinates": [15.88669, 108.32110], "source": "OSM amenity/bus_station"},
]
