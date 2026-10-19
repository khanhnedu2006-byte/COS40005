"""Danh sách POI viết cứng. Toạ độ dưới đây là GẦN ĐÚNG – nhóm phải kiểm tra lại trên bản đồ."""

# score (1–10) = 0.5 × tần suất trong tour + 0.3 × số review (chuẩn hoá) + 0.2 × đánh giá tài xế
# pass_road: đường chính mà xe chạy ngang để tính là "đã ghé" POI
POIS = [
    {"poi_id": "P01", "name": "Dragon Bridge", "lat": 16.0611, "lon": 108.2272,
     "category": "bridge", "score": None, "pass_road": "Nguyễn Văn Linh"},
    {"poi_id": "P02", "name": "Han River Bridge", "lat": 16.0722, "lon": 108.2264,
     "category": "bridge", "score": None, "pass_road": "Lê Duẩn"},
    {"poi_id": "P03", "name": "My Khe Beach", "lat": 16.0544, "lon": 108.2477,
     "category": "beach", "score": None, "pass_road": "Võ Nguyên Giáp"},
    {"poi_id": "P04", "name": "Linh Ung Pagoda (Son Tra)", "lat": 16.1003, "lon": 108.2779,
     "category": "landmark", "score": None, "pass_road": "Hoàng Sa"},
    {"poi_id": "P05", "name": "Marble Mountains", "lat": 16.0034, "lon": 108.2630,
     "category": "landmark", "score": None, "pass_road": "Lê Văn Hiến"},
    {"poi_id": "P06", "name": "Hoi An Ancient Town", "lat": 15.8801, "lon": 108.3380,
     "category": "landmark", "score": None, "pass_road": "TODO"},
    # TODO: bổ sung từ poi_template_danang.csv
]
