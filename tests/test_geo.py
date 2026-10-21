from busroute.utils.geo import haversine_m, point_in_polygon, normalize_road_name
from busroute.utils.timeslots import in_windows, slot_of, to_minutes


def test_haversine_one_degree_lat():
    assert abs(haversine_m(16.0, 108.0, 17.0, 108.0) - 111_195) < 200


def test_point_in_square():
    square = [(0, 0), (0, 1), (1, 1), (1, 0)]
    assert point_in_polygon(0.5, 0.5, square)
    assert not point_in_polygon(1.5, 0.5, square)
    assert not point_in_polygon(0.5, 0.5, [])


def test_normalize_road_name():
    assert normalize_road_name("Đường  Nguyễn Văn Linh ") == normalize_road_name("nguyễn văn linh")


def test_time_windows():
    windows = [("10:30", "12:00"), ("16:30", "19:00")]
    assert in_windows(to_minutes("11:00"), windows)
    assert not in_windows(to_minutes("12:00"), windows)
    assert slot_of(to_minutes("17:00")) == "ban_evening"


def test_normalize_keeps_road_type_prefix():
    # "Đường tỉnh", "Đường cao tốc" là một phần của tên, không được bỏ
    assert normalize_road_name("Đường tỉnh 609") == "đường tỉnh 609"
    assert normalize_road_name("Đường cao tốc Đà Nẵng - Quảng Ngãi") == "đường cao tốc đà nẵng - quảng ngãi"
