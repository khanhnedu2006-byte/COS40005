from busroute.graph import restrictions as r


def test_whitelist_allows_main_road():
    assert r.is_statically_allowed({"name": "Đường Nguyễn Văn Linh", "osmid": 1})


def test_whitelist_blocks_unknown_alley():
    assert not r.is_statically_allowed({"name": "Kiệt 45 Lê Duẩn", "osmid": 2})


def test_merged_names_after_simplify():
    assert r.is_statically_allowed({"name": ["Kiệt 12", "Bạch Đằng"], "osmid": [3, 4]})


def test_time_ban(monkeypatch):
    square = [(16.0, 108.0), (16.0, 108.3), (16.1, 108.3), (16.1, 108.0)]
    monkeypatch.setattr(r, "TIME_BANNED_AREAS",
                        [{"polygon": square, "windows": [("10:30", "12:00")]}])
    assert r.is_time_banned(16.05, 108.2, 11 * 60)
    assert not r.is_time_banned(16.05, 108.2, 9 * 60)
