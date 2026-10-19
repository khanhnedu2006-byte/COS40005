"""Khung giờ trong ngày. Ranh giới khớp với giờ cấm vành đai trung tâm."""

# slot_name: (bắt đầu, kết thúc)
TIME_SLOTS = {
    "morning": ("05:00", "10:30"),
    "ban_noon": ("10:30", "12:00"),
    "afternoon": ("12:00", "16:30"),
    "ban_evening": ("16:30", "19:00"),
    "night": ("19:00", "23:59"),
}


def to_minutes(hhmm):
    # "10:30" → 630
    h, m = map(int, hhmm.split(":"))
    return h * 60 + m


def in_windows(minute_of_day, windows):
    # True nếu thời điểm nằm trong một trong các khung [start, end)
    return any(to_minutes(s) <= minute_of_day < to_minutes(e) for s, e in windows)


def slot_of(minute_of_day):
    # Trả về tên khung giờ chứa thời điểm; ngoài mọi khung → "night"
    for name, (s, e) in TIME_SLOTS.items():
        if to_minutes(s) <= minute_of_day < to_minutes(e):
            return name
    return "night"
