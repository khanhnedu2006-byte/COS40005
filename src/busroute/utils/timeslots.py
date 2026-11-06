"""Khung giờ trong ngày. Ranh giới khớp với giờ cấm vành đai trung tâm."""

from collections.abc import Iterable

# slot_name: (bắt đầu, kết thúc)
TIME_SLOTS = {
    "morning": ("05:00", "10:30"),
    "ban_noon": ("10:30", "12:00"),
    "afternoon": ("12:00", "16:30"),
    "ban_evening": ("16:30", "19:00"),
    "night": ("19:00", "23:59"),
}


def to_minutes(hhmm: str) -> int:
    """Convert an HH:MM string to minutes after midnight."""
    h, m = map(int, hhmm.split(":"))
    return h * 60 + m


def in_windows(minute_of_day: int, windows: Iterable[tuple[str, str]]) -> bool:
    """Check membership in any half-open [start, end) window in HH:MM."""
    return any(to_minutes(s) <= minute_of_day < to_minutes(e) for s, e in windows)


def slot_of(minute_of_day: int) -> str:
    """Return the containing time slot, or night when no slot contains it."""
    for name, (s, e) in TIME_SLOTS.items():
        if to_minutes(s) <= minute_of_day < to_minutes(e):
            return name
    return "night"
