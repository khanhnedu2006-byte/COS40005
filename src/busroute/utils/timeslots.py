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


_SLOT_WINDOWS: tuple[tuple[str, int, int], ...] = tuple(
    (name, to_minutes(start), to_minutes(end))
    for name, (start, end) in TIME_SLOTS.items()
)


def _compute_slot(minute_of_day: int) -> str:
    for name, start_m, end_m in _SLOT_WINDOWS:
        if start_m <= minute_of_day < end_m:
            return name
    return "night"


_SLOT_BY_MINUTE: tuple[str, ...] = tuple(_compute_slot(m) for m in range(1440))


def slot_of(minute_of_day: int) -> str:
    """Return the containing time slot, or night when no slot contains it."""
    if 0 <= minute_of_day < 1440:
        return _SLOT_BY_MINUTE[minute_of_day]
    return "night"
