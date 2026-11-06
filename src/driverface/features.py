"""Đặc trưng khuôn mặt từ landmark (ví dụ MediaPipe Face Mesh)."""

import numpy as np


def eye_aspect_ratio(eye):
    """Compute eye aspect ratio from six ordered two-dimensional landmarks.

    Args:
        eye: Six (x, y) landmarks in p1 through p6 order.

    Returns:
        The sum of two vertical distances divided by twice the horizontal
        distance. A zero horizontal distance leads to a non-finite result.
    """
    eye = np.asarray(eye, dtype=float)
    vertical = np.linalg.norm(eye[1] - eye[5]) + np.linalg.norm(eye[2] - eye[4])
    horizontal = np.linalg.norm(eye[0] - eye[3])
    return vertical / (2.0 * horizontal)


def mouth_aspect_ratio(mouth):
    """Compute mouth aspect ratio from eight ordered landmarks.

    Args:
        mouth: Eight (x, y) landmarks in p1 through p8 order.

    Returns:
        The sum of three vertical distances divided by three times the
        horizontal distance. A zero horizontal distance is non-finite.
    """
    mouth = np.asarray(mouth, dtype=float)
    vertical = (
        np.linalg.norm(mouth[1] - mouth[7])
        + np.linalg.norm(mouth[2] - mouth[6])
        + np.linalg.norm(mouth[3] - mouth[5])
    )
    horizontal = np.linalg.norm(mouth[0] - mouth[4])
    return vertical / (3.0 * horizontal)


def perclos(ear_series, threshold: float = 0.2) -> float:
    """Calculate the fraction of eye-aspect-ratio samples below a threshold.

    Args:
        ear_series: Eye aspect ratio samples for the observation window.
        threshold: Values strictly below this threshold count as closed.

    Returns:
        The fraction of closed samples, or 0.0 for an empty series.
    """
    ear_series = np.asarray(ear_series)
    return float((ear_series < threshold).mean()) if len(ear_series) else 0.0
