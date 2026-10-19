"""Đặc trưng khuôn mặt từ landmark (ví dụ MediaPipe Face Mesh)."""
import numpy as np


def eye_aspect_ratio(eye):
    # eye: 6 điểm (x, y) theo thứ tự p1..p6. EAR thấp kéo dài → nhắm mắt / buồn ngủ
    eye = np.asarray(eye, dtype=float)
    vertical = np.linalg.norm(eye[1] - eye[5]) + np.linalg.norm(eye[2] - eye[4])
    horizontal = np.linalg.norm(eye[0] - eye[3])
    return vertical / (2.0 * horizontal)


def mouth_aspect_ratio(mouth):
    # mouth: 8 điểm (x, y). MAR cao kéo dài → ngáp
    mouth = np.asarray(mouth, dtype=float)
    vertical = (np.linalg.norm(mouth[1] - mouth[7]) + np.linalg.norm(mouth[2] - mouth[6])
                + np.linalg.norm(mouth[3] - mouth[5]))
    horizontal = np.linalg.norm(mouth[0] - mouth[4])
    return vertical / (3.0 * horizontal)


def perclos(ear_series, threshold=0.2):
    # Tỉ lệ thời gian mắt nhắm trong một cửa sổ – chỉ số buồn ngủ phổ biến
    ear_series = np.asarray(ear_series)
    return float((ear_series < threshold).mean()) if len(ear_series) else 0.0
