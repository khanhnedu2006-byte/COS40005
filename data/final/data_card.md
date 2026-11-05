# Data card – od_matrix.csv

- Sinh ngày: 2026-11-05 · nguồn: `od_matrix.csv`
- Điểm: 55 (50 POI, 5 hub)
- Dòng: 14,850 · feasible = 1: 12,557 (84.6 %)
- Thời gian: TomTom Routing API, travelMode=bus, dài 12 m / rộng 2,5 m / cao 3,6 m, departAt = thứ Tư, giờ đại diện mỗi khung (08:00 · 11:15 · 14:30 · 17:45 · 20:30), kẹt xe lịch sử.
- Giờ cấm: tránh vành đai trung tâm (CV 4883/UBND-SGTVT 2019), cho phép trục Duy Tân – Nguyễn Văn Trỗi – Trần Thị Lý (CV 1710/UBND-SGTVT 2022).
- Kiểm chứng: xem `reports/gps_validation.csv` (so với chặng GPS thật cùng thứ + giờ).

## Thời gian theo khung giờ (phút, các cặp feasible)

| slot | n | trung vị | p90 | km/h trung vị |
|---|---|---|---|---|
| morning | 2970 | 23.2 | 57.7 | 25.8 |
| ban_noon | 1826 | 29.4 | 59.0 | 28.8 |
| afternoon | 2970 | 22.5 | 56.9 | 26.4 |
| ban_evening | 1821 | 31.2 | 61.8 | 27.1 |
| night | 2970 | 21.1 | 53.2 | 27.3 |

## Không đi được (22 cặp điểm × khung giờ)

P009@ban_noon, P009@ban_evening, P014@ban_noon, P014@ban_evening, P016@ban_noon, P016@ban_evening, P018@ban_noon, P018@ban_evening, P019@ban_noon, P019@ban_evening, P021@ban_noon, P021@ban_evening, P023@ban_noon, P023@ban_evening, P025@ban_noon, P025@ban_evening, P044@ban_noon, P044@ban_evening, P045@ban_noon, P045@ban_evening, P046@ban_noon, P046@ban_evening

## Kết quả kiểm tra

- [x] Đủ cột
- [x] Đủ mọi cặp × 5 khung giờ
- [x] Không trùng (from, to, slot)
- [x] Không có cặp tự nối
- [x] Khung giờ hợp lệ
- [x] feasible=1 thì duration > 0 và distance > 0
- [x] Tốc độ trung bình 5–90 km/h
- [x] Giờ cấm không đi sâu vào vành đai
- [x] ban_noon ≥ 0,8 × morning (cảnh báo)
- [x] ban_evening ≥ 0,8 × afternoon (cảnh báo)
- [x] A→B / B→A trong 0,5–2 (cảnh báo)
- [x] Mỗi điểm đi được ra ngoài ở mọi khung giờ (cảnh báo nếu POI trong vành đai)
