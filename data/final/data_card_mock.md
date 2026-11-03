# Data card – od_matrix.csv

- Sinh ngày: 2026-11-03 · nguồn: `od_matrix_mock.csv`
- Điểm: 55 (50 POI, 5 hub)
- Dòng: 14,850 · feasible = 1: 11,874 (80.0 %)
- Thời gian: TomTom Routing API, travelMode=bus, dài 12 m / rộng 2,5 m / cao 3,6 m, departAt = thứ Tư, giờ đại diện mỗi khung (08:00 · 11:15 · 14:30 · 17:45 · 20:30), kẹt xe lịch sử.
- Giờ cấm: tránh vành đai trung tâm (CV 4883/UBND-SGTVT 2019), cho phép trục Duy Tân – Nguyễn Văn Trỗi – Trần Thị Lý (CV 1710/UBND-SGTVT 2022).
- Kiểm chứng: xem `reports/gps_validation.csv` (so với chặng GPS thật cùng thứ + giờ).

## Thời gian theo khung giờ (phút, các cặp feasible)

| slot | n | trung vị | p90 | km/h trung vị |
|---|---|---|---|---|
| morning | 2970 | 21.2 | 76.2 | 25.0 |
| ban_noon | 1482 | 36.9 | 90.7 | 22.0 |
| afternoon | 2970 | 22.1 | 79.3 | 24.0 |
| ban_evening | 1482 | 40.5 | 99.8 | 20.0 |
| night | 2970 | 17.6 | 63.5 | 30.0 |

## Không đi được (32 cặp điểm × khung giờ)

P004@ban_noon, P004@ban_evening, P005@ban_noon, P005@ban_evening, P009@ban_noon, P009@ban_evening, P014@ban_noon, P014@ban_evening, P016@ban_noon, P016@ban_evening, P018@ban_noon, P018@ban_evening, P019@ban_noon, P019@ban_evening, P021@ban_noon, P021@ban_evening, P023@ban_noon, P023@ban_evening, P024@ban_noon, P024@ban_evening, P025@ban_noon, P025@ban_evening, P026@ban_noon, P026@ban_evening, P043@ban_noon, P043@ban_evening, P044@ban_noon, P044@ban_evening, P045@ban_noon, P045@ban_evening, P046@ban_noon, P046@ban_evening

## Kết quả kiểm tra

- [x] Đủ cột
- [x] Đủ mọi cặp × 5 khung giờ
- [x] Không trùng (from, to, slot)
- [x] Không có cặp tự nối
- [x] Khung giờ hợp lệ
- [x] feasible=1 thì duration > 0 và distance > 0
- [x] Tốc độ trung bình 5–90 km/h
- [x] Giờ cấm không đi sâu vào vành đai
- [x] ban_noon ≥ 0,8 × morning
- [x] ban_evening ≥ 0,8 × afternoon
- [x] A→B / B→A trong 0,5–2 (cảnh báo)
- [x] Mỗi điểm đi được ra ngoài ở mọi khung giờ (cảnh báo nếu POI trong vành đai)
