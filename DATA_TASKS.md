# DATA TASKS v2 – Ma trận thời gian giữa 55 điểm (cập nhật 03/11/2026)

> Thay thế kế hoạch 3 chặng cũ (21/10 → 31/10). Bản cũ xem trong lịch sử git.
> Thiết kế mới do nhóm chốt 01–03/11: **node = POI + hub**, **thời gian lấy từ TomTom API theo khung giờ**,
> bỏ whitelist đường chính và bảng giới hạn tốc độ.

## 1. Đầu ra cuối: `data/final/od_matrix.csv` (một file duy nhất để train)

Mỗi dòng = đi từ điểm `from_id` tới `to_id` trong một khung giờ. 55 điểm × 54 × 5 khung giờ = **14.850 dòng**.

| Cột | Ý nghĩa |
|---|---|
| `from_id`, `to_id` | `P001`–`P050` (POI), `H01`–`H05` (hub) |
| `from_type`, `to_type` | `poi` / `hub` |
| `from_lat`, `from_lon`, `to_lat`, `to_lon` | toạ độ (đặc trưng cho GNN) |
| `to_poi_score` | điểm Bayes của POI đích (hub = 0) |
| `slot` | `morning` 05–10:30 · `ban_noon` 10:30–12 · `afternoon` 12–16:30 · `ban_evening` 16:30–19 · `night` 19–24 |
| `duration_s` | thời gian xe 45 chỗ (TomTom `travelMode=bus`, kẹt xe lịch sử theo giờ) |
| `distance_m` | quãng đường của tuyến được chọn |
| `feasible` | 1 = có tuyến hợp lệ; 0 = không tới được trong khung giờ này (vd. POI trong vành đai lúc cấm) |
| `pois_passed` | POI chạy ngang qua dọc đường (≤ 100 m), cách nhau bằng `;` |
| `through_ring` | tuyến có đi sâu vào vành đai không (chỉ có thể = 1 ngoài giờ cấm) |

## 2. Cách tính `duration_s` (scripts/fetch_routes.py)

1. Mỗi cặp điểm × mỗi khung giờ gọi TomTom Routing 1 lần: `travelMode=bus`, dài 12 m / rộng 2,5 m / cao 3,6 m,
   `departAt` = thứ Tư tới, giờ đại diện (08:00 · 11:15 · 14:30 · 17:45 · 20:30), lấy tối đa 4 tuyến.
2. **Không** khai tải trọng: TomTom sẽ áp giờ cấm xe tải nặng (thử nghiệm: bắt đi vòng 31 km thay vì 12 km).
3. Giờ cấm (`ban_noon`, `ban_evening`): gửi `avoidAreas` = 10 hình chữ nhật phủ 82 % vành đai, sau đó kiểm tra lại
   từng tuyến bằng đa giác; tuyến đi sâu > 200 m vào vành đai (bỏ 300 m đầu/cuối và dải 25 m sát mép = đường bao) bị loại.
4. Chọn tuyến nhanh nhất còn lại → `duration_s`, `distance_m`, `pois_passed`. Không còn tuyến → `feasible = 0`.
5. Kiểm chứng: so TomTom với 20 chặng GPS thật cùng thứ + giờ (`validate`) → `reports/gps_validation.csv`.

Hạn mức TomTom miễn phí ~2.500 request/ngày → 14.850 request ≈ **6 ngày chạy** (script tự chạy tiếp từ chỗ dừng).

## 3. Trạng thái

| Việc | Người | Trạng thái |
|---|---|---|
| Bản đồ nền OSM (`data/stage1/`) | Khánh | ✅ 21/10 |
| Đa giác vành đai cấm (`data/references/central_ring.geojson`) | Tuấn Anh | ✅ 28/10 – đoạn Bạch Đằng vẽ trùng mặt đường, nên lùi vào 20–30 m |
| 50 POI + điểm Bayes (`configs/pois.py`) | Tuấn Anh, Quân | ✅ 03/11 – đã sửa số liệu POI 21, 38 |
| Rà toạ độ POI (`reports/poi_audit.csv`) | Khánh | ✅ 03/11 – 29/55 điểm có cảnh báo, chờ duyệt |
| 5 hub đề xuất (`configs/hubs.py`) | Khánh | ⏳ chờ nhóm duyệt |
| Script TomTom `fetch_routes.py` (diag/test/compare/validate/fetch/build) | Khánh | ✅ đã thử 1 cặp, chạy đúng |
| Chặng GPS thật (`build_gps_trips.py`, 20 chặng) | Khánh | ✅ |
| Chạy `validate` (20 request) | Khánh | ⏳ |
| Chạy `fetch` 6 ngày | Khánh | ⏳ sau khi duyệt hub + toạ độ |
| `build` + kiểm tra + `data_card.md` | Khánh | ⏳ |
| `env.py` cho đồ thị đầy đủ 55 nút, đọc `od_matrix.csv` | Đức | ⏳ có thể làm song song với ma trận giả |

## 4. Cần nhóm quyết định trước khi chạy `fetch`

1. **Duyệt 5 hub** trong `configs/hubs.py` (sân bay T2, cổng cảng Tiên Sa, bến xe trung tâm, Mỹ Khê, bến xe Hội An).
2. **Duyệt toạ độ POI** theo `reports/poi_audit.csv` – đổi toạ độ sau khi chạy phải lấy lại toàn bộ cặp của POI đó:
   - `on_bridge`: Cầu Rồng (P001), Cầu Thuận Phước (P020) → đặt ở đầu cầu phía xe chạy ngang (Cầu Rồng xuất phát trên cầu
     bị vòng quay đầu, giờ cấm mất 43 phút thay vì 27).
   - `far_from_road` (7 điểm, có cột `suggest_lat/lon`): biển Mỹ Khê, Ngũ Hành Sơn, An Bàng, Cửa Đại, Bãi Cỏ Cháy,
     Asia Park, Cầu Thuận Phước.
   - `inside_ring` (15 điểm): đúng luật, giờ cấm sẽ có `feasible = 0`.
3. **Asia Park** (Google ghi đóng vĩnh viễn) và **Đỉnh Bàn Cờ** (cách đường chính 2,3 km, cần kiểm tra quy định xe lớn lên
   Sơn Trà): giữ hay bỏ?

## 5. Đã bỏ so với kế hoạch cũ

- `MAIN_ROADS` (whitelist 57 đường), `speeds.py` (giới hạn tốc độ), map matching GPS theo từng đoạn.
- Dữ liệu độ rộng đường: thay bằng kích thước xe khai cho TomTom; nếu tìm được số liệu nhà nước thì dùng để kiểm tra.
- Không dùng Google Routes API (điều khoản cấm tải hàng loạt và dùng để train AI); Goong giữ làm phương án dự phòng
  (`--provider goong`, không có giờ xuất phát).
