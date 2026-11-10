# DATA TASKS v2 – Ma trận thời gian giữa 55 điểm (cập nhật 05/11/2026 – ĐÃ KHOÁ)

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
   **Trục được phép:** Duy Tân – Nguyễn Văn Trỗi – Trần Thị Lý (Công văn 1710/UBND-SGTVT, 04/2022; GPS 14/10/2026 xác nhận xe 45 chỗ vẫn đi Duy Tân lúc 17:22, 18:19) → không tính là vào vành đai, vùng tránh chừa trục này.
4. Chọn tuyến nhanh nhất còn lại → `duration_s`, `distance_m`, `pois_passed`. Không còn tuyến → `feasible = 0`.
5. Kiểm chứng: so TomTom với 20 chặng GPS thật cùng thứ + giờ (`validate`) → `reports/gps_validation.csv`.
   Lần 1 (03/11): MAPE 33,5 % tổng; trên 10 chặng cùng tuyến MAPE 24 %, GPS ÷ TomTom = 0,82 (TomTom chậm hơn ~18 %).
   `build --scale 0.82` nhân hệ số này (chờ nhóm duyệt).

Hạn mức TomTom miễn phí ~2.500 request/ngày → 14.850 request ≈ **6 ngày chạy** (script tự chạy tiếp từ chỗ dừng).

## 3. Trạng thái

| Việc | Người | Trạng thái |
|---|---|---|
| Bản đồ nền OSM (`data/stage1/`) | Khánh | ✅ 21/10 |
| Đa giác vành đai cấm (`data/references/central_ring.geojson`) | Tuấn Anh | ✅ 28/10 – đoạn Bạch Đằng vẽ trùng mặt đường, nên lùi vào 20–30 m |
| 50 POI + điểm Bayes (`configs/pois.py`) | Tuấn Anh, Quân | ✅ 03/11 – đã sửa số liệu POI 21, 38 |
| Rà toạ độ POI (`reports/poi_audit.csv`) | Khánh | ✅ 03/11 – đã đổi toạ độ P001, P006, P007, P012, P020, P035, P036, P038 |
| 5 hub (`configs/hubs.py`) | Khánh | ✅ chốt 03/11 |
| Script TomTom `fetch_routes.py` (diag/test/compare/validate/fetch/build) | Khánh | ✅ đã thử 1 cặp, chạy đúng |
| Chặng GPS thật (`build_gps_trips.py`, 20 chặng) | Khánh | ✅ |
| Chạy `validate` (20 request) | Khánh | ✅ 03/11 – `reports/gps_validation.csv` |
| Chạy `fetch` (14.850 request TomTom) | Khánh | ✅ 05/11 |
| `build` + kiểm tra + `data_card.md` | Khánh | ✅ 05/11 – 12/12 đạt, **KHOÁ DỮ LIỆU** |
| `env.py` cho đồ thị đầy đủ 55 nút, đọc `od_matrix.csv` | Đức | ⏳ có thể làm song song với ma trận giả |

## 4. Quyết định trước khi chạy `fetch` – ĐÃ CHỐT 03/11 (giữ 5 hub, đổi toạ độ 8 POI theo audit, giữ Asia Park + Đỉnh Bàn Cờ)

1. **Duyệt 5 hub** trong `configs/hubs.py` (sân bay T2, cổng cảng Tiên Sa, bến xe trung tâm, Mỹ Khê, bến xe Hội An).
2. **Duyệt toạ độ POI** theo `reports/poi_audit.csv` – đổi toạ độ sau khi chạy phải lấy lại toàn bộ cặp của POI đó:
   - `on_bridge`: Cầu Rồng (P001), Cầu Thuận Phước (P020) → đặt ở đầu cầu phía xe chạy ngang (Cầu Rồng xuất phát trên cầu
     bị vòng quay đầu, giờ cấm mất 43 phút thay vì 27).
   - `far_from_road` (7 điểm, có cột `suggest_lat/lon`): biển Mỹ Khê, Ngũ Hành Sơn, An Bàng, Cửa Đại, Bãi Cỏ Cháy,
     Asia Park, Cầu Thuận Phước.
   - `inside_ring` (15 điểm): đúng luật, giờ cấm sẽ có `feasible = 0`.
3. **Asia Park** (Google ghi đóng vĩnh viễn) và **Đỉnh Bàn Cờ** (cách đường chính 2,3 km, cần kiểm tra quy định xe lớn lên
   Sơn Trà): giữ hay bỏ?

## 5. Kết quả khoá dữ liệu (05/11/2026)

- `data/final/od_matrix.csv`: 14.850 dòng, feasible = 12.557 (84,6 %), `scale = 1.0` (thời gian TomTom gốc).
- Giờ cấm: 11 điểm nằm sâu trong vành đai (> 100 m từ đường bao, ngoài trục Duy Tân) không đi/tới được:
  P009, P014, P016, P018, P019, P021, P023, P025, P044, P045, P046.
- Cảnh báo đã biết: 11 cặp ngắn (0,2–0,4 %) giờ cấm nhanh hơn giờ thường do TomTom chỉ đưa lối tắt khi tránh vành đai.
- TomTom chậm hơn GPS thật ~18 % trên tuyến trùng → báo cáo nêu rõ; có thể build lại với `--scale 0.82` (không tốn request).

## 6. Đã bỏ so với kế hoạch cũ

- `MAIN_ROADS` (whitelist 57 đường), `speeds.py` (giới hạn tốc độ), map matching GPS theo từng đoạn.
- Dữ liệu độ rộng đường: thay bằng kích thước xe khai cho TomTom; nếu tìm được số liệu nhà nước thì dùng để kiểm tra.
- Không dùng Google Routes API (điều khoản cấm tải hàng loạt và dùng để train AI); Goong giữ làm phương án dự phòng
  (`--provider goong`, không có giờ xuất phát).


## 7. Cập nhật 09/11/2026 – A, B là địa chỉ bất kỳ

Bài toán gốc: người dùng gõ địa chỉ A, B bất kỳ. `od_matrix.csv` giữ vai trò phần POI → POI; 5 hub chỉ là ví dụ A/B.

| Việc | Trạng thái |
|---|---|
| Chặng đi ra từ Cầu Rồng: khai `vehicleHeading` = 90°, lấy lại 270 chặng (`fetch --refetch-from P001`) | ⏳ chờ chạy |
| `src/busroute/graph/ab_estimator.py`: thời gian A/B bất kỳ trên đồ thị OSM, hiệu chỉnh theo TomTom (`scripts/calibrate_estimator.py`) | ✅ sai số 15–18 % trên hub, khớp feasible 97,5–99,9 % |
| 5 điểm kiểm tra T01–T05 (`configs/test_points.py`) → `data/final/test_legs.csv` (`fetch --with-test-points`, ~2.850 lượt) | ⏳ chờ chạy |
| Đặc tả train / kiểm tra / demo cho A/B bất kỳ: `docs/AB_ROUTING.md` | ✅ |


## 8. Cập nhật 10/11/2026 – dataset ĐỒ THỊ ĐƯỜNG để RL tự chọn đường từng ngã rẽ

Nhóm chọn cho RL tự học đường đi (không đưa sẵn đường nhanh nhất). Dataset mới (`data/final/road_graph_card.md`):

| File | Nội dung |
|---|---|
| `pois.csv` | 60 điểm (50 POI + 5 hub + 5 điểm kiểm tra): điểm số, nút gắn (`node_id`, giờ cấm `node_id_ban`) |
| `road_nodes.csv` | 27.297 ngã tư (OSM), cờ sâu trong vành đai, POI lân cận |
| `road_edges.csv` | 54.655 đoạn có hướng: chiều dài, loại đường, `size_level` big/medium/small, số làn, `time_<slot>` × 5, `ring_deep`, `corridor`, `pois_passed` |

- Bỏ đường dân sinh / kiệt hẻm. Thời gian đoạn = chiều dài × tốc độ hiệu chỉnh từ TomTom (không dùng maxspeed: sai 61 %).
- Kiểm chứng: đường nhanh nhất trên đồ thị lệch TomTom 13–15 % (điểm kiểm tra), 18–21 % (POI); khớp đi được 98–100 %.
- `od_matrix.csv` / `test_legs.csv` giữ làm chuẩn so sánh.
