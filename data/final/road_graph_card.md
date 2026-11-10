# Data card – đồ thị đường cho RL (road_nodes / road_edges / pois)

- Sinh ngày: 2026-11-10 · script: `scripts/build_road_graph.py` · kiểm tra: `scripts/check_road_graph.py`
- `road_nodes.csv`: 37,750 nút (ngã tư / điểm cuối đường; `node_id`, `lat`, `lon`, `ring_deep`) · `road_edges.csv`: 84,919 đoạn có hướng, 19,080 km (`edge_id`, `from_node`, `to_node`, `size_level`, `width_m`, `time_<slot>` × 5, `ring_deep`, `pois_passed`; bảng đầy đủ có `length_m`, `road_class`, `lanes`, `name`… ở `data/interim/road_edges_full.csv`) · `pois.csv`: 50 POI (`point_id`, `name`, `score`)
- Nguồn đường: OpenStreetMap (© OpenStreetMap contributors, ODbL), tải bằng OSMnx 21/10/2026, vùng [107.20, 14.95, 108.75, 16.25]; giữ thành phần liên thông mạnh lớn nhất.
- Đường khu dân cư `residential`: chỉ giữ phố **có tên thật** (không phải "Kiệt / Hẻm / Ngõ / K12…") hoặc **≥ 2 làn** → mức `local`. Loại bỏ: residential không tên, kiệt / hẻm, `living_street`.
- Xe chỉ **đi ngang** POI (không dừng đón trả): thưởng lấy qua `road_edges.pois_passed`. A, B (địa chỉ bất kỳ) gắn vào nút lúc chạy bằng `ABEstimator.snap()`.
- Bảng phụ không dùng khi train: `data/interim/points_snap.csv` (toạ độ, nút gắn `node_id` / `node_id_ban`, `ring_deep` của 60 điểm) – cho kiểm tra và bản đồ.

## Bề rộng mặt đường (`width_m`) – nguồn nhà nước

- Nguồn: **Cổng Thông tin điện tử thành phố Đà Nẵng – CSDL đường phố** (https://danang.gov.vn/ttdt/csdl/duong-pho),
  2.441 đoạn phố, cột "Chiều rộng mặt (m)". Lưu ở `data/references/danang_duong_pho.csv`. **Phải ghi nguồn khi dùng lại.**
- Ghép theo tên đường (chuẩn hoá) trong vùng Đà Nẵng cũ; phố nhiều đoạn bề rộng khác nhau định vị theo 2 ngã giao
  "điểm đầu – điểm cuối" (`scripts/street_width.py`). Khớp 2.038 / 2.197 tên phố.
- "2 x 10,5" = 21 m (hai chiều tách riêng); "7,5 & 5,5" lấy phần rộng nhất.
- **23.270 đoạn (2.256 km) có bề rộng thật**; còn lại (quốc lộ, đường tỉnh, Quảng Nam cũ, Hội An, phố chưa có trong danh
  sách) để trống và xếp mức theo OSM.
- **Bỏ hẳn 1.210 đoạn có mặt đường < 5 m** (xe 45 chỗ rộng 2,5 m không vào được).

## Mức đường (`size_level`)

Có `width_m` → xếp theo mét; không có → theo loại đường OSM.

| Mức | Theo bề rộng thật | Theo OSM `highway` (khi thiếu bề rộng) | km (thật + OSM) | Gợi ý reward |
|---|---|---|---|---|
| big | ≥ 14 m (15, 21, 30 m…) | motorway, trunk, primary (+ link) | 3,695 (601 + 3,095) | cộng nhiều |
| medium | 9 – 14 m (10,5 m) | secondary, tertiary (+ link) | 6,710 (412 + 6,298) | cộng ít |
| small | 7 – 9 m (7,5 m) | unclassified, road | 5,963 (502 + 5,461) | trừ |
| local | 5 – 7 m (5,5 m) | residential có tên hoặc ≥ 2 làn | 2,713 (741 + 1,972) | trừ nặng |

Đoạn không có bề rộng thật mà có `lanes` ≥ 4 được nâng một mức.

## Thời gian từng đoạn (`time_<slot>`, giây)

= chiều dài × giây/mét của nhóm đường trong khung giờ, hiệu chỉnh (bình phương tối thiểu không âm, sai số tương đối) cho khớp 14.850 thời gian TomTom (`configs/estimator_params.json`). Tốc độ trung bình (km/h):

| slot | trunk | primary | secondary | tertiary | unclassified | residential |
|---|---|---|---|---|---|---|
| morning | 24.7 | 27.3 | 27.8 | 16.5 | 18.9 | 10.3 |
| ban_noon | 23.2 | 28.6 | 30.2 | 19.0 | 19.7 | 14.2 |
| afternoon | 21.9 | 28.0 | 28.6 | 16.9 | 19.0 | 10.3 |
| ban_evening | 22.4 | 26.8 | 28.7 | 18.4 | 20.0 | 13.3 |
| night | 28.1 | 29.5 | 29.7 | 17.7 | 18.2 | 9.8 |

Không dùng giới hạn tốc độ (`maxspeed`): thử trên 2.970 chặng cho thời gian chỉ bằng 40 % thực tế (sai số 61 %).

## Luật vành đai

- `ring_deep = 1` (2824 đoạn): trung điểm nằm sâu trong vành đai (cách mép > 25 m, ngoài trục Duy Tân) → **che** trong `ban_noon`, `ban_evening` (CV 4883/UBND-SGTVT 2019).
- `corridor = 1` (144 đoạn): trục Duy Tân – Nguyễn Văn Trỗi – Trần Thị Lý, được đi trong giờ cấm (CV 1710/UBND-SGTVT 2022).
- POI nằm trên đoạn `ring_deep` tự động không ăn điểm trong giờ cấm (đoạn đã bị che). A/B sâu trong vành đai: `ABEstimator` báo không hợp lệ trong giờ cấm.

## Kiểm chứng: đường nhanh nhất trên đồ thị so với TomTom

| slot | nhóm | số chặng | sai số TB | đồ thị ÷ TomTom (trung vị) |
|---|---|---|---|---|
| afternoon | hub | 520 | 18.4 % | 1.03 |
| afternoon | poi | 2450 | 19.5 % | 0.97 |
| afternoon | point | 570 | 14.1 % | 1.02 |
| ban_evening | hub | 404 | 15.8 % | 0.99 |
| ban_evening | poi | 1420 | 16.7 % | 0.97 |
| ban_evening | point | 454 | 13.0 % | 1.00 |
| ban_noon | hub | 404 | 14.9 % | 1.01 |
| ban_noon | poi | 1425 | 16.0 % | 0.97 |
| ban_noon | point | 454 | 12.5 % | 1.01 |
| morning | hub | 520 | 18.1 % | 1.02 |
| morning | poi | 2450 | 20.2 % | 0.97 |
| morning | point | 570 | 14.9 % | 1.01 |
| night | hub | 520 | 16.3 % | 1.02 |
| night | poi | 2450 | 19.4 % | 0.98 |
| night | point | 570 | 14.2 % | 1.02 |

Số liệu so sánh dùng nút gắn trong `data/interim/points_snap.csv`. `point` = 5 điểm kiểm tra T01–T05, không dùng ở bất kỳ bước hiệu chỉnh nào. Khớp "đi được / không": 100 % giờ thường, ~98 % giờ cấm.

## Hạn chế

- Thời gian từng đoạn là **ước lượng suy ra từ TomTom**, không phải số đo trực tiếp; không có phí dừng đèn / rẽ riêng ở nút.
- `pois_passed`: bán kính 100 m quanh hình dạng thật của đoạn; 9 POI khuôn viên rộng / lùi sâu được nới tới đoạn gần nhất + 20 m (Tam Tòa 180 m, Bàn Cờ 153 m, Bà Nà 149 m, Apollo 145 m, Tiên Sơn 133 m, Thanh Hà 134 m, Phước Lâm 132 m; tối đa 250 m).
- Phân loại `residential` theo tên / số làn trên OSM: có thể còn sót vài phố hẹp có tên, hoặc thiếu đường rộng chưa đặt tên.
- Trong giờ cấm, 9 POI trong lõi vành đai không có đoạn hợp lệ đi ngang (đúng luật).
