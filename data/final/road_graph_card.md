# Data card – đồ thị đường cho RL (road_nodes / road_edges / pois)

- Sinh ngày: 2026-11-10 · script: `scripts/build_road_graph.py` · kiểm tra: `scripts/check_road_graph.py`
- `road_nodes.csv`: 27,297 nút (ngã tư / điểm cuối đường) · `road_edges.csv`: 54,655 đoạn có hướng, 15,564 km · `pois.csv`: 60 điểm (50 POI, 5 hub, 5 điểm kiểm tra)
- Nguồn đường: OpenStreetMap (© OpenStreetMap contributors, ODbL), tải bằng OSMnx 21/10/2026, vùng [107.20, 14.95, 108.75, 16.25]; giữ thành phần liên thông mạnh lớn nhất.
- Loại bỏ: `residential`, `living_street` (kiệt, hẻm – xe 45 chỗ không vào). POI/A/B gắn vào nút gần nhất trên đường còn lại.

## Mức đường (`size_level`)

| Mức | Loại OSM `highway` | km | Gợi ý reward |
|---|---|---|---|
| big | motorway, trunk, primary (+ link) | 3,482 | không phạt |
| medium | secondary, tertiary (+ link) | 6,828 | phạt nhẹ / km |
| small | unclassified, road | 5,254 | phạt nặng / km |

Đoạn có `lanes` ≥ 4 được nâng một mức. OSM chỉ có `lanes` ở 3–69 % đoạn tùy loại, `width` gần như không có.

## Thời gian từng đoạn (`time_<slot>`, giây)

= chiều dài × giây/mét của nhóm đường trong khung giờ, hiệu chỉnh (bình phương tối thiểu không âm, sai số tương đối) cho khớp 14.850 thời gian TomTom (`configs/estimator_params.json`). Tốc độ trung bình (km/h):

| slot | trunk | primary | secondary | tertiary | unclassified |
|---|---|---|---|---|---|
| morning | 24.7 | 27.3 | 27.8 | 16.5 | 18.9 |
| ban_noon | 23.2 | 28.6 | 30.2 | 19.0 | 19.7 |
| afternoon | 21.9 | 28.0 | 28.6 | 16.9 | 19.0 |
| ban_evening | 22.4 | 26.8 | 28.7 | 18.4 | 20.0 |
| night | 28.1 | 29.5 | 29.7 | 17.7 | 18.2 |

Không dùng giới hạn tốc độ (`maxspeed`): thử trên 2.970 chặng cho thời gian chỉ bằng 40 % thực tế (sai số 61 %).

## Luật vành đai

- `ring_deep = 1` (1076 đoạn): trung điểm nằm sâu trong vành đai (cách mép > 25 m, ngoài trục Duy Tân) → **che** trong `ban_noon`, `ban_evening` (CV 4883/UBND-SGTVT 2019).
- `corridor = 1` (110 đoạn): trục Duy Tân – Nguyễn Văn Trỗi – Trần Thị Lý, được đi trong giờ cấm (CV 1710/UBND-SGTVT 2022).
- `pois.csv.ring_deep = 1`: điểm nằm sâu trong vành đai → giờ cấm không tới/đi. Giờ cấm dùng `node_id_ban` (nút còn đường ra vào hợp lệ) thay cho `node_id`.

## Kiểm chứng: đường nhanh nhất trên đồ thị so với TomTom

| slot | nhóm | số chặng | sai số TB | đồ thị ÷ TomTom (trung vị) |
|---|---|---|---|---|
| afternoon | hub | 520 | 18.7 % | 1.02 |
| afternoon | poi | 2450 | 20.6 % | 0.93 |
| afternoon | point | 570 | 13.8 % | 1.02 |
| ban_evening | hub | 404 | 16.3 % | 1.00 |
| ban_evening | poi | 1420 | 19.1 % | 0.94 |
| ban_evening | point | 454 | 13.5 % | 0.99 |
| ban_noon | hub | 404 | 15.5 % | 1.00 |
| ban_noon | poi | 1425 | 18.8 % | 0.95 |
| ban_noon | point | 454 | 13.1 % | 1.00 |
| morning | hub | 520 | 18.3 % | 1.00 |
| morning | poi | 2450 | 21.2 % | 0.92 |
| morning | point | 570 | 15.0 % | 0.99 |
| night | hub | 520 | 16.4 % | 0.98 |
| night | poi | 2450 | 20.2 % | 0.92 |
| night | point | 570 | 13.8 % | 1.00 |

`point` = 5 điểm kiểm tra T01–T05, không dùng ở bất kỳ bước hiệu chỉnh nào. Khớp "đi được / không": 100 % giờ thường, ~98 % giờ cấm.

## Hạn chế

- Thời gian từng đoạn là **ước lượng suy ra từ TomTom**, không phải số đo trực tiếp; không có phí dừng đèn / rẽ riêng ở nút.
- Một số POI cách đường xe lớn xa: Đỉnh Bàn Cờ 1,9 km, Núi Thần Tài 0,5 km, Đầm Sen 0,4 km (`snap_dist_m`).
- `pois_passed` dùng bán kính 100 m quanh hình dạng thật của đoạn.
