# Dataset đồ thị đường cho RL – xe 45 chỗ, Đà Nẵng (bản 10/11/2026)

Bài toán: đi từ A đến B bất kỳ trong thời gian C, **tối đa tổng điểm POI đi ngang**. RL dò đường từng ngã rẽ.

## 3 file

| File | Dòng | Cột | Ý nghĩa |
|---|---|---|---|
| `pois.csv` | 50 | `point_id`, `name`, `score` | POI và điểm hấp dẫn (Bayes trên số sao Google, 3,97–4,54 → nên đưa về 1–10) |
| `road_nodes.csv` | 37,713 | `node_id`, `lat`, `lon`, `ring_deep` | Nút (ngã rẽ). `ring_deep = 1`: sâu trong vành đai → A/B không hợp lệ trong giờ cấm |
| `road_edges.csv` | 84,766 | `edge_id`, `from_node`, `to_node`, `size_level`, `time_morning`, `time_ban_noon`, `time_afternoon`, `time_ban_evening`, `time_night`, `ring_deep`, `pois_passed` | Đoạn đường có hướng = 1 lựa chọn của RL |

- `size_level`: `big` / `medium` / `small` / `local` – theo **bề rộng mặt đường thật** (CSDL đường phố Đà Nẵng: ≥ 14 m, 9–14, 7–9, 5–7 m) khi có, còn lại theo loại đường OSM. Đoạn < 5 m, kiệt, hẻm đã bị bỏ.
- `time_<slot>`: giây để chạy hết đoạn. Khung giờ: morning 05–10:30 · ban_noon 10:30–12 (cấm) · afternoon 12–16:30 · ban_evening 16:30–19 (cấm) · night 19–24.
- `ring_deep = 1` (đoạn): **che** trong `ban_noon`, `ban_evening` (vành đai trung tâm cấm xe > 30 chỗ; trục Duy Tân được đi).
- `pois_passed`: mã POI mà đoạn đi ngang (cách nhau bởi `;`).

## Cách dùng trong RL

```
mỗi bước:  ở from_node → chọn 1 đoạn đi ra (che ring_deep nếu giờ cấm)
           thời gian += time_<khung giờ hiện tại>
           reward đường đi theo size_level: đường to cộng, đường hẹp trừ nặng
           ghi nhận POI trong pois_passed (mỗi POI một lần)
kết thúc:  tới B trong C
chọn tuyến: dò lại nhiều lần, lấy tuyến có TỔNG score POI cao nhất
```
Hai loại điểm **tách riêng**: reward đường đi (dạy dò đường) và điểm POI (chọn tuyến). Trọng số reward nằm trong code env, không trong dataset.

## Độ tin cậy

Đường nhanh nhất trên đồ thị so với TomTom: lệch 13–15 % (A/B ngẫu nhiên), 16–20 % (chặng POI); khớp đi được / không 98–100 %. Chi tiết: `road_graph_card.md`.

## Nguồn

© OpenStreetMap contributors (ODbL) · Cổng Thông tin điện tử thành phố Đà Nẵng – CSDL đường phố (www.danang.gov.vn) · thời gian hiệu chỉnh theo TomTom Routing API · CV 4883/UBND-SGTVT 2019, CV 1710/UBND-SGTVT 2022.
