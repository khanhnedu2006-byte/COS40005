# A/B BẤT KỲ – đặc tả cho train, kiểm tra và demo (09/11/2026)

> Bài toán: người dùng **gõ địa chỉ A, địa chỉ B** (như Google Maps), giờ xuất phát và quỹ thời gian **C**.
> Tìm lộ trình A → … → B về trong thời gian C, **tổng điểm POI lớn nhất**, đúng luật vành đai.
> 5 hub (`configs/hubs.py`) KHÔNG phải giới hạn của bài toán – chỉ là ví dụ A/B và một phần bộ kiểm tra.

## 1. Dữ liệu dùng cho mỗi chuyến

| Chặng | Nguồn khi TRAIN | Nguồn khi KIỂM TRA | Nguồn khi DEMO |
|---|---|---|---|
| POI → POI | `data/final/od_matrix.csv` (TomTom) | như train | như train |
| A → POI, POI → B, A → B | `ab_rows()` – ước lượng trên đồ thị OSM | `data/final/test_legs.csv` (TomTom) | TomTom trực tiếp (hoặc `ab_rows()` nếu hết quota) |

`ab_rows(est, A, B, scores=…)` (`src/busroute/graph/ab_estimator.py`) trả về 505 dòng
(5 khung giờ × (50 + 50 + 1)) **cùng cột với `od_matrix.csv`**, `from_type`/`to_type` = `point`.
Ghép với phần POI → POI rồi nạp vào `ODMatrix` như bình thường. Chạy ~0,7 s / cặp A/B.

Độ chính xác của `ab_rows` so với TomTom trên điểm chưa dùng để hiệu chỉnh (5 hub):
sai số trung bình 15–18 % tuỳ khung giờ, khớp "đi được / không" 97,5–99,9 %
(`reports/estimator_validation.csv`). Cài thêm `scipy`.

```python
import numpy as np
from busroute.graph.ab_estimator import load_default, ab_rows, sample_points
est = load_default(pois)                          # pois: [{"id","lat","lon"}, …] – 50 POI
rng = np.random.default_rng(0)
A, B = sample_points(est, 2, rng)                 # hoặc toạ độ từ ô tìm địa chỉ
rows = ab_rows(est, {**A, "id": "A"}, {**B, "id": "B"}, scores=poi_scores)
```

## 2. Sinh chuyến TRAIN

1. **A, B:** `sample_points()` – điểm ngẫu nhiên trên đường xe lớn vào được, vùng Đà Nẵng – Hội An – Bà Nà;
   A ≠ B, cách nhau ≥ 2 km. Thỉnh thoảng (≈ 20 %) cho A hoặc B trùng một POI hay hub để model quen cả trường hợp đó.
2. **Giờ xuất phát:** đều trong 05:00–21:00, **≥ 30 % rơi vào hoặc chạm giờ cấm** (10:30–12:00, 16:30–19:00).
3. **Quỹ thời gian C:** = thời gian đi thẳng A → B ở khung giờ xuất phát × hệ số ngẫu nhiên 1,2–3,0.
4. **Tái sử dụng:** mỗi cặp A/B tính `ab_rows` một lần, dùng cho ~16–32 chuyến (đổi giờ xuất phát, C).
5. Nếu A hoặc B nằm sâu trong vành đai và giờ xuất phát rơi vào giờ cấm → không có chuyến hợp lệ: bỏ, sinh lại.

## 3. Đặc trưng cho model (để chạy được với A/B chưa từng thấy)

- KHÔNG dùng embedding theo mã điểm (P001, H01…) – A/B mỗi lần một khác.
- Mỗi nút: toạ độ (chuẩn hoá), điểm POI (giãn về 1–10), đã thu chưa, thời gian từ điểm hiện tại tới nút
  (khung giờ hiện tại), thời gian ngắn nhất từ nút về B, số POI chạy ngang trên chặng tới nút.
- Toàn cục: thời gian còn lại / C, khung giờ (one-hot), giờ trong ngày.

## 4. Mask (đã có trong `od_env.py`) + một bổ sung

Bổ sung: chặng **xuất phát trước** giờ cấm có `through_ring = 1` mà **giờ tới** rơi vào giờ cấm → che
(nếu không, xe đang ở trong vành đai sau 10:30 / 16:30). Chặng của A/B từ `ab_rows` luôn có `through_ring = 0`
ở giờ cấm vì đồ thị giờ cấm đã bỏ các đoạn trong vành đai.

## 5. Bộ KIỂM TRA (chấm điểm cuối)

- 10 điểm: H01–H05 + T01–T05 (`configs/test_points.py`), thời gian TomTom thật
  (`od_matrix.csv` + `data/final/test_legs.csv`, lấy bằng `fetch_routes.py fetch --with-test-points`).
- 90 cặp A→B có hướng × 5 giờ xuất phát (08:00, 11:00, 14:30, 17:00, 20:00) × 3 mức C (1,5 / 2 / 3 × đi thẳng)
  = **1.350 trường hợp**, giống nhau cho mọi thuật toán (PPO, greedy, Dijkstra, ALNS, VNS).
- Metrics: tỉ lệ hoàn thành, tổng điểm, số POI (chính + dọc đường), tổng thời gian, thời gian chạy,
  **optimality gap** so với điểm tốt nhất của ALNS/VNS trên cùng trường hợp.
- So thêm: model train bằng `ab_rows` nhưng chấm bằng TomTom → cho biết sai số ước lượng có làm hỏng lộ trình không.

## 6. DEMO

Ô tìm địa chỉ (Goong Autocomplete) → toạ độ A, B → kiểm tra: trong vùng phục vụ, không nằm sâu trong vành đai lúc cấm
→ TomTom cho A → POI, POI → B, A → B (~100–200 lượt, 40–80 s) hoặc `ab_rows` (tức thì, sai số ~17 %)
→ model → vẽ lộ trình. API key chỉ để ở back-end (`.env`).
