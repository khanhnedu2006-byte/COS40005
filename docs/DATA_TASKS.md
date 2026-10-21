# DATA TASKS – Làm dataset theo 3 chặng nối tiếp (21/10 → 31/10/2026)

> Làm **một lần, đủ toàn bộ**. Dataset được khoá ngày **31/10** và không sửa nữa. Từ **1/11**, người làm routing bắt đầu train.
>
> Mỗi chặng nhận file của chặng trước, **chỉ thêm cột, không sửa cột cũ**. Phải chạy qua script kiểm tra thì mới được giao cho chặng sau.

```
Chặng 1 – Khánh                       21–23/10   Bản đồ nền (code đã có sẵn)
        ↓ giao 23/10
Chặng 2 – Data (chính) + Back-end (hỗ trợ)  24–27/10   Luật đường + POI
        ↓ giao 27/10
Chặng 3 – Khánh                       27–30/10   GPS + thời gian đi + hoàn thiện
        ↓ giao 30/10
Khoá dataset (cả nhóm)            31/10
        ↓
Người làm routing                 từ 1/11
```

Quy ước 5 khung giờ `<slot>`: `morning` 05:00–10:30 · `ban_noon` 10:30–12:00 · `afternoon` 12:00–16:30 · `ban_evening` 16:30–19:00 · `night` 19:00–24:00

---

## Chặng 1 – Khánh (Nguyen Nam Khanh): Bản đồ nền · 21/10 → 23/10

**Trạng thái code: ✅ đã viết xong và đã test** (`tests/test_stage1.py`: 8/8 đạt; `check_stage1.py`: 12/12 mục đạt trên đồ thị giả lập).

| File | Vai trò |
|---|---|
| `src/busroute/graph/osm_fetch.py` | Tải đồ thị qua Overpass theo bbox: tự chuyển máy chủ khi lỗi, thử lại với thời gian chờ tăng dần, lưu cache `.graphml`, giữ thành phần liên thông mạnh lớn nhất |
| `src/busroute/graph/base_tables.py` | Đánh số `node_idx`/`edge_idx`; chuẩn hoá tên đường; `out_degree`; `subgraph_id`; bảng tên đường |
| `scripts/stage1_build_base.py` | Chạy toàn bộ chặng 1 |
| `scripts/check_stage1.py` | Kiểm tra trước khi giao (trả mã lỗi 1 nếu có mục FAIL) |

**Việc còn lại: chạy 2 lệnh trên máy có mạng**
```powershell
cd "D:\VSCode file\Swinburne\COS40005\source_code"
.venv\Scripts\activate
python scripts/stage1_build_base.py     # lần đầu có thể mất 10–30 phút vì phải tải bản đồ
python scripts/check_stage1.py          # phải ra "Sẵn sàng giao chặng 2"
```
- `bbox` trong `configs/settings.yaml` phải là toàn Đà Nẵng mới: `[107.20, 14.95, 108.75, 16.25]`. Nếu trước đó đã tải vùng nhỏ thì **xoá `data/cache/road_graph_drive.graphml`** rồi chạy lại.
- Nếu Overpass lỗi liên tục, chạy lại vào khoảng 6–12 giờ trưa giờ Việt Nam. Cache giữ lại phần đã tải nên không phải tải từ đầu.

**Giao (trưa 23/10):**

| File | Cột |
|---|---|
| `data/stage1/nodes_base.parquet` | `node_idx, osm_node_id, lat, lon, out_degree, subgraph_id` (`core` = nội thành, `hoian`, `outer` = còn lại) |
| `data/stage1/edges_base.parquet` | `edge_idx, from_idx, to_idx, osm_way_id, osm_way_ids, road_name, road_name_raw, road_class, length_m` |
| `reports/osm_road_names.csv` | `road_name, road_name_raw, road_class, n_edges, total_length_km` (dài nhất đứng đầu) |

Ghi chú về cột:
- `road_name`: tên đã chuẩn hoá (viết thường, bỏ chữ "Đường"), dùng để so khớp.
- `road_name_raw`: tên gốc trên OSM, dùng để hiển thị.
- Một cạnh có thể mang nhiều tên, ngăn cách bởi `" | "`.
- `osm_way_ids` chứa mọi way bị gộp vào cạnh, ngăn cách bởi `;`. Cột này dùng cho `ACCESS_SPURS` và `BLOCKED_WAYS`.

**Kiểm tra (`check_stage1.py`):** đủ cột; chỉ số liên tục; không rỗng; `length_m` > 0; mọi node trong bbox Đà Nẵng; đồ thị liên thông mạnh; có vùng `core` và `hoian`; có danh sách tên đường.

---

## Chặng 2 – Người làm data (chính) + Người làm back-end (hỗ trợ): Luật đường + POI · 24/10 → 27/10

**Nhận:** `nodes_base.parquet`, `edges_base.parquet`, `osm_road_names.csv`

**Cách chia:**
- **Người làm data (khoảng 75 % khối lượng):** toàn bộ **nội dung nhập tay**, tức là quyết định đường nào được đi, POI nào, điểm bao nhiêu.
- **Người làm back-end (khoảng 25 %):** **3 script** đọc nội dung đó và áp vào đồ thị. Người làm back-end **không phải** quyết định nội dung.

> Người làm data **nên bắt đầu từ 21/10** những việc không cần bản đồ (D2.1–D2.4), vì 24–27/10 quá ngắn để làm từ đầu.

### Người làm data

| Task | Việc cần làm | Đầu ra | Hạn |
|---|---|---|---|
| **D2.1** | **POI:** khoảng 40 POI, mỗi POI có `poi_id`, tên, toạ độ (5 chữ số thập phân, lấy từ Google Maps), `category` (`bridge`, `beach`, `landmark`, `entertainment`, `rest_stop`), `pass_road` | `config/pois.py` | 23/10 |
| **D2.2** | **Chấm điểm POI 1–10:** 0.5 × tần suất trong tour + 0.3 × số review (chuẩn hoá) + 0.2 × đánh giá tài xế; lưu số liệu gốc | `data/reference/poi_scoring.csv` + cột `score` trong `pois.py` | 26/10 |
| **D2.3** | **Vành đai trung tâm:** vẽ đa giác trên geojson.io theo các tuyến trong quy định 2019, ghi nguồn | `TIME_BANNED_AREAS` trong `config/restrictions.py` | 23/10 |
| **D2.4** | **Nháp `MAIN_ROADS`** (30–60 đường) + `limit_class` của từng đường | `config/roads.py`, `config/speeds.py` | 23/10 |
| **D2.5** | **Sửa tên trong `MAIN_ROADS`** theo báo cáo đối chiếu B2.1 của người làm back-end, để 100 % tên khớp OSM | `config/roads.py` | 25/10 |
| **D2.6** | **Phỏng vấn 2–3 tài xế** (5 câu trong phụ lục), sửa whitelist và `pass_road` theo câu trả lời | `docs/driver_interviews.md` | 26/10 |
| **D2.7** | **`ACCESS_SPURS`:** với mỗi POI trong báo cáo liên thông B2.2, tìm đoạn đường vào trên OSM và Street View; ghi way ID, ngày kiểm tra, người kiểm tra | `ACCESS_SPURS` trong `config/roads.py` | 27/10 |
| **D2.8** | Đọc `connectivity_check.md` bản cuối và xác nhận nội dung đúng | Xác nhận trên Zalo | 27/10 |

### Người làm back-end

| Task | Việc cần làm | Đầu ra | Hạn |
|---|---|---|---|
| **B2.1** | **Đối chiếu tên đường:** so `MAIN_ROADS` (sau `normalize_road_name`) với `osm_road_names.csv`. Liệt kê tên khớp, tên không khớp, và gợi ý 3 tên gần nhất bằng `difflib.get_close_matches` | `scripts/check_road_names.py` + `reports/road_name_check.csv` | 24/10 |
| **B2.2** | **Áp luật vào đồ thị:** đọc file chặng 1 + `config/*.py`, rồi:<br>• gắn `is_main_road`, `is_access_spur`, `limit_class`, `limit_kmh`; **bỏ cạnh ngoài whitelist**<br>• gắn POI vào node gần nhất (≤ 300 m, dùng `haversine_m`) → `is_poi, poi_id, poi_name, poi_category, poi_score`; gắn `passes_poi_id` cho cạnh trên `pass_road`<br>• `in_central_ring` (dùng `point_in_polygon`), `is_pickup_point` (node thuộc đường chính)<br>• giữ thành phần liên thông lớn nhất, liệt kê POI bị tách<br>Có thể dùng lại logic trong `graph/restrictions.py` và `graph/poi_snap.py`. **Chạy lại mỗi khi người làm data sửa config.** | `scripts/stage2_apply_rules.py` + `reports/connectivity_check.md` | 26/10 |
| **B2.3** | **Script kiểm tra chặng 2** (theo mẫu `check_stage1.py`): đủ cột; mọi POI có điểm và nằm trong mạng; không còn cạnh ngoài whitelist; mọi cạnh có `limit_kmh`; vẫn liên thông | `scripts/check_stage2.py` | 27/10 |

**Phối hợp:** người làm data sửa config → người làm back-end chạy lại `stage2_apply_rules.py` → người làm data xem `connectivity_check.md` → lặp lại tới khi `check_stage2.py` đạt hết.

**Giao (tối 27/10):**

| File | Cột thêm so với chặng 1 |
|---|---|
| `data/stage2/nodes_rules.parquet` | `is_poi, poi_id, poi_name, poi_category, poi_score, in_central_ring, is_pickup_point` |
| `data/stage2/edges_rules.parquet` | `is_main_road, is_access_spur, limit_class, limit_kmh, passes_poi_id` (đã bỏ các cạnh ngoài whitelist) |
| `src/busroute/config/*.py` | `pois.py`, `roads.py`, `restrictions.py`, `speeds.py` hoàn chỉnh |
| `reports/connectivity_check.md` | Số node/cạnh sau lọc; các POI đã thêm spur |

---

## Chặng 3 – Khánh (Nguyen Nam Khanh): GPS + thời gian đi + hoàn thiện · 27/10 → 30/10

**Nhận:** `nodes_rules.parquet`, `edges_rules.parquet` và GPS đã làm sạch (`data/interim/gps/`).

> Có thể chuẩn bị từ 21/10: lọc GPS theo ranh giới, đánh dấu `is_moving`, viết sẵn code map matching và hiệu chỉnh tốc độ.

**Việc cần làm:**
1. **Map matching** GPS lên `edges_rules` (kiểu HMM). Ra `verified_by_gps`, `gps_samples`, và `driver_path` cho từng chuyến thật.
2. **Hệ số tốc độ** theo (`limit_class` × khung giờ):
   - chỉ dùng file có `time_precision = second`;
   - nhóm nào thiếu mẫu thì gộp với nhóm gần nhất hoặc dùng mặc định;
   - **giữ riêng 1 ngày log** để đo sai số.
3. Tính cho 5 khung giờ: `speed_kmh_<slot>` (chặn trên bởi `limit_kmh`), `travel_time_s_<slot>`, `blocked_<slot>`.
4. **Đặc trưng:** `x_norm`, `y_norm`; lưu mean/std vào `feature_stats.json`.
5. **Bộ test chuyến thật:** A, B, giờ xuất phát, thời lượng thật, `driver_path`, `driver_poi_score`.
6. Viết **`data_card.md`**: nguồn dữ liệu, các bước xử lý, giả định về tốc độ, giới hạn (chỉ 7 file log), sai số trên ngày giữ riêng.
7. Viết `scripts/check_stage3.py` và chạy qua.

**Giao (tối 30/10) – dataset hoàn chỉnh:**

| File | Nội dung |
|---|---|
| `data/processed/nodes.parquet` | Cột chặng 2 + `x_norm, y_norm` |
| `data/processed/edges.parquet` | Cột chặng 2 + `speed_kmh_<slot>, travel_time_s_<slot>, blocked_<slot>` (×5), `verified_by_gps, gps_samples` |
| `data/processed/edge_index.npy` | `[2, E]` cho PyTorch Geometric |
| `data/processed/calibration.csv` | `limit_class, slot, factor, n_samples, source, holdout_error_pct` |
| `data/processed/feature_stats.json` | mean/std các đặc trưng |
| `data/processed/test_real_trips.parquet` | `trip_id, bus_id, date, start_node_idx, end_node_idx, start_minute, actual_duration_s, driver_path, driver_poi_score` |
| `data/processed/data_card.md` | Mô tả dataset |

**Kiểm tra trước khi giao:**
- Không có giá trị rỗng.
- Mọi `travel_time_s_*` > 0.
- `blocked_ban_*` = True đúng các cạnh trong vành đai (kiểm tra tay 5 cạnh).
- Sai số trên ngày giữ riêng có ghi trong data card.

---

## Khoá dataset · 31/10 (cả nhóm + người làm routing)

- [ ] `check_stage1.py`, `check_stage2.py`, `check_stage3.py` đều chạy qua trên dataset cuối.
- [ ] Đồ thị phủ toàn Đà Nẵng mới và liên thông.
- [ ] Đủ khoảng 40 POI có điểm, POI nào cũng tới được.
- [ ] Không có cạnh ngoài whitelist hoặc `ACCESS_SPURS`.
- [ ] Đủ cột thời gian đi và cột cấm cho 5 khung giờ.
- [ ] Có bộ test chuyến thật và `data_card.md`.
- [ ] Người làm routing nạp dataset và chạy Dijkstra thử trên 5 cặp A/B.

Đạt hết thì gắn tag Git **`dataset-final`**, lưu một bản chỉ đọc lên OneDrive, và **không sửa nữa**.

> Các chuyến train (`episodes`) **không thuộc dataset khoá**. Người làm routing sinh chúng bằng code từ dataset, với seed cố định.

---

## Phụ lục – 5 câu phỏng vấn tài xế (người làm data)
1. Những đường nào ở Đà Nẵng và Hội An xe 45 chỗ **không vào được** hoặc rất khó vào?
2. Đường nào nên **tránh vào giờ nào** (tắc, cấm, chợ, trường học)?
3. Để đến từng điểm tham quan chính, anh thường **đi đường nào**?
4. Xe hay **dừng nghỉ** hoặc đón trả khách ở đâu?
5. Có đoạn nào thường **chạy rất chậm** không? Khoảng bao nhiêu km/h?
