# PROJECT CONTEXT – Tour Bus Routing & Driver Facial Analysis (Da Nang)

> **Dành cho thành viên nhóm:** Dán toàn bộ file này vào đầu cuộc trò chuyện với AI (ChatGPT, Claude, Gemini, Copilot…) trước khi hỏi. File mô tả dự án đang vận hành thế nào, các quyết định đã chốt, và quy ước code. Nếu có quyết định mới, cập nhật file này thay vì giải thích lại cho AI từ đầu.

---

## 0. Hướng dẫn cho AI đọc file này

Bạn là trợ lý kỹ thuật cho một nhóm sinh viên 5 người làm capstone **COS40005 – Swinburne Vietnam (Da Nang)**. Khi trả lời:

1. **Tuân thủ các quyết định đã chốt ở Mục 3.** Không đề xuất lại các hướng đã bị loại (Mục 9) trừ khi được hỏi trực tiếp.
2. **Quy ước code:** tên biến, hàm, class, file **bằng tiếng Anh**; **comment trong code bằng tiếng Việt**.
3. Giải thích cho thành viên bằng **tiếng Việt**. Slide, báo cáo, tài liệu nộp trường viết bằng **tiếng Anh**.
4. Dùng thuật ngữ **"45-seat tour bus"**, không dùng "coach".
5. Trích dẫn học thuật: bài báo **từ 2018 trở đi**, **có trên Google Scholar**.
6. Nếu thiếu thông tin, nói rõ là đang giả định, và giả định đó phải nhất quán với file này.

---

## 1. Bài toán

Một công ty lữ hành ở Đà Nẵng vận hành **xe du lịch 45 chỗ**. Họ cần:

- **Giai đoạn 1 – Routing:** Cho điểm đầu **A**, điểm cuối **B** và **giới hạn thời gian** do nhà xe đặt, tìm lộ trình **đi ngang qua nhiều địa điểm nổi tiếng (POI) nhất có thể**: danh thắng, khu vui chơi, cầu nổi tiếng, điểm nghỉ. Đồng thời phải tôn trọng các ràng buộc riêng của xe 45 chỗ: không vào đường hẹp, và tránh vành đai trung tâm vào giờ cấm.
- **Thử nghiệm thực tế:** Nhà xe chạy thử lộ trình và gửi lại **video trong cabin**.
- **Giai đoạn 2 – Driver facial analysis:** Huấn luyện mô hình phân tích khuôn mặt tài xế (buồn ngủ, mất tập trung) từ video cabin.
- **Tùy chọn:** Dùng kết quả giai đoạn 2 để tối ưu lại routing. Nếu hết thời gian thì dự án dừng ở phân tích khuôn mặt.

**Phạm vi bản đồ:** **toàn bộ thành phố Đà Nẵng mới**. Từ 01/07/2025, thành phố bao gồm cả tỉnh Quảng Nam cũ và Hội An. Không giới hạn ở nội thành, và cũng không chỉ là tuyến Đà Nẵng → Hội An.

**Mô hình chuyến đi ở giai đoạn hiện tại:** xe **chỉ chạy ngang qua POI, không dừng**. Việc dừng, thời gian tham quan và giờ mở cửa có thể bổ sung ở giai đoạn sau.

**Về mặt học thuật**, đây là một biến thể của **Orienteering Problem / Tourist Trip Design Problem (TTDP)**: thưởng gắn với đoạn đường, có ngân sách thời gian, và mạng lưới đường thay đổi theo giờ trong ngày.

---

## 2. Thời gian và nhóm

- Nhóm 5 người. Hạn chót **cuối tháng 11/2026**.
- Ưu tiên giai đoạn 1 (routing) và phần xử lý dữ liệu. Giai đoạn 2 phụ thuộc vào việc nhà xe gửi video.

---

## 3. Các quyết định thiết kế ĐÃ CHỐT

| # | Quyết định | Chi tiết |
|---|---|---|
| D1 | **OSM chỉ cung cấp bản đồ nền** | Lấy đồ thị đường qua **OSMnx (Overpass API)**, có cache. Không tải file `.pbf` về. OSM chỉ cho biết nút giao, đoạn đường, độ dài, tên đường và loại đường. |
| D2 | **Không dùng thẻ OSM để suy ra đường cấm** | Không dùng `maxwidth`, `hgv`, `access` hay regex "Kiệt/Hẻm" để quyết định xe đi được hay không. **Mọi ràng buộc đều viết cứng (hard-code)** trong code. |
| D3 | **Whitelist đường chính** | Mặc định **mọi đường bị cấm**, trừ các đường có trong `MAIN_ROADS` và các đoạn tiếp cận `ACCESS_SPURS`. Nhờ vậy, bỏ sót một đường chỉ làm mất một lựa chọn chứ không dẫn xe vào hẻm. |
| D4 | **POI viết cứng là một danh sách trong code** | Mỗi POI là **một node** trên đồ thị, được gắn với một đường chính mà xe đi ngang (`pass_road`). Mỗi POI chỉ được tính điểm **1 lần mỗi chuyến**. |
| D5 | **Cấm theo giờ ở vành đai trung tâm** | Xe khách trên 30 chỗ bị cấm vào vành đai trung tâm Đà Nẵng trong khung **10:30–12:00** và **16:30–19:00** (quy định năm 2019). Vành đai được biểu diễn bằng một đa giác viết cứng. |
| D6 | **Thuật toán routing: PPO + GNN** | Trước đây là Double DQN, đã đổi sang PPO. GNN mã hóa đồ thị đường. Các cạnh bị cấm được che bằng **action mask**. |
| D7 | **Thời gian di chuyển = độ dài ÷ tốc độ** | Tốc độ ưu tiên lấy từ GPS thật. Không có GPS thì dùng giới hạn của Thông tư 38/2024 × hệ số hiệu chỉnh. Luôn chặn trên bởi giới hạn luật. Xem Bước 4. |
| D8 | **Không dùng cloud cho dữ liệu khuôn mặt** | Xử lý offline trên ổ cứng mã hóa đặt tại Việt Nam. Không dùng AWS. Không đưa dữ liệu khuôn mặt tài xế lên Google Colab, vì server đặt ở nước ngoài. |

---

## 4. Dữ liệu hiện có

### 4.1 Log GPS hộp đen (từ nhà xe)
- Nằm ở `source_code/dataset/`. Có **7 file**, mỗi file là một xe trong một ngày, tên file dạng `<biển số>_<YYYY-MM-DD>.csv`.
- Mọi file đã được chuẩn hoá bằng `scripts/normalize_dataset.py` về cùng một định dạng, mỗi cột một thông tin: `bus_id, plate, timestamp (YYYY-MM-DD HH:MM:SS), latitude, longitude, device_speed_kmh, device_km, time_precision`.
  - Cột gộp `V/KM/L` của hộp đen đã được tách ra. Phần nhiên liệu (L) luôn trống nên đã bỏ.
  - `time_precision = minute` đánh dấu bản export cũ (ngày 01/10) không có giây. Bản mới (10, 11 và 14/10) là `second`.
  - Đã xoá 2 file không dùng được cho routing: 43B-05367 ngày 01/10 (đỗ cả ngày, 0 km) và 43H-14909 ngày 01/10 (toàn bộ ở Huế).
- Xe được ẩn danh: `bus_01`=43F-00409, `bus_02`=43H11687, `bus_03`=43H-14909, `bus_04`=43B-05169, `bus_05`=43B-05367.
- Log mới (10–14/10) tốt hơn nhiều: không trùng, có giây, mỗi file 73–176 km. Một số chuyến chạy ra Huế, nên cần **lọc theo ranh giới Đà Nẵng** trước khi hiệu chỉnh tốc độ.
- Có 23–48% dòng trùng lặp. Tốc độ thiết bị báo có lúc sai, ví dụ báo 435 km/h trong khi tính lại chỉ khoảng 44 km/h. Có dấu hiệu file đã bị chỉnh tay trên Google Sheets.
- **Hệ quả:** phần lớn các đoạn đường sẽ dùng **tốc độ mặc định**. Nhóm cần xin nhà xe thêm log 2–4 tuần trên toàn Đà Nẵng, bản export gốc có giây, model xe, và danh sách các tour hay chạy.

### 4.2 POI
- Nhóm tự ghim POI trên bản đồ rồi viết cứng vào code. Có template `poi_template_danang.csv` với 20 POI ứng viên.
- **Điểm POI (1–10)** = 0.5 × tần suất xuất hiện trong tour + 0.3 × số lượt review (đã chuẩn hóa) + 0.2 × đánh giá của tài xế.

---

## 5. Pipeline xử lý dữ liệu (6 bước)

```
[1] Clean GPS → [2] Pull road graph (OSM API) → [3] Add hard-coded POI nodes
     → [4] Estimate travel time → [5] Apply hard-coded restrictions → [6] Export graph dataset
```

### Bước 1 – Làm sạch GPS (`clean_gps.py`, đã xong)
1. Bỏ dòng trùng.
2. Khôi phục giây bằng cách rải đều các điểm trong cùng một phút (chỉ áp dụng cho bản export cũ). Bản export mới (cột tiếng Việt `Ngày tháng, Vĩ độ, Kinh độ`) đã có giây nên dùng trực tiếp.
3. Bỏ điểm cách trung vị hơn **100 km**.
4. Lặp lại việc bỏ điểm có tốc độ suy ra lớn hơn **120 km/h**.
5. Tính lại quãng đường và tốc độ bằng **haversine**, không tin cột tốc độ của thiết bị.
6. Tách chuyến khi mất tín hiệu quá **10 phút**. Một **điểm dừng** là khi xe ở trong bán kính **50 m** ít nhất **3 phút**.

Hằng số trong code: `MAX_SPEED_KMH=120`, `FAR_KM=100`, `STOP_RADIUS_M=50`, `STOP_MIN_MINUTES=3`, `TRIP_GAP_MINUTES=10`.

### Bước 2 – Lấy đồ thị đường từ OSM API
- `osmnx.graph_from_place(...)` hoặc `graph_from_polygon(...)` cho toàn Đà Nẵng mới, với `network_type="drive"`.
- Rút gọn đồ thị (simplify) và giữ thành phần liên thông lớn nhất.
- Node là nút giao, edge là đoạn đường. **Không lọc đường ở bước này.**

### Bước 3 – Thêm POI (viết cứng)
```python
# Danh sách POI – nhóm tự điền, mỗi POI là một node
POIS = [
    {"name": "Dragon Bridge", "lat": ..., "lon": ..., "category": "bridge", "score": 9, "pass_road": "Nguyễn Văn Linh"},
    # ...
]
```
- Gắn POI vào node đường gần nhất. Nếu khoảng cách **lớn hơn 300 m** thì kiểm tra tay.

### Bước 4 – Ước lượng thời gian di chuyển
`travel_time_s = length_m / speed_mps`. OSM chỉ cho **độ dài**; **tốc độ** lấy theo thứ tự ưu tiên:

1. **GPS thật của nhà xe** (sau map matching): đoạn có đủ mẫu (`MIN_SAMPLES`) thì dùng **trung vị tốc độ** theo khung giờ và đánh dấu `verified_by_gps=True`.
2. **Không có GPS:** `giới hạn luật × hệ số hiệu chỉnh`. Hệ số = tốc độ thực ÷ giới hạn luật, đo trên các đoạn đã có GPS, theo loại đường × khung giờ.
3. Mọi tốc độ đều **bị chặn trên bởi giới hạn luật**.

**Giới hạn luật – Thông tư 38/2024/TT-BGTVT** (hiệu lực 01/01/2025):

| Khu vực | Đường đôi (có dải phân cách) | Đường không có dải phân cách |
|---|---|---|
| Khu đông dân cư (mọi xe cơ giới) | 60 km/h | 50 km/h |
| Ngoài khu dân cư (xe chở người trên 28 chỗ) | 80 km/h | 70 km/h |

> ⚠️ Bảng tốc độ cố định cũ (35/30/25/20 km/h) **không có căn cứ** và đã bỏ.
>
> Dữ liệu GPS hiện tại chưa đủ để hiệu chỉnh: trung vị chênh nhau giữa các xe (bus_01 khoảng 40 km/h, bus_02 khoảng 20 km/h, bus_04 khoảng 15 km/h). Trong lúc chờ thêm log, hệ số hiệu chỉnh tạm là một **giả định nêu rõ trong báo cáo** (ví dụ 0.5 ở nội thành). Cần **phân tích độ nhạy**: chạy routing với tốc độ ±20% và xem lộ trình thay đổi thế nào.

### Bước 5 – Áp ràng buộc xe 45 chỗ (viết cứng, whitelist)
```python
# Mạng đường chính được phép đi – nhóm xác nhận với tài xế
MAIN_ROADS = ["Võ Nguyên Giáp", "Trường Sa", "Hoàng Sa", "Nguyễn Văn Linh", "Bạch Đằng",
              "Trần Hưng Đạo", "Nguyễn Tất Thành", "Điện Biên Phủ", "Ngô Quyền",
              "Lê Văn Hiến", "Võ Chí Công", "Quốc lộ 1A", "ĐT602", ...]

# Đoạn ngắn ngoài mạng chính nhưng cần để tiếp cận POI (theo OSM way ID, kiểm tra kỹ từng đoạn)
ACCESS_SPURS = [{"osm_way_id": 123456789, "poi": "...", "checked_by": "driver/street view"}]

# Vùng cấm theo giờ
TIME_BANNED_AREAS = [{
    "name": "Central ring",
    "polygon": [(lat, lon), ...],
    "windows": [("10:30", "12:00"), ("16:30", "19:00")],
}]

def is_edge_allowed(edge, time_of_day):
    # Mặc định cấm; chỉ cho đi nếu thuộc đường chính hoặc đoạn tiếp cận POI
    on_main = normalize(edge["name"]) in main_road_names
    on_spur = edge["osm_way_id"] in spur_way_ids
    if not (on_main or on_spur):
        return False
    # Cấm theo giờ ở vành đai trung tâm
    for area in TIME_BANNED_AREAS:
        if point_in_polygon(edge["midpoint"], area["polygon"]) and in_windows(time_of_day, area["windows"]):
            return False
    return True
```
- `normalize()` bỏ tiền tố "Đường", bỏ khoảng trắng thừa và viết thường, vì tên đường trên OSM không đồng nhất.
- Sau khi lọc, **kiểm tra liên thông** bằng `networkx`: A, B và mọi POI phải nằm cùng một thành phần. Nếu bị đứt thì thêm đoạn vào `ACCESS_SPURS`.
- **Vòng lặp kiểm chứng:** với mỗi lộ trình sinh ra, liệt kê các đoạn chưa `verified_by_gps`. Nhóm xem lại bằng Street View, rồi sửa danh sách nếu cần.

### Bước 6 – Xuất dataset đồ thị
- **nodes.parquet:** `node_id, lat, lon, is_poi, poi_name, poi_score, in_central_ring`
- **edges.parquet:** `edge_id, from_node, to_node, length_m, road_type, name, osm_way_id, travel_time_<slot>, blocked_<slot>, verified_by_gps`
- Thêm `edge_index` để nạp vào **PyTorch Geometric**.

---

## 6. Môi trường RL (PPO + GNN) – thiết kế đề xuất

> Phần này **chưa cài đặt**. Đây là hướng hiện tại; thay đổi thì cập nhật lại ở đây.

- **State:** node hiện tại, thời gian đã đi, giờ trong ngày, mask các POI đã ghé, thời gian còn lại. Embedding đồ thị lấy từ GNN (ví dụ GraphSAGE hoặc GAT).
- **Action:** chọn một cạnh đi ra từ node hiện tại. Các cạnh bị cấm theo `is_edge_allowed(edge, time_of_day)` bị che bằng **action mask**.
- **Reward:**
  - `+poi_score` khi đi ngang một POI **lần đầu**.
  - Một khoản phạt nhỏ theo thời gian di chuyển.
  - Thưởng khi tới **B** trong ngân sách thời gian.
  - Phạt lớn nếu vượt ngân sách hoặc không tới được B.
- **Kết thúc episode:** khi tới B, hết thời gian, hoặc không còn action hợp lệ.
- **Baseline để so sánh:** đường ngắn nhất (Dijkstra) và greedy theo POI score/thời gian.

---

## 7. Giai đoạn 2 – Phân tích khuôn mặt tài xế (khái quát)

- **Đầu vào:** video cabin từ đợt chạy thử.
- **Mục tiêu:** phát hiện buồn ngủ (nhắm mắt, ngáp) và mất tập trung (quay đầu, nhìn điện thoại).
- **Pháp lý:**
  - **Nghị định 10/2020:** xe từ 9 chỗ trở lên phải có thiết bị giám sát hành trình và camera ghi hình người lái.
  - **Luật Bảo vệ dữ liệu cá nhân 91/2025/QH15:** cần sự đồng ý (Điều 9), có ràng buộc về chuyển dữ liệu ra nước ngoài (Điều 20), và dữ liệu sinh trắc học là dữ liệu nhạy cảm (Điều 31).
  - **Australian Privacy Principles:** vì đây là dự án của Swinburne.
- **Hệ quả:** xin đồng ý bằng văn bản của tài xế, xử lý offline trong nước, ẩn danh hóa, và giới hạn thời gian lưu trữ.

---

## 8. File và tài liệu của dự án

Kiến trúc chi tiết và cấu trúc source code nằm ở `docs/ARCHITECTURE.md`.

| File | Nội dung | Trạng thái |
|---|---|---|
| `src/busroute/gps/clean.py` | Bước 1, làm sạch GPS (trước đây là `clean_gps.py`) | Xong |
| `src/busroute/...` | Khung code cho Bước 2–6 và routing | Đã có khung; các chỗ `TODO` cần nhóm viết tiếp |
| `source_code/dataset/` | 7 file log GPS đã chuẩn hoá | Xong |
| `scripts/normalize_dataset.py` | Chuẩn hoá file export của hộp đen (cả bản cũ và bản mới) | Xong |
| `poi_template_danang.csv` | 20 POI ứng viên | Cần nhóm điền và chấm điểm |
| `Routing_Data_Processing.pptx` | 10 slide về pipeline dữ liệu (tiếng Anh) | Slide 5 và 8 cần sửa theo D2 và D3 |
| `COS40005_Research_Report_Final.docx` | Báo cáo nghiên cứu theo template IEEE, khoảng 2.350 từ, 14 tài liệu tham khảo | **Còn mô tả hướng cũ (Valhalla)**, cần cập nhật sang PPO + GNN và ràng buộc viết cứng |

---

## 9. Các hướng ĐÃ LOẠI – không đề xuất lại

- ❌ Double DQN: đã chuyển sang PPO.
- ❌ Valhalla hoặc routing engine có sẵn làm lõi thuật toán.
- ❌ Dùng thẻ OSM (`highway`, `maxwidth`, `hgv`, regex tên hẻm…) để tự động suy ra đường cấm.
- ❌ Blacklist đường cấm làm cơ chế chính: đã chuyển sang whitelist đường chính.
- ❌ Tải file OSM `.pbf` về máy: dùng API.
- ❌ AWS hoặc server cloud cho huấn luyện và lưu dữ liệu.
- ❌ Google Colab cho dữ liệu khuôn mặt.
- ❌ Google Maps API trả phí cho dữ liệu giao thông thời gian thực.
- ❌ Phạm vi chỉ nội thành hoặc chỉ tuyến Đà Nẵng → Hội An.
- ❌ Dữ liệu dạng ảnh: dữ liệu routing là log GPS.

---

## 10. Câu hỏi còn mở

- [ ] Danh sách `MAIN_ROADS` cuối cùng (cần phỏng vấn tài xế).
- [ ] Đa giác chính xác của vành đai trung tâm.
- [ ] Điểm A, B và ngân sách thời gian điển hình của các tour.
- [ ] Thêm log GPS từ nhà xe.
- [ ] Có thêm điểm dừng và thời gian tham quan ở phiên bản sau hay không.
- [ ] Kiến trúc GNN cụ thể và thông số huấn luyện PPO.

---

_Cập nhật lần cuối: 18/10/2026. Ai thay đổi quyết định thì sửa Mục 3, Mục 9 và ghi ngày ở đây._
