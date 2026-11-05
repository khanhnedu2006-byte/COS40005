# PROJECT CONTEXT – Tour Bus Routing & Driver Facial Analysis (Da Nang)

> **Dành cho thành viên nhóm:** dán toàn bộ file này vào đầu cuộc trò chuyện với AI (ChatGPT, Claude, Gemini, Copilot…) trước khi hỏi. File mô tả dự án đang vận hành thế nào, các quyết định đã chốt và quy ước code. Khi có quyết định mới, cập nhật file này thay vì giải thích lại cho AI.

---

## 0. Hướng dẫn cho AI đọc file này

Bạn là trợ lý kỹ thuật cho nhóm 5 sinh viên làm capstone **COS40005 – Swinburne Vietnam, cơ sở Hà Nội**. Bài toán đặt ở **Đà Nẵng**. Khi trả lời:

1. **Tuân thủ các quyết định đã chốt ở Mục 3.** Không đề xuất lại các hướng đã loại ở Mục 9, trừ khi được hỏi trực tiếp.
2. **Quy ước code:** tên biến, hàm, class và file **bằng tiếng Anh**; **comment trong code bằng tiếng Việt**.
3. Giải thích cho thành viên bằng **tiếng Việt**. Slide, báo cáo và tài liệu nộp trường viết bằng **tiếng Anh**.
4. Dùng thuật ngữ **"45-seat tour bus"**.
5. Trích dẫn học thuật: bài báo **từ 2018 trở đi**, **có trên Google Scholar**.
6. Thiếu thông tin thì nói rõ là đang giả định, và giả định phải nhất quán với file này.

**Cách làm việc của nhóm:** giao việc và trao đổi trong **nhóm Zalo**; họp online **9 giờ tối thứ Hai hằng tuần trên Google Meet**; **không dùng Jira**. Repo: `github.com/khanhnedu2006-byte/COS40005`.

---

## 1. Bài toán

Một công ty lữ hành ở Đà Nẵng vận hành **xe du lịch 45 chỗ**. Dự án gồm ba phần:

- **Giai đoạn 1 – Routing:** cho điểm đầu **A**, điểm cuối **B**, giờ xuất phát và **ngân sách thời gian**, tìm lộ trình **đi ngang qua nhiều POI nổi tiếng nhất có thể**. Xe chỉ chạy ngang qua, không dừng. Lộ trình phải tránh **vành đai trung tâm trong giờ cấm**.
- **Thử nghiệm thực tế:** nhà xe chạy thử lộ trình và gửi lại **video trong cabin**.
- **Giai đoạn 2 – Driver facial analysis:** mô hình phân tích khuôn mặt tài xế (buồn ngủ, mất tập trung) từ video cabin.

**Phạm vi bản đồ:** toàn bộ **thành phố Đà Nẵng mới**, gồm cả Quảng Nam cũ và Hội An từ 01/07/2025.

**Về mặt học thuật**, đây là **Time-Dependent Orienteering Problem** trên đồ thị đầy đủ giữa các điểm, tức một biến thể của TTDP.

**Đội:**

| Người | Phụ trách |
|---|---|
| Nguyễn Nam Khánh | Trưởng nhóm, dữ liệu, giai đoạn 2 |
| Nguyễn Anh Quân | Dữ liệu (POI) |
| Lê Tuấn Anh | Back-end, hỗ trợ dữ liệu (đa giác vành đai, POI) |
| Hoàng Minh Đức | Mô hình routing |
| Du Nguyễn Đình Khang | Front-end |

Hạn chót: **cuối tháng 11/2026**.

---

## 2. Thay đổi lớn ngày 01–03/11/2026

Nhóm **bỏ cách tính thời gian trên từng đoạn đường OSM** (whitelist đường chính, giới hạn tốc độ, map matching). Thay vào đó:

- **Node = 50 POI + 5 hub** (điểm đón/trả khách). Đồ thị là đồ thị **đầy đủ** giữa 55 điểm.
- **Thời gian di chuyển giữa mỗi cặp điểm lấy từ TomTom Routing API** theo 5 khung giờ:
  - `travelMode=bus`, khai kích thước xe 45 chỗ, có dữ liệu kẹt xe lịch sử;
  - giờ cấm thì gửi kèm vùng tránh là vành đai.
- Toàn bộ dữ liệu train nằm trong **một file `data/final/od_matrix.csv`**.

Chi tiết các bước và trạng thái nằm trong `docs/DATA_TASKS.md` của repo.

---

## 3. Các quyết định thiết kế ĐÃ CHỐT

| # | Quyết định | Chi tiết |
|---|---|---|
| D1 | **Node = POI + hub** | 50 POI trong `configs/pois.py` (mã `P001`–`P050`) và 5 hub trong `configs/hubs.py` (mã `H01`–`H05`, **đang chờ nhóm duyệt**). A và B được chọn trong số các hub. |
| D2 | **Thời gian = TomTom API theo khung giờ** | `scripts/fetch_routes.py`. Mỗi cặp × mỗi khung giờ là 1 request, `departAt` là thứ Tư sắp tới. **Không khai tải trọng xe**, vì nếu khai thì TomTom áp giờ cấm xe tải và bắt đi vòng. Kiểm chứng bằng 20 chặng GPS thật (lệnh `validate`). |
| D3 | **Cấm theo giờ ở vành đai trung tâm** | Công văn 4883/UBND-SGTVT (19/07/2019): xe khách trên 30 chỗ không vào vành đai trong **10:30–12:00** và **16:30–19:00**. Đa giác ở `data/references/central_ring.geojson`.<br>• Một tuyến bị coi là "vào vành đai" khi đi sâu hơn 200 m vào trong, không tính 300 m đầu/cuối và dải 25 m sát mép (dải này là đường bao, được phép đi).<br>• Không còn tuyến hợp lệ thì `feasible = 0`.<br>• **Được phép đi trục Duy Tân – Nguyễn Văn Trỗi – Trần Thị Lý** (Công văn 1710/UBND-SGTVT, 04/2022; GPS 2026 xác nhận xe vẫn đi trục này trong giờ cấm). |
| D4 | **Điểm POI = điểm Bayes trên số sao Google Maps** | `score = (v/(v+m))·R + (m/(v+m))·C`. Trong đó v là số review, R là số sao, m là số review trung bình, C là số sao trung bình. Nhóm đã chọn công thức này. **Bỏ** đánh giá của tài xế vì không liên hệ được tài xế. |
| D5 | **Thuật toán routing: PPO + GNN** | GNN (ví dụ GraphSAGE hoặc attention) trên đồ thị đầy đủ 55 nút. Action là chọn điểm tiếp theo. Action mask che các cặp có `feasible = 0` và các điểm không còn kịp về B. Baseline là greedy và đường ngắn nhất. |
| D6 | **OSM chỉ dùng làm bản đồ nền và để kiểm tra** | Đồ thị chặng 1 (`data/stage1/`) dùng để rà toạ độ POI (`scripts/audit_pois.py`) và vẽ bản đồ. Không dùng nó để tính thời gian. |
| D7 | **Không dùng cloud cho dữ liệu khuôn mặt** | Xử lý offline trên ổ cứng mã hoá tại Việt Nam. Không dùng AWS. Không đưa dữ liệu khuôn mặt lên Google Colab. |
| D8 | **Dữ liệu nhạy cảm không đưa lên git** | Không commit GPS thô, toạ độ chặng GPS hay video. API key để trong `.env` và không bao giờ commit. |

---

## 4. Dữ liệu

### 4.1 File train: `data/final/od_matrix.csv`

55 × 54 cặp × 5 khung giờ = **14.850 dòng**. Các cột:

- `from_id`, `to_id`, `from_type`, `to_type`, `from_lat`, `from_lon`, `to_lat`, `to_lon`
- `to_poi_score`, `slot`, `duration_s`, `distance_m`
- `feasible`, `pois_passed` (danh sách `;`), `through_ring`

Các khung giờ: `morning` 05:00–10:30 · `ban_noon` 10:30–12:00 · `afternoon` 12:00–16:30 · `ban_evening` 16:30–19:00 · `night` 19:00–24:00.

### 4.2 Log GPS hộp đen (từ nhà xe)

- 7 file nằm trong `source_code/dataset/` và **không đưa lên git**. Đã chuẩn hoá bằng `scripts/normalize_dataset.py`.
- Nhà xe **không gửi thêm log**.
- Vai trò hiện tại: tách thành **20 chặng chạy thật** (`scripts/build_gps_trips.py`, ghi vào `data/interim/` và không đưa lên git) để **kiểm chứng thời gian TomTom**. Kết quả tổng hợp ở `reports/gps_validation.csv`.

### 4.3 POI

- 50 POI: Đà Nẵng nội thành, Sơn Trà, Ngũ Hành Sơn, Hội An, Bà Nà.
- `reports/poi_audit.csv` cảnh báo các POI đặt sai chỗ:
  - **trên cầu** (dễ khiến xe phải vòng quay đầu);
  - **cách đường quá xa**;
  - **nằm trong vành đai** (15 điểm; trong giờ cấm sẽ không tới được, đúng luật).
- Asia Park được Google ghi "đóng vĩnh viễn". Nhóm cần quyết định giữ hay bỏ.

---

## 5. Pipeline dữ liệu

```
configs/pois.py + configs/hubs.py ─► audit_pois.py (rà toạ độ) ─► nhóm duyệt
        │
        ▼
fetch_routes.py fetch --provider tomtom   (≈ 2.400 request/ngày, ≈ 6 ngày, tự chạy tiếp)
        ▼
fetch_routes.py build  ─► data/final/od_matrix.csv ─► train PPO + GNN
        ▲
build_gps_trips.py ─► fetch_routes.py validate ─► reports/gps_validation.csv (sai số so với GPS)
```

---

## 6. Môi trường RL (PPO + GNN)

- **State:** điểm hiện tại, giờ hiện tại và khung giờ, thời gian còn lại, các POI đã đi qua. Thêm embedding GNN của 55 nút; đặc trưng của mỗi nút gồm toạ độ, điểm POI, đã đi qua chưa, và thời gian ngắn nhất về B.
- **Action:** chọn điểm tiếp theo. Bị che (mask) khi:
  - `feasible = 0` trong khung giờ hiện tại;
  - POI đã đi qua rồi;
  - `duration_s(i→j) + T_B(j)` lớn hơn thời gian còn lại.
- **Reward:**
  - `+score` của điểm đến;
  - `+score` của các POI trong `pois_passed`, mỗi POI chỉ tính lần đầu;
  - phạt nhỏ theo thời gian;
  - thưởng khi tới B.
  - Gợi ý: đưa `score` về thang 1–10 bằng min–max, vì điểm Bayes chỉ nằm trong khoảng 3,97–4,54.
- **Train:** sinh ngẫu nhiên A, B (trong số hub), giờ xuất phát và ngân sách.
- **Baseline:** greedy (điểm/phút) và đường ngắn nhất A→B.

---

## 7. Giai đoạn 2 – Phân tích khuôn mặt tài xế (khái quát)

- **Đầu vào:** video cabin từ đợt chạy thử.
- **Mục tiêu:** phát hiện buồn ngủ và mất tập trung.
- **Pháp lý:**
  - Nghị định 10/2020 (camera giám sát người lái);
  - Luật Bảo vệ dữ liệu cá nhân 91/2025/QH15: cần sự đồng ý, có ràng buộc chuyển dữ liệu ra nước ngoài, dữ liệu sinh trắc học là dữ liệu nhạy cảm;
  - Australian Privacy Principles.
- **Hệ quả:** xin đồng ý bằng văn bản, xử lý offline trong nước, ẩn danh hoá, giới hạn thời gian lưu.

---

## 8. File chính trong repo

| File | Nội dung |
|---|---|
| `docs/DATA_TASKS.md` | Kế hoạch dữ liệu v2, trạng thái, việc chờ quyết định |
| `scripts/fetch_routes.py` | Lấy thời gian từ TomTom (hoặc Goong). Các lệnh: `diag`, `test`, `compare`, `validate`, `fetch`, `build` |
| `scripts/audit_pois.py` | Rà toạ độ POI và hub, ghi `reports/poi_audit.csv` |
| `scripts/build_gps_trips.py` | Tách chặng GPS thật |
| `configs/pois.py`, `configs/pois_score.py` | 50 POI và cách tính điểm Bayes |
| `configs/hubs.py` | 5 hub đề xuất |
| `data/references/central_ring.geojson` | Đa giác vành đai, Tuấn Anh vẽ 28/10 |
| `src/busroute/routing/` | Khung `env.py`, `gnn.py`, `ppo_train.py`. **Cần viết lại cho đồ thị đầy đủ** |

---

## 9. Các hướng ĐÃ LOẠI – không đề xuất lại

- ❌ Whitelist đường chính (`MAIN_ROADS`), bảng giới hạn tốc độ / Thông tư 38, map matching GPS theo từng đoạn.
- ❌ Routing trên đồ thị đường OSM hàng chục nghìn nút.
- ❌ **Google Routes / Distance Matrix API**: điều khoản cấm tải hàng loạt và cấm dùng kết quả để train AI.
- ❌ Khai tải trọng xe cho TomTom (bị áp giờ cấm xe tải).
- ❌ Double DQN: đã chuyển sang PPO.
- ❌ Đánh giá của tài xế trong điểm POI.
- ❌ AWS hoặc cloud cho dữ liệu khuôn mặt; Google Colab cho dữ liệu khuôn mặt.
- ❌ Phạm vi chỉ nội thành hoặc chỉ tuyến Đà Nẵng → Hội An.

Phương án dự phòng nếu TomTom có vấn đề: **Goong** (`--provider goong`), loại xe `truck`, không có giờ xuất phát.

---

## 10. Câu hỏi còn mở

- [ ] Duyệt 5 hub và chỉnh toạ độ các POI bị cảnh báo, **trước khi chạy `fetch`**.
- [ ] Giữ hay bỏ Asia Park (đã đóng cửa) và Đỉnh Bàn Cờ (xe lớn lên Sơn Trà?).
- [ ] Kết quả `validate`: sai số TomTom so với GPS.
- [ ] Ngân sách thời gian điển hình của tour.
- [ ] Kiến trúc GNN cụ thể và thông số PPO.

---

_Cập nhật lần cuối: 03/11/2026 (chuyển sang thiết kế 55 điểm + TomTom). Ai thay đổi quyết định thì sửa Mục 3 và Mục 9, rồi ghi ngày ở đây._
