# SMOKE.md — bài kiểm hồi quy HCPD

Bài kiểm lặp lại được, chạy **trước và sau mỗi lần đụng code**. Mỗi dòng là một
yêu cầu trong `SRS.md`; bước kiểm lấy nguyên từ tiêu chí chấp nhận của yêu cầu đó.

- **Dòng `Tay`**: chạy app rồi làm bằng tay. Ngân sách ≤ **15 phút** cho toàn bộ
  phần tay.
- **Dòng `Lệnh`**: chạy một lệnh, tính ngân sách **riêng** (≈ 1 phút).
- Mục này **không sửa lỗi**. Dòng nào fail → đổi trạng thái trong `SRS.md` sang
  Broken/Partial và mở một item PB mới; không vá tại chỗ.

Chạy app: `.\.venv\Scripts\python.exe main.py` · Chạy test:
`.\.venv\Scripts\python.exe -m pytest -q`

**Mốc của lần điền này:** commit `b3a439e`, pytest baseline **216 passed / 0 failed**.

---

## Bảng kết quả

| # | ID | Cách | Bước kiểm | Kỳ vọng | Kết quả | Ngày |
|---|---|---|---|---|---|---|
| 1 | FR-CAP-02 | Tay | Chạy app 5 phút, bật F1 | `FPS` > 0 suốt 5 phút; log không có exception từ `CameraWorker` | **PASS** — chạy > 10 phút, FPS ≈ 30, không exception `CameraWorker` nào trong log | 2026-10-11 |
| 2 | FR-TRK-01 | Lệnh | `.\.venv\Scripts\python.exe scripts\compare_solvers.py` — **giải pose từ landmark đã lưu, KHÔNG kiểm phần phát hiện mặt của MediaPipe** (phần đó do dòng 1 và 3 phủ) | 1200/1200 khung giải được | **PASS** — 1200 khung thực từ `artifacts/landmarks.npz`; SQPNP: 0 tz<0, 0 dist rác | 2026-10-08 |
| 3 | FR-TRK-02 | Tay | Che mặt 5 giây rồi bỏ che (chi tiết ở §A) | Không crash; ảnh không nhảy về gốc; F1 → `Face LOST`; bỏ che → bắt lại trong ~1 s; WASD/QE chỉnh tay được lúc mất mặt | **PARTIAL** — che 5 s: không crash, không nhảy về gốc, bắt lại gần như tức thì ✓. **FAIL mục con:** WASD/QE **không hề** đổi `Eye mm`, kể cả khi đang thấy mặt → PB-039, §D | 2026-10-11 |
| 4 | FR-POSE-01 | Lệnh | `.\.venv\Scripts\python.exe scripts\compare_solvers.py`, đọc hàng `SQPNP` | tz<0 = 0% · reproj trung vị ≤ 10.42 px · rung Z ngồi yên p95 ≤ 4.63 mm | **PASS 2/3** — tz<0 = 0 (0.0%) ✓ · reproj med = 10.42 px ✓ · p95 **không đo được bằng lệnh này** (xem §C) | 2026-10-08 |
| 5 | FR-FLT-01 | Tay | Ngồi yên 10 giây, đọc hàng `Eye mm` trên F1 | Ba số không nhảy quá ±1 đơn vị | **PASS (thô)** — `Eye mm` nằm trong ±1 suốt 10 s. Không chứng minh ngưỡng 0,78 mm — xem §C | 2026-10-11 |
| 6 | FR-PROJ-01 | Tay | Đọc hai hàng `P0 [...]` / `P1 [...]` trên F1, dịch đầu ngang rồi dọc | Hai hàng **đổi theo đầu** và không đối xứng (l≠r, b≠t). Phần quét tổng hợp ±100 mm ≤ 5 px **không chạy lại** — xem §C | **PASS (nửa tay)** — hai hàng `P0`/`P1` đổi theo đầu | 2026-10-11 |
| 7 | FR-CFG-01 | Tay | Đổi một knob lớp trong `content.json` → F6 nạp lại; rồi đổi tên `user.json`, khởi động lại (chi tiết + bước hoàn nguyên ở §B) | Thông tin lớp trên F1 đổi theo knob; thiếu `user.json` vẫn khởi động được | **PASS kèm ghi chú** — thiếu `user.json`: app khởi động, log `Z path: LEGACY`; trả lại → `Z path: CALIBRATED (face_scale=0.9555)` ✓. Nhưng `adaptive_k_max` 4→2 **chỉ ăn khi F6 nạp ảnh KHÁC**; nạp lại đúng ảnh cũ thì không đổi gì → PB-041, §D | 2026-10-11 |
| 8 | FR-IMG-01 | Tay | Nạp `D:\PycharmProjects\test_images\t2.png` bằng **cả hai** cách: F6 và kéo-thả vào cửa sổ | Content plane đổi ảnh cả hai lần, không cần khởi động lại | **PASS** — cả F6 lẫn kéo-thả đều đổi ảnh | 2026-10-11 |
| 9 | FR-DEPTH-01 | Tay | Sau khi ảnh đã nạp xong, bấm lần lượt F7, F8, F9, F10. Trong lúc đang nạp ảnh thì nhìn `FPS` | F7–F10 dựng lại lớp **không** chạy lại inference (không có khựng vài giây); `FPS` không về 0 lúc đang nạp | **PASS** — F7–F10 không chạy lại inference. Ghi chú: FPS tụt tạm từ ≈ 30 xuống **13–19** kèm khựng ngắn rồi hồi | 2026-10-11 |
| 10 | FR-LAYER-09 | Tay | Đọc khối `--- layers ---` trên F1 | Đúng K lớp; `z_m` tăng dần từ `layer_z_far_m` tới `layer_z_near_m` | **PASS** — F1 báo K = 3; z đọc từ log: L0 1100 / L1 925 / L2 750 mm (xa→gần) ✓. Ghi chú: **F1 không in `z_m`**, phải đọc log | 2026-10-11 |
| 11 | FR-RND-01 | Tay | Dịch đầu ngang, chụp màn hình ở hai vị trí đầu rồi so | Lớp gần trượt **nhiều hơn** lớp xa | **PASS (yếu)** — mắt x −146 → +73 mm ở z 515/526 mm; chênh dịch gần↔xa dự đoán chỉ **≈ 19 mm**. Ghi chú: câu chữ của SRS **ngược dấu** — xem §D | 2026-10-11 |
| 12 | FR-RND-02 | Tay | Nhoài tới 15–20 cm rồi lùi lại, chụp hai ảnh ở hai khoảng cách (chi tiết ở §A) | Khoảng hở giữa lớp gần và lớp xa đổi thấy rõ | **FAIL** — z 506 → 250 mm ở x ≈ 0 (dịch 256 mm, vượt cả mức 15–20 cm mà tiêu chí đòi), tỉ lệ kích thước gần/xa chỉ đổi **1,28 → 1,35**, tức **5,5%**. Không phải "đổi thấy rõ". Nguyên nhân gốc đã biết, xem §D | 2026-10-11 |
| 13 | NFR-07 | Lệnh | `.\.venv\Scripts\python.exe -m pytest -q` | ≥ 216 passed, 0 failed | **PASS** — `216 passed in 3.05s` | 2026-10-08 |

---

## Kiểm thêm (không đánh số, không thuộc PB-029)

**Dấu của trục X trên F1.** Bật F1, **nghiêng đầu sang phải**, đọc số đầu tiên của
hàng `Eye mm`. Ghi lại nó đi **dương hay âm**: **DƯƠNG** (đo 2026-10-11)

Đây là câu hỏi còn mở trong `MEMORY.md` mục 10, cần cho Phase B1. Lỗi #1 từng làm
phần tinh chỉnh mống mắt *khuếch đại* lỗi thị sai thay vì bù, và dấu của X là chỗ
duy nhất F1 nhìn ra được mà test tổng hợp không thay được. Bản ghi màn hình
2026-10-07 không có overlay F1 nên chưa trả lời được.

---

## §A — Chi tiết hai dòng cần thao tác

**Dòng 3 (FR-TRK-02), che mặt.** Ngồi đúng tư thế thường, bật F1. Lấy tay hoặc tờ
giấy che hẳn mặt khỏi webcam, giữ **5 giây**, nhìn: app còn chạy không, ảnh có
nhảy về vị trí gốc không, F1 có đổi sang `Face LOST` không. Trong lúc còn che, bấm
`W`/`A`/`S`/`D`/`Q`/`E` xem `Eye mm` có đổi không. Bỏ che, bấm giờ xem bao lâu thì
`Face` quay lại `DETECTED` — kỳ vọng **~1 giây**. *(SRS ghi ngưỡng ≤ 30 khung
`[GIẢ ĐỊNH]`; ở ~30 fps thì 30 khung ≈ 1 s. Ngưỡng này chưa ai đo, giữ nhãn
`[GIẢ ĐỊNH]` cho tới khi có số.)*

**Dòng 12 (FR-RND-02), nhoài tới.** Chụp một ảnh màn hình ở khoảng cách ngồi
thường, rồi nhoài tới **15–20 cm** và chụp ảnh thứ hai. So hai ảnh: khoảng hở giữa
lớp gần nhất và lớp xa nhất phải đổi thấy rõ.

## §B — Dòng 7 (FR-CFG-01): bước làm và bước hoàn nguyên

1. Mở `assets/config/content.json`, đổi **một** knob lớp (ví dụ `gvf_threshold`
   hoặc `layer_z_far_m`). **Ghi lại giá trị cũ.**
2. Trong app bấm F6 nạp lại ảnh → khối `--- layers ---` trên F1 phải đổi theo.
3. **Hoàn nguyên:** đặt knob về đúng giá trị cũ.
4. Đổi tên `assets/config/user.json` thành `user.json.bak`, khởi động lại app →
   app vẫn phải chạy được.
5. **Hoàn nguyên:** đổi tên về `user.json`, khởi động lại, và kiểm log khởi động
   có dòng **`Z path: CALIBRATED`**. Nếu log ghi `LEGACY` thì `face_scale` chưa
   được nạp lại — dừng và báo, vì mọi phép đo Z sau đó sẽ sai.

## §C — Hai tiêu chí không kiểm được trong bài này

**Dòng 4, "rung Z ngồi yên p95 ≤ 4.63 mm".** `compare_solvers.py` in cột
`jitter med` = **trung vị độ dịch giữa hai khung liên tiếp** trên bản ghi 1200
khung (`_median_step`, `scripts/compare_solvers.py:117`), SQPNP cho **4.8 mm**.
Đó **không phải** p95 của Z lúc ngồi yên: khác thống kê (trung vị ≠ p95), khác
trục (toàn vector ≠ riêng Z), khác điều kiện (bản ghi có cử động ≠ ngồi yên). Con
số 4.63 mm trong SRS đến từ một phép đo riêng (commit `e38ce93`, `MEMORY.md` mục 5),
không tái lập bằng lệnh này. **PB-028** sẽ đo σ_Z từ CSV đứng yên ≥ 300 khung.

**Dòng 5, "nhiễu sau lọc ≤ 1 mm (đo được 0.78 mm)".** Bước tay ở trên **không
chứng minh ngưỡng 0.78 mm**: F1 in `Eye mm` làm tròn về mm nguyên
(`src/renderer/debug_overlay.py:184`), nên mắt thường không đọc được dưới 1 mm. Nó
chỉ cho biết bộ lọc còn làm việc hay không. Số thật cũng do **PB-028** đo từ CSV.

**Dòng 6, "quét mắt tổng hợp ±100 mm khớp giải tích ≤ 5 px".** Đây là phép quét
tổng hợp, không phải việc tay; bằng chứng đã có từ `kooima_mesh_test`
(`MEMORY.md` mục 5) và **không chạy lại** trong smoke. Phần kiểm tay chỉ phủ vế
sau của tiêu chí (`l≠r, b≠t`, và ma trận đổi theo đầu).

## §D — Phát hiện thêm từ đợt chạy 2026-10-11

**Dòng 12 vì sao FAIL.** Lớp nằm 750 / 925 / 1100 mm **sau** mặt kính. Tỉ lệ
kích thước biểu kiến gần/xa là `(D + 1100) / (D + 750)`:

| D (mm) | tỉ lệ |
|---|---|
| 506 | 1606 / 1256 = **1,279** |
| 250 | 1350 / 1000 = **1,350** |

Nhoài vào gần nửa mét mà tỉ lệ chỉ nhúc nhích 5,5%. Đây **không phải lỗi mới**:
nó là đúng triệu chứng mà **PB-009** (đưa lớp về mặt kính) và **PB-034** (cảnh
hộp) sinh ra để chữa. Không mở item mới; chỉ đổi trạng thái `FR-RND-02` trong
SRS sang **Partial**.

**Câu chữ của FR-RND-01 ngược dấu.** SRS viết "lớp gần trượt **nhiều hơn** lớp
xa". Với mọi lớp nằm sau kính thì ngược lại: lớp ở độ sâu `t` trượt trên màn
theo hệ số `t / (D + t)`, nên `t` càng lớn — tức lớp **càng xa** — càng trượt
nhiều. Số đo khớp: hệ số 0,556 (gần) và 0,647 (xa), chênh 0,091; mắt dịch 219 mm
cho chênh dịch 219 × 0,091 ≈ **19,9 mm**, khớp con số ≈ 19 mm quan sát được.
Câu chữ chỉ đúng nếu có lớp nằm **trước** kính (`t < 0`), là điều PB-009 nhắm
tới. Đề nghị sửa lời SRS, không đổi code.

**Overscan tính cho D = 600 mm nhưng người xem ngồi 250–520 mm.** Dòng 11 và 12
đo được z thật 250–526 mm, trong khi `overscan_design_distance_mm` = 600 và
`e_max` = 150 mm. Càng ngồi gần thì mép lớp càng dễ lọt vào khung — khớp với
hiện tượng **lộ mép ảnh** ở ≈ 20/27 khung của bản ghi 2026-10-07. Liên quan
`FR-LAYER-04` và **PB-034**.

**Dấu trục X đã chốt.** Nghiêng đầu sang phải → `Eye X` **dương**. Khép câu hỏi
mở trong `MEMORY.md` mục 10; cần cho Phase B1.
