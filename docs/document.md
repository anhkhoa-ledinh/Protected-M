# HCPD — Tài liệu chi tiết toàn bộ hệ thống

Tài liệu tra cứu đầy đủ cho **test_true3D**: pipeline tổng thể, chức năng từng
thư mục, từng file, và **từng hàm** trong file.

> Đối chiếu với code trên đĩa ngày **2026-11-01**, HEAD `cb913c1` của `master`
> (PB-074 → PB-076). `pytest` = **393 passed, 5 subtests passed**.
> Tài liệu liên quan: `docs/KY_THUAT_HE_THONG.md`.
>
> Ký hiệu: hàm/biến bắt đầu bằng `_` là **private** (dùng nội bộ module/class).

---

## 0. Luật cập nhật tài liệu

1. Mỗi khi một thay đổi làm sai một mục bất kỳ trong file này, PHẢI xuất ra một
   khối "PATCH TÀI LIỆU" ở cuối phiên, dạng: mục số mấy, dòng cũ, dòng mới.
   Không chờ được hỏi.
2. Khi phát hiện file này mâu thuẫn với code thật, tin CODE, nói rõ mâu thuẫn,
   và xuất PATCH. Không im lặng suy luận theo file.
3. File này có thể cũ. Với mọi con số hoặc công thức sẽ dùng làm cơ sở cho một
   chẩn đoán, phải xác minh lại bằng code trước khi dựa vào.

---

# PHẦN I — TỔNG QUAN

## 1. Project này là gì

**HCPD (Head-Coupled Perspective Display)** biến màn hình thường thành **cửa sổ
nhìn vào không gian 3D ảo**. Webcam theo dõi vị trí mắt người xem; **hình học
cảnh không bao giờ di chuyển** — chỉ vị trí camera ảo và khối chóp chiếu
(frustum) lệch trục thay đổi. Một tấm ảnh 2D đặt sau "cửa sổ" được tách thành
**nhiều lớp độ sâu rời rạc** ở các mức Z khác nhau, tạo thị sai 2.5D khi người
xem nghiêng đầu.

| Hạng mục | Giá trị |
|---|---|
| Ngôn ngữ | Python 3 |
| Đồ hoạ | GLFW + PyOpenGL (OpenGL 3.3 core) |
| Tracking | MediaPipe Face Landmarker → `cv2.solvePnP` |
| Chiếu | Robert Kooima generalized off-axis frustum |
| Depth | Depth Anything Small (transformers), dự phòng MiDaS small |
| Chạy | `.\.venv\Scripts\python.exe main.py` |
| Test | `.\.venv\Scripts\python.exe -m pytest` (177 test, ~6 s) |
| Git | Repo git, nhánh `master` |

**Webcam không bao giờ hiện trong cảnh chính.** Có một khung PiP nhỏ bật bằng
F4 nhưng đó là HUD gỡ lỗi, không phải nội dung cảnh.

## 2. Pipeline toàn hệ thống

### 2.1. Ba luồng chạy song song

```
┌────────────────────────────────────────────────────────────────────┐
│ Luồng 1  CameraWorker    AsyncCameraCapture → deque(maxlen=2)      │
│ Luồng 2  TrackingWorker  MediaPipe → solvePnP → mắt → lọc          │
│ Luồng 3  Main (GLFW)     đọc PoseBus → VirtualCamera → vẽ          │
└────────────────────────────────────────────────────────────────────┘
                                │
                        PoseBus (pose mới nhất + tuổi)
                                │
                      VirtualCamera.eye_position_mm
                                │
              Kooima off-axis projection (mặt phẳng screen.json)
                                │
            SceneRenderer → scene graph + ContentPlane + các HUD
```

Luồng thứ tư — **DepthWorker** — chạy suy luận độ sâu đơn ảnh **một lần cho mỗi
file media** (không phải mỗi frame), do `ContentPlaneRenderer` sở hữu.

### 2.2. Chuỗi xử lý pose (đầy đủ)

```
webcam → FramePacket (BGR)
  → FaceTracker.process()                    MediaPipe, 478 landmark
  → LandmarkSet {6 điểm PnP, 2 tâm mống mắt}
  → solve_head_pose()                        cv2.solvePnP (EPNP)
  → HeadPoseEstimator.estimate()             ┌ CALIBRATED: dùng thẳng |tvec.z|
                                             └ LEGACY: _monocular_scale
  → EyePositionEstimator.estimate()          nội suy X,Y về phía mống mắt 12% (IRIS_BLEND)
  → chặn biên: X,Y ±2000mm, Z [250, 4000]
  → PoseFilterPipeline.update()              làm mượt + dự đoán
  → PoseBus.publish()
  → VirtualCamera.set_eye_mm()               _sanitize_eye chặn biên lần 2
  → compute_off_axis_projection()            ma trận chiếu Kooima
  → GPU
```

### 2.3. Chuỗi xử lý nội dung ảnh (Mode B — đường chính)

```
nạp ảnh (ImageSource / frame đầu của VideoPlayer)
      │
      ▼ DepthWorker (luồng nền, cache theo đường dẫn)
  depth map float32 HxW, chuẩn hoá [0,1],  1 = gần
      │
      ▼ 1. CHỌN K TỰ ĐỘNG          adaptive_k.py
  lấy mẫu ~10k → k-means 1-D cho K = k_min..k_max
  gvf_variance[K] = 1 − WCSS/TSS, n_init = 4
  LUẬT: K nhỏ nhất có gvf_variance[K] ≥ gvf_threshold (0.90), nếu không → k_max
      │
      ▼ 2. ĐIỀU CHỈNH ĐỘ SÂU NGỮ NGHĨA   semantic_depth.py
  dải phân vị độ sâu → connected components → watershed trên
  (0.72·gradient RGB + 0.28·gradient depth) cho blob lớn
  • làm phẳng vùng phương sai thấp về median (mặt + thân thành một khối)
  • đẩy xa nhau các vùng kề nhau cùng độ sâu nhưng khác vị trí Y
      │
      ▼ 3. PHÂN CỤM ĐỘ SÂU              layer_clustering.py
  k-means 1-D tất định (k-means++ seed, ≤40k mẫu)
  nhãn sắp xếp xa → gần; rank 0 = xa nhất
      │
      ▼ 4. PHỐI MÀU KHÔNG KHÍ           stage_layer_color()
  màu khí quyển = RGB trung bình lớp xa nhất (dự phòng #B8C4D0 nếu quá tối)
  chỉnh trong không gian tuyến tính: tương phản quanh pivot luma của chính lớp,
  pha về màu khí quyển, rồi giảm bão hoà trong HSV
      │
      ▼ 5. VIỀN SÁNG                    add_rim_light()  chỉ lớp gần nhất, 0.3
      │
      ▼ 6. LÀM MỀM BIÊN + RGBA          extract_layer_rgba()
  feather cơ bản = dilate + Gaussian blur; feather rộng hơn ở biên EDGE
  (classify_silhouette_edges: biên ∧ gradient RGB cao);
  color bleed pha RGB biên về ảnh composite phía sau
      │
      ▼ 7. PHÂN LOẠI ĐƯỜNG NỐI (Phase 2a)   boundary_classifier.py
  classify_layer_seams() trên depth GỐC (trước semantic) — CHỈ LƯU, chưa hàn
      │
      ▼ 8. DỰNG LƯỚI
  relief BẬT  → create_layer_relief_vertical_plane_uv (chia nhỏ, phồng Z)
  relief TẮT  → create_vertical_plane_uv (quad phẳng tại layer.z_m)
      │
      ▼ 9. LẤP KHE HỞ (C1)          src/content/disocclusion.py
  với mỗi lớp có thứ gì đó che phía trước:
  • dải = pixel bị lớp gần che, nằm trong bán kính fill_px quanh biên
  • mask inpaint = TOÀN BỘ vùng bị che (không chỉ dải) → Telea chỉ được
    đọc pixel nền, không hút màu tiền cảnh vào khe (C3)
  • alpha nâng lên trong dải, vuốt mềm ở rìa ngoài
      │
      ▼ 10. QUAD SƯƠNG MÙ            giữa hai lớp xa nhất
      │
      ▼ GPU: texture RGBA, vẽ xa→gần, blend, TẮT ghi depth buffer
```

## 3. Hệ quy chiếu & quy ước (sai chỗ này là hỏng hết)

| Không gian | Trục | Đơn vị |
|---|---|---|
| OpenCV camera (output solvePnP) | X phải, Y **xuống**, Z tới trước vào cảnh | mm |
| HCPD world / OpenGL | X phải, Y **lên**, Z **về phía người xem** | pose **mm**, GL **m** |
| Depth map chuẩn hoá | — | `1.0 = gần`, `0.0 = xa` |

**Quan trọng:** `depth` và `depth_norm` là **nghịch đảo nhau**.
`depth_norm` trong code lớp là *độ xa*: `0.0 = lớp gần nhất`, `1.0 = xa nhất`.

Ba hàm chuyển đổi trong `src/pose/head_pose.py`:
- **`R_CV_TO_HCPD = diag(-1, -1, 1)`** — phép đổi hệ **duy nhất**, định thức +1.
  Webcam nhìn về phía người xem nên Z camera đã trùng world +Z; hệ thuận tay
  phải khi đó ép camera X nằm dọc world **−X**.
- `_opencv_to_hcpd_translation`: `R_CV_TO_HCPD @ tvec + webcam_offset_mm`
- `_opencv_rot_to_hcpd`: `R_CV_TO_HCPD · R · R_CV_TO_HCPDᵀ`
- `eye_position.py` đảo dấu **cả hai** offset mống mắt (cùng một quy ước), chia dy cho
  `fy`, rồi cộng `webcam_offset_mm[:2]` để về **cùng gốc tâm màn hình** với
  `translation_mm` trước khi nội suy.
- `_euler_from_rotation`: pitch/yaw/roll độ (chỉ để hiển thị, không dùng cho chiếu)

**Trước 2026-09-13 đây là ba quy ước khác nhau.** Translation dùng
`diag(1,−1,1)` — định thức **−1**, là phép phản chiếu chứ không phải chuyển hệ
rigid — khiến thị sai ngang **ngược dấu**, và cảnh có cảm giác *xoay theo* người
xem thay vì đứng yên như nhìn qua cửa sổ. Rotation dùng `diag(1,−1,−1)`: là phép
quay hợp lệ nhưng quanh **trục khác**; nó chỉ đi vào HUD F1 nên sai lệch sống sót
rất lâu mà không ai thấy. Phần tinh chỉnh mống mắt đảo Y mà **không** đảo X, nên
nó *khuếch đại* lỗi chứ không bù.

Xác nhận bằng **đo thước vật lý**, không phải suy luận: header kết quả ruler của
`tracking_benchmark` ghi `pred' = (-t_x, t_y, -t_z) so +axis matches physical
marks`, với MAE ngang 3.71 mm trên các vạch ±100 mm. Đó là phép đo đầu-cuối trên
đúng webcam đó, nên nó loại trừ luôn khả năng **driver lật gương** — thứ mà đọc
code không thể loại trừ.

`abs(tz)` đã bỏ: độ sâu không dương giờ bị coi là **nghiệm PnP hỏng** và trả về
`detected=False`. Đếm trên 1077 khung của một phiên thật: **0 khung** có
`tz <= 0`, nên `abs` chưa từng cứu gì — nó chỉ che mất một tín hiệu lỗi.

---

# PHẦN II — CHỨC NĂNG TỪNG THƯ MỤC

```
test_true3D/
├── main.py                 Điểm vào: đọc tham số, cấu hình log, khởi chạy app
├── run.bat / run.ps1       Script chạy bằng venv
├── convert_nyu.py          Công cụ độc lập: NYU Depth v2 .mat → ảnh
├── requirements.txt        Khai báo phụ thuộc
│
├── assets/
│   ├── config/             File cấu hình JSON/YAML
│   ├── shaders/            Mã GLSL
│   └── models/             face_landmarker.task (model MediaPipe)
│
├── src/
│   ├── config/    Định nghĩa schema cấu hình + nạp/ghi JSON-YAML
│   ├── camera/    Thu hình webcam (đồng bộ + bất đồng bộ) và đo thời gian
│   ├── tracking/  MediaPipe: phát hiện landmark khuôn mặt và mống mắt
│   ├── pose/      solvePnP, ước lượng pose đầu, vị trí mắt, calibration
│   ├── filters/   Làm mượt thời gian (1€/Kalman/EMA) + dự đoán chuyển động
│   ├── math/      Toán: ma trận, quaternion, chiếu Kooima, frustum, đổi hệ trục
│   ├── engine/    Điều phối ứng dụng, các worker, bus pose, đo hiệu năng
│   ├── renderer/  Cửa sổ GL, shader, mesh, content plane, các lớp HUD
│   ├── scene/     Scene graph và các hàm sinh hình học thủ tục
│   ├── content/   Ước lượng độ sâu + pipeline tách lớp + lấp khe hở (vùng nghiên cứu)
│   └── gui/       Wizard calibration và hộp thoại chọn file
│
├── tests/         26 file pytest, 177 test — không cần OpenGL hay webcam
├── scripts/       Công cụ chẩn đoán/báo cáo offline (chạy headless)
├── artifacts/     PNG và CSV do scripts sinh ra
├── Promt_for_Cursor/  15 file đặc tả nhiệm vụ viết tay (đã gitignore, không lên repo)
└── docs/          Tài liệu
```

---

# PHẦN III — CHI TIẾT TỪNG FILE VÀ TỪNG HÀM

## 3.1. Gốc project

### `main.py` — Điểm vào ứng dụng

| Hàm | Công dụng |
|---|---|
| `parse_args(argv)` | Đọc tham số dòng lệnh: `--config`, `--calibrate`, `-v/--verbose` |
| `main(argv)` | Cấu hình logging, cài `threading.excepthook` để lỗi trong luồng nền không bị nuốt, gộp `default.json` + `user.json`, rồi chạy wizard hoặc `HcpdApplication` |

### `convert_nyu.py`
Công cụ nghiên cứu độc lập, chuyển bộ dữ liệu NYU Depth v2 (`.mat`, 2.97 GB) ra
ảnh. **Không phải phụ thuộc lúc chạy** ứng dụng.

---

## 3.2. `src/config/` — Cấu hình

### `schema.py` — Định nghĩa toàn bộ tham số bằng dataclass

| Lớp | Công dụng |
|---|---|
| `CameraSettings` | Thông số webcam: device_id, độ phân giải, fps, fourcc, kích thước queue |
| `ScreenSettings` | Kích thước màn hình vật lý (mm), độ phân giải (px), tâm màn hình |
| `WebcamMountSettings` | Vị trí webcam so với tâm màn hình (mm) |
| `UserSettings` | `ipd_mm` (khoảng cách đồng tử), `viewing_distance_mm` |
| `FilterSettings` | Loại bộ lọc, tham số 1€/Kalman/EMA, bật/tắt và giới hạn dự đoán |
| `OptimizationSettings` | Ngưỡng latency, giới hạn fps tracking, bật giảm mẫu, cache ma trận chiếu |
| `TrackingSettings` | Ngưỡng tin cậy MediaPipe, đường dẫn model |
| `AdaptiveQualitySettings` | Bậc chất lượng tự động (100/75/50/35), fps mục tiêu và tối thiểu |
| `RendererSettings` | vsync, MSAA, kích thước cửa sổ, thư mục shader |
| `UiSettings` | Bố cục khung PiP webcam |
| `ContentSettings` | **~80 tham số** cho content plane: depth, tách lớp, adaptive K, semantic, khí quyển, sương mù, relief, và bốn khoá PB-074 dưới đây |
| `AppConfig` | Gốc cây cấu hình, gom tất cả nhóm trên |

Bốn khoá hình ảnh thêm ở PB-074 (2026-10-30/31). **Mặc định của cả bốn dựng
lại đúng từng byte ảnh trước đó**, có test chặn; người dùng chọn bằng mắt rồi
mới ghi vào `content.demo.json`:

| Khoá | Giá trị | Nghĩa |
|---|---|---|
| `edge_style` | `"soft"` (mặc định) \| `"diecut"` | PB-072. `diecut` = bỏ viền sáng, dilate 0 + blur 3 px, bỏ feather rộng 18 px, bỏ pha màu lớp sau, và **tẩy màu nền khỏi vành mép** trước khi feather |
| `layer_shadow` | `False` (mặc định) \| `True` | PB-073 "Dimensional Layering": mỗi lớp đổ bóng xuống lớp xa hơn kế tiếp. Bốn khoá đi kèm: `shadow_offset_frac` 0.012, `shadow_blur_frac` 0.010, `shadow_opacity` 0.45, `shadow_angle_deg` 90 |
| `fill_method` | `"telea"` (mặc định) \| `"fsr"` \| `"lama"` | PB-074 Phase 5. Cùng dải, cùng mask, chỉ đổi thuật toán lấp khe. **FSR không dùng được** (đắt ~40×, vân không tốt hơn). **`"lama"`** = big-lama ONNX trên CPU: vân tốt hơn hẳn (4/6 ảnh) nhưng **~11 s mỗi ảnh**, quá ngân sách +6 s. Thiếu model thì tự rơi về `telea` |
| `surround` | `"void"` (mặc định) \| `"box"` | PB-034. `box` = hộp năm mặt quanh các lớp, miệng hộp đúng bằng hình chữ nhật màn hình ở z = 0 |
| `plate_fill` | `"telea"` (mặc định) \| `"shiftmap"` \| `"lama"` | PB-075. Cách lấp riêng cho **tấm nền** (lớp xa nhất, lấp cả lỗ); lớp giữa vẫn theo `fill_method`. Đo ra: cả `shiftmap` (3/6) lẫn `lama` (1/6) đều **trượt** ngưỡng chất lượng 4/6 — và đổi riêng tấm nền chỉ làm `fill_tex_ratio` nhúc nhích **0,091**, so với **0,990** khi đổi mọi lớp |

| Hàm | Công dụng |
|---|---|
| `ContentSettings.resolved_media_path(root)` | Chuyển đường dẫn media tương đối thành tuyệt đối theo gốc project |
| `ContentSettings.resolved_plane_height_m(aspect)` | Tính chiều cao mặt phẳng: lấy từ config, hoặc suy ra từ width/aspect nếu để 0 |
| `AppConfig.to_dict()` | Chuyển toàn cây cấu hình thành dict (dùng để ghi file) |
| `AppConfig.from_dict(data)` | Dựng lại cấu hình từ dict lồng nhau |
| `_merge_dataclass(cls, data)` | Tạo dataclass từ dict, **bỏ qua key lạ** (nên key không khai báo sẽ bị mất im lặng) |

### `loader.py` — Đọc/ghi cấu hình

| Hàm | Công dụng |
|---|---|
| `load_config(path)` | Nạp cấu hình từ `.json` hoặc `.yaml`, trả về `AppConfig` |
| `save_config(config, path)` | Ghi `AppConfig` ra JSON/YAML. **Ghi đè toàn bộ file** — làm mất key ngoài schema |
| `merge_user_config(base, user)` | Nạp `default.json` rồi phủ `user.json` lên bằng deep-merge |
| `_deep_merge(base, override)` | Gộp đệ quy hai dict lồng nhau |
| `_load_yaml(text)` | Parse YAML (yêu cầu PyYAML) |
| `_dump_yaml(payload)` | Serialize sang YAML |

---

## 3.3. `src/camera/` — Thu hình webcam

### `capture.py` — Thu hình đồng bộ

| Lớp / Hàm | Công dụng |
|---|---|
| `CaptureConfig` | Dataclass tham số thu hình (device, w/h, fps, buffer, fourcc) |
| `CaptureConfig.to_dict()` / `.from_dict()` | Serialize / dựng lại từ dict |
| `FramePacket` | Một frame kèm metadata: ảnh, chỉ số frame, timestamp, độ trễ thu |
| `FramePacket.height/.width/.channels` | Thuộc tính đọc kích thước từ mảng ảnh |
| `CameraCapture.__init__(config)` | Khởi tạo (chưa mở thiết bị) |
| `CameraCapture.open()` | Mở webcam qua `cv2.VideoCapture`, ưu tiên backend DSHOW trên Windows, áp đặt độ phân giải/fps/fourcc, log thông số thực tế |
| `CameraCapture.close()` | Giải phóng thiết bị |
| `CameraCapture.read()` | Đọc **một** frame, đo độ trễ, trả `FramePacket` hoặc `None` |
| `CameraCapture.grab_latest(max_discard)` | Xả các frame cũ trong buffer, chỉ giữ frame mới nhất |
| `CameraCapture.get_backend_info()` | Truy vấn thông số thực tế từ driver |
| `.is_open / .config / .frame_index / .timer` | Thuộc tính trạng thái |
| `__enter__ / __exit__` | Hỗ trợ dùng với `with` |

### `async_capture.py` — Thu hình bất đồng bộ

| Lớp / Hàm | Công dụng |
|---|---|
| `AsyncCaptureStats` | Thống kê: số frame sản xuất, số bị bỏ, độ sâu queue, đang chạy hay không |
| `AsyncCameraCapture.__init__` | Bọc `CameraCapture` bằng một `deque` có giới hạn |
| `.start()` | Mở camera và khởi động luồng thu nền |
| `.stop(timeout)` | Dừng luồng, đóng camera, xoá queue |
| `.get_latest()` | Trả frame mới nhất **không** lấy ra khỏi queue (dùng cho HUD preview) |
| `.pop_latest()` | Lấy frame mới nhất **và xoá toàn bộ queue** (dùng cho tracking — luôn ưu tiên frame mới) |
| `.wait_for_frame(timeout)` | Chặn cho đến khi có frame hoặc hết giờ |
| `.stats()` | Trả `AsyncCaptureStats` |
| `._capture_loop()` | Vòng lặp luồng nền: đọc frame, đẩy vào queue, đếm frame bị bỏ |

### `timing.py` — Đo thời gian

| Lớp / Hàm | Công dụng |
|---|---|
| `TimingStats` | Thống kê cuộn: khoảng cách frame, fps, độ trễ thu |
| `FrameTimer.now()` | Trả timestamp đơn điệu (giây) |
| `FrameTimer.tick(capture_latency_ms)` | Ghi nhận một mốc frame, cập nhật thống kê cuộn |
| `FrameTimer.reset()` | Xoá toàn bộ dữ liệu tích luỹ |
| `FrameTimer.elapsed_ms(start)` | Số ms đã trôi từ mốc `start` |

---

## 3.4. `src/tracking/` — Phát hiện landmark

### `landmarks.py` — Chỉ số landmark và model 3D

| Hằng / Lớp / Hàm | Công dụng |
|---|---|
| `PNP_LANDMARKS` | 6 chỉ số MediaPipe dùng cho solvePnP: mũi(1), cằm(152), đuôi mắt trái(33)/phải(263), khoé miệng trái(61)/phải(291) |
| `LEFT_IRIS_CENTER` (468), `RIGHT_IRIS_CENTER` (473) | Chỉ số tâm mống mắt |
| `FACE_MODEL_RAW` | Model mặt 3D **đơn vị tuỳ ý** như bản gốc (lớn gấp ~4.7 lần mm) — giữ lại làm nguồn gốc |
| `FACE_MODEL_UNITS_PER_MM` = 5.0 | Hệ số quy đổi, neo theo nhân trắc học: đuôi mắt ↔ đuôi mắt của model = 450 đơn vị ≈ 90 mm ở người lớn |
| `FACE_MODEL_MM` | Model mặt 3D **theo mm thật** = `FACE_MODEL_RAW / 5.0` (90.0 mm mắt↔mắt, 67.3 mm mũi→cằm) |
| `LandmarkSet` | Gói kết quả: 6 điểm ảnh cho PnP, hai tâm mống mắt, toàn bộ landmark |
| `LandmarkSet.has_iris` | Có dữ liệu mống mắt hay không |
| `normalized_to_pixels(landmarks, w, h)` | Đổi toạ độ chuẩn hoá [0,1] sang pixel |

### `face_tracker.py` — Bọc MediaPipe

| Lớp / Hàm | Công dụng |
|---|---|
| `FaceTrackingResult` | Kết quả một frame: có phát hiện không, `LandmarkSet`, độ tin cậy, kích thước ảnh |
| `FaceTracker.__init__` | Thử khởi tạo **Tasks API** trước, không được thì rơi về **Solutions API** |
| `.process(bgr_image, process_scale)` | Hàm chính: giảm mẫu ảnh (theo adaptive quality) → BGR→RGB → phát hiện → đổi landmark về pixel gốc |
| `._build_landmark_set(...)` | Trích 6 điểm PnP và 2 tâm mống mắt từ mảng landmark đầy đủ |
| `._detect_tasks(rgb)` | Chạy `FaceLandmarker.detect_for_video`, tính độ tin cậy = trung bình `presence` |
| `._detect_solutions(rgb)` | Chạy `FaceMesh.process` (API cũ), luôn trả confidence = 1.0 |
| `._init_tasks_api()` | Nạp `face_landmarker.task`, chế độ VIDEO |
| `._init_solutions_api()` | Dự phòng: `mp.solutions.face_mesh` với `refine_landmarks=True` |
| `._resolve_model_path()` | Tìm file model theo config rồi theo `assets/models/` |
| `.close()` | Giải phóng tài nguyên MediaPipe |
| `.is_available / .backend` | Trạng thái: có dùng được không, đang dùng backend nào |

### `iris.py` — Tiện ích mống mắt

| Hàm | Công dụng |
|---|---|
| `iris_midpoint(landmarks)` | Trung điểm hai tâm mống mắt (pixel) — đây là "vị trí mắt" trên ảnh |
| `inter_pupillary_distance_px(landmarks)` | Khoảng cách pixel giữa hai tâm mống mắt |

---

## 3.5. `src/pose/` — Ước lượng pose và calibration

### `solve_pnp.py`

| Hàm | Công dụng |
|---|---|
| `solve_head_pose(model_mm, image_px, intrinsics, fast)` | Gọi `cv2.solvePnP`; `fast=True` dùng **`SOLVEPNP_SQPNP`** (đổi từ `SOLVEPNP_EPNP` ngày 2026-09-19), ngược lại `SOLVEPNP_ITERATIVE`. Trả `(success, rvec, tvec)`. Bảng đo 1200 khung biện minh cho lựa chọn này nằm ngay trong file, xem §4.6 |

### `calibration.py` — Nội tham số camera + calibration khuôn mặt

**Phần 1 — nội tham số:**

| Lớp / Hàm | Công dụng |
|---|---|
| `CameraIntrinsics` | Dataclass: fx, fy, cx, cy, hệ số méo |
| `.matrix` | Trả ma trận K 3×3 |
| `.from_image_size(w, h, fov_deg=63)` | **Ước lượng** nội tham số từ kích thước ảnh và giả định FOV 63° (đường chạy thật dùng cách này — không có calibration camera thực) |
| `.from_dict(data, w, h)` | Nạp khối intrinsics **đã đo** và co giãn về độ phân giải đang chạy. **Bắt buộc** có `width`/`height` trong khối (fx/fy/cx/cy là đại lượng pixel, không mang sang độ phân giải khác được) và **bắt buộc** khớp tỉ lệ khung hình (lệch > 0.01 → `ValueError`, vì webcam đổi FOV theo sensor mode chứ không resize). Bản cũ mặc định `fx = fy = width` = FOV ngang 53,1°, lệch 23% so với `from_image_size` (63°) — nay **báo lỗi thay vì đoán** |
| `CAMERA_INTRINSICS_KEY` | `"camera_intrinsics"` — tên khối trong `camera_calibration.json` |
| `default_camera_calibration_path()` | Trả `assets/config/camera_calibration.json` |
| `load_camera_intrinsics(w, h, path=None)` | Đọc intrinsics đã đo cho độ phân giải đang chạy; trả `None` khi không dùng được, **và log lý do** (chế độ hỏng ở đây là im lặng: file này từng tồn tại mà không ai đọc) |
| `resolve_intrinsics(w, h, path=None)` | **Một cửa vào duy nhất**: có số đo thì dùng, không thì `from_image_size`. `TrackingWorker` và `CalibrationWizard` phải dùng chung hàm này — calibrate ở một fx rồi chạy ở fx khác là tái tạo đúng cái ghép sai mà `resolve_face_scale` sinh ra để bắt |
| | **Trên máy này (kiểm 2026-10-08):** `camera_calibration.json` hiện có `fx = 960` nhưng **không khai `width`/`height`** → bị từ chối, log `IGNORING camera_calibration.json: …`, và `resolve_intrinsics(1280,720)` trả `fx = fy = 1044.39` — **đúng bằng đường cũ**. Tức #10 hôm nay không đổi một con số nào, nó đổi cơ chế và bắt hệ nói to khi giả định bị thay |

**Phần 2 — calibration `face_scale` (Phase B):**

Nguyên lý: `cv2.solvePnP` trả `tvec` theo **đơn vị của model 3D được đưa vào**.
Nếu mặt thật lớn gấp k lần `FACE_MODEL_MM` thì `tvec.z` sai đúng hệ số k, nên
**một số vô hướng duy nhất** nhân vào model là đủ để khử sai số này.

| Hằng số | Giá trị | Ý nghĩa |
|---|---|---|
| `FACE_SCALE_MIN / MAX` | 0.6 / 1.6 | Cổng kiểm tra tỉnh táo — ngoài khoảng này thì **báo lỗi**, không bao giờ rơi về giá trị mặc định |
| `CALIBRATION_FRAMES` | 60 | Số frame thu liên tiếp |
| `CALIBRATION_MIN_VALID_FRAMES` | 30 | Số frame hợp lệ tối thiểu |
| `CALIBRATION_MIN_PRESENCE` | 0.5 | Ngưỡng presence MediaPipe |
| `CALIBRATION_MAX_RMSE_PX` | 15.0 | Ngưỡng sai số tái chiếu. Nâng từ 6.0 vì đo thật trên `z_diag.csv` cho RMSE trung vị 13.18 px — ngưỡng 6.0 sẽ loại mọi phiên calibration hợp lệ |
| `FACE_CALIBRATION_KEY` | `"face_calibration"` | Tên khối trong `user.json` |

| Lớp / Hàm | Công dụng |
|---|---|
| `FaceCalibrationError` | Ngoại lệ khi calibration thất bại |
| `FaceCalibration` | Kết quả: `face_scale`, khoảng cách D dùng để đo, `tz_median_mm`, `tz_std_mm`, số frame hợp lệ, thời điểm |
| `.to_dict()` / `.from_dict()` | Serialize / dựng lại, có kiểm tra cổng tỉnh táo khi đọc |
| `default_user_config_path()` | Trả đường dẫn `assets/config/user.json` |
| `compute_face_scale(tz_samples, D, *, min_valid_frames, calibrated_at, intrinsics_fx=0.0)` | **Hàm cốt lõi**: `face_scale = D / median(\|tvec.z\|)`. Báo lỗi khi thiếu mẫu, D ≤ 0, hoặc kết quả ngoài `[0.6, 1.6]`. `intrinsics_fx` **ghi lại fx lúc đo** vào kết quả — thiếu nó thì không ai biết scale này thuộc về fx nào |
| `load_face_calibration(path)` | Đọc calibration đã lưu; trả `None` nếu thiếu file/thiếu khối/ngoài cổng → giữ đường legacy |
| `FX_MIGRATION_TOLERANCE` | 0.005 — cổng tỉnh táo cho phép quy đổi theo tỉ số fx |
| `resolve_face_scale(intrinsics, path=None)` | Trả `face_scale` **hợp lệ cho đúng intrinsics này**, hoặc `None` → legacy. `tvec.z` tỉ lệ với fx nên scale đo ở fx này sai ở fx khác đúng theo tỉ số, mà **sai âm thầm** vì cổng `[0.6, 1.6]` quá rộng. Quy đổi `face_scale × fx_stored / fx_live` **không chính xác tuyệt đối** — **phần dư chưa đo `[TBD]`**: bảng số trong docstring của hàm (`src/pose/calibration.py`) không truy được về script, CSV hay test nào, và `test_changed_fx_migrates_by_the_fx_ratio` chỉ kiểm công thức tỉ số. Vẫn áp dụng, vì từ chối sẽ đẩy người dùng về `_monocular_scale` với sai số **không chặn**; hiệu chuẩn lại thì hết |
| `save_face_calibration(calib, path)` | **Gộp** vào `user.json`, giữ nguyên mọi key khác (vì `save_config` ghi đè toàn bộ file) |

### `head_pose.py` — Chuyển output solvePnP thành pose 6DoF

**Phần 1 — chẩn đoán trục Z (Phase A, mặc định TẮT):**

| Hằng / Lớp / Hàm | Công dụng |
|---|---|
| `Z_DIAG_COLUMNS` | 10 cột của `artifacts/z_diag.csv` |
| `_read_z_diagnostics_flag()` | Đọc cờ `debug.z_diagnostics` từ `default.json` rồi `user.json` |
| `_ZDiagnostics.enabled` | Giải cờ một lần (lazy), mở file CSV nếu bật |
| `._open()` | Tạo `artifacts/z_diag.csv`, ghi header, đăng ký `atexit` |
| `.begin_frame()` | Bắt đầu một dòng mới (gọi từ `estimate()`) |
| `.set(**values)` | Điền giá trị vào dòng đang mở |
| `.commit()` | Ghi dòng ra đĩa và tăng `frame_idx` |
| `.close()` | Đóng file |
| `z_diagnostics` | Instance dùng chung toàn tiến trình |
| `_reprojection_rmse_px(...)` | Sai số RMS (pixel) giữa điểm model chiếu lại và landmark thật. **Bất biến với `face_scale`** — nó đo sai lệch *hình dạng*, không phải kích thước |

**Phần 2 — ước lượng pose:**

| Lớp / Hàm | Công dụng |
|---|---|
| `HeadPoseResult` | Kết quả một khung. **Hệ quy chiếu**: `pose.rotation`, `pitch_deg`/`yaw_deg`/`roll_deg` và `translation_mm` đều là **HCPD**; `rotation_matrix` là output **OpenCV thô** của `cv2.Rodrigues`, cố ý giữ nguyên làm chẩn đoán, không có consumer — phải qua `_opencv_rot_to_hcpd` trước khi đem so với bất cứ trường nào ở trên |
| `HeadPoseEstimator.__init__(..., face_scale)` | Nếu có `face_scale` → nhân model và log `"Z path: CALIBRATED"`; không có → dùng model gốc và log `"Z path: LEGACY"` |
| `.update_intrinsics(intrinsics, face_scale=None)` | Nhận intrinsics mới giữa phiên **và mọi thứ phụ thuộc fx**. Trước đây chỉ bộ ước lượng mống mắt được cập nhật, còn solvePnP giải mãi với K cũ — đúng cái ma trận quyết định độ sâu và hướng ngang. Ba thứ xử lý cùng chỗ: `_intrinsics`; `face_scale` (caller phải gọi `resolve_face_scale` rồi truyền vào); `_reference_depth_raw` — **xoá chứ không quy đổi**, vì nó là `tvec.z` thô đo dưới fx cũ, bắt lại ở frame sau là chính xác |
| `.estimate(landmarks, timestamp)` | **Hàm chính.** solvePnP → Rodrigues → nếu `tvec.z <= 0` thì coi là nghiệm PnP hỏng, trả `detected=False` kèm cảnh báo → rẽ nhánh:<br>• **CALIBRATED**: dùng `tvec.z` trực tiếp, **bỏ qua** `_monocular_scale` (chạy cả hai sẽ sửa lỗi hai lần)<br>• **LEGACY**: neo tỉ lệ theo frame đầu<br>rồi đổi hệ trục sang HCPD và trích Euler |
| `._ensure_depth_baseline(depth_raw)` | Ghi lại độ sâu solvePnP ở **frame đầu tiên** làm mốc khử nhập nhằng tỉ lệ đơn ảnh |
| `._monocular_scale()` | Trả `reference_distance / reference_depth_raw`. Docstring giải thích vì sao dùng tỉ lệ **theo từng frame** sẽ triệt tiêu hết biến thiên Z |
| `R_CV_TO_HCPD` | Hằng module `diag(-1, -1, 1)`, định thức **+1**. Phép đổi hệ **duy nhất** cho cả translation, rotation và offset mống mắt (xem §3) |
| `_opencv_rot_to_hcpd(R)` | Đổi ma trận xoay OpenCV → HCPD/OpenGL: `R_CV_TO_HCPD · R · R_CV_TO_HCPDᵀ` |
| `_opencv_to_hcpd_translation(tvec, offset)` | `R_CV_TO_HCPD @ tvec + webcam_offset_mm`. Dạng cũ `[tx, -ty, abs(tz)]` có định thức −1 (phản chiếu) nên thị sai ngang bị ngược. **Không đảo dấu offset ở đây** — từng có bug trừ thay vì cộng gây lệch 206 mm, đã sửa |
| `_euler_from_rotation(R)` | Trích pitch/yaw/roll (độ) từ ma trận 3×3. **Phải truyền ma trận HCPD** (`_opencv_rot_to_hcpd`), không phải OpenCV thô: quaternion của caller ở hệ HCPD, đưa vào hai hệ khác nhau chính là lỗi #4 — trước đây pitch và yaw ngược dấu với quaternion của cùng khung. Liên hợp qua `diag(−1,−1,1)` đưa phép quay quanh **n** về quanh **M·n**, nên **pitch** (quanh X) và **yaw** (quanh Y) đảo dấu còn **roll** (quanh Z) thì không. Đo trên `R_cv = Rz(5)Ry(30)Rx(−12)`: cũ = (−12, +30, 5), mới = (+12, −30, 5). Phân rã nội tại Z-Y-X quanh trục HCPD; nhánh suy biến (`sy < 1e-6`, yaw ±90°) ghim roll = 0 và dồn phần còn lại vào pitch — đó là quy ước suy biến chuẩn, không phải phục hồi. **CHỈ ĐỂ HIỂN THỊ**: Kooima dựng view từ hệ trục màn hình và **vị trí** mắt, hướng đầu không vào phép chiếu. Dấu tuyệt đối của từng góc thừa hưởng từ hệ trục của `FACE_MODEL_MM`, **chưa đối chiếu thước đo vật lý** — chỗ này chỉ bảo đảm *nhất quán* với `rotation` |

### `eye_position.py` — Tinh chỉnh về vị trí mắt

| Hàm | Công dụng |
|---|---|
| `EyePositionEstimator.IRIS_BLEND = 0.12` | Phân số nội suy từ mốc PnP về phía ước lượng mống mắt: 0.0 = giữ nguyên PnP, 1.0 = trùng ước lượng mống mắt. Trước đây 0.12 là trọng số của một vị trí **cộng thêm**, nên thực tế thành hệ số khuếch đại 1,12× |
| `EyePositionEstimator.__init__(ipd_mm, fx, cx, cy, fy=None, webcam_offset_mm=None)` | Lưu nội tham số để quy đổi pixel → mm. `fy=None` → lấy bằng `fx` (pixel vuông, đúng với `from_image_size`). `webcam_offset_mm` là vị trí webcam trong hệ world/HCPD, **bắt buộc** để đưa ước lượng mống mắt về cùng gốc với `translation_mm`; để zeros là coi ống kính nằm ở tâm màn, với `screen.json` hiện tại (+103 mm trên Y) thành lệch dọc hằng |
| `.estimate(head_pose, landmarks, timestamp)` | Dời translation từ tâm đầu về trung điểm hai mống mắt theo **ba bước**: (1) un-project mống mắt trong gốc **camera** — `−(px − c)·depth/f`, đảo dấu **cả X lẫn Y** cùng quy ước `R_CV_TO_HCPD`, dx chia `fx` và **dy chia `fy`**, với `depth = max(translation_mm[2] − webcam_offset_mm[2], 250)`; (2) cộng `webcam_offset_mm[:2]` để về gốc **tâm màn hình**; (3) **nội suy, không cộng**: `eye[:2] += IRIS_BLEND·(iris_world − translation_mm[:2])`. Tính chất định nghĩa: mống mắt chiếu đúng tia PnP thì hiệu chỉnh **đúng bằng 0** — dạng cũ `eye[:2] += 0.12·iris_cam` không bao giờ bằng 0 được, nó cho 1,12× trên X (+12,1 mm tại offset thật 100 mm) và lệch hằng trên Y: −12,5 mm đo trên code cũ (−12,48/−12,46/−12,45 tại z = 500/600/700 mm), so với dạng đóng −0,12 × 103 = −12,4 mm; chênh 0,1 mm là tỉ số z/(z−5), chính là phần lệch gốc Z mà bước (1) dọn nốt. **Chỉ sửa X và Y — Z được truyền nguyên vẹn** |
| `.update_intrinsics(fx, cx, cy, fy=None)` | Cập nhật khi độ phân giải frame thay đổi; `fy=None` → `fy = fx` |
| `.set_webcam_offset_mm(offset)` | Cập nhật vị trí webcam (mm, hệ HCPD) |

---

## 3.6. `src/filters/` — Lọc thời gian và dự đoán

### `one_euro.py` — Bộ lọc 1€ (Casiez et al., 2012)

| Hàm | Công dụng |
|---|---|
| `OneEuroConfig` | `min_cutoff`, `beta`, `d_cutoff` |
| `OneEuroFilter.filter(value, timestamp)` | Lọc thông thấp **thích nghi**: tần số cắt tăng theo tốc độ biến thiên → đứng yên thì mượt, di chuyển nhanh thì ít trễ |
| `._smooth(value, prev, dt, cutoff)` | Một bước lọc thông thấp bậc nhất |
| `.reset()` | Xoá trạng thái |

### `kalman.py`, `ema.py`

| Hàm | Công dụng |
|---|---|
| `KalmanFilter1D.filter(measurement, timestamp)` | Kalman 1-D (dự đoán + hiệu chỉnh) với nhiễu tiến trình/đo lường cấu hình được |
| `EmaFilter.filter(value, timestamp)` | Trung bình trượt luỹ thừa; `timestamp` không dùng, giữ để đồng nhất API |

### `pose_smoother.py` — Áp bộ lọc lên pose

| Hàm | Công dụng |
|---|---|
| `PoseSmoother.__init__(filter_type, ...)` | Tạo **7 bộ lọc vô hướng độc lập**: 3 cho translation + 4 cho quaternion |
| `.smooth(pose)` | Lọc từng kênh rồi **chuẩn hoá lại quaternion** |
| `.set_type(filter_type)` | Đổi thuật toán lúc chạy (dựng lại toàn bộ bộ lọc) |
| `._build_filters(type)` | Factory: `one_euro` / `kalman` / `ema` / `none` |

### `prediction.py` — Dự đoán chuyển động bù độ trễ

| Hàm | Công dụng |
|---|---|
| `PredictionConfig` | Bật/tắt, số frame dự đoán, trần ms, `min_velocity_mm_s` (5.0), `velocity_smoothing` (0.0), `velocity_dt_min_s` (0.0), `velocity_dt_max_s` (0.0). **Ba knob sau mặc định = 0.0 nghĩa là TẮT** và `default.json` không đặt chúng — chúng có dây nối đầy đủ nhưng chưa được kích hoạt (xem ghi chú dưới bảng) |
| `MotionPredictor.observe(pose)` | Cập nhật vận tốc **mà không** ngoại suy |
| `.update(pose)` | Cập nhật vận tốc rồi ngoại suy theo chân trời tính được |
| `.extrapolate(pose, dt)` | Ngoại suy tại **luồng render** theo tuổi của pose |
| `._update_velocity(pose)` | Vận tốc tịnh tiến = Δp/Δt; vận tốc góc từ delta quaternion. Có kẹp `dt` vào `[velocity_dt_min_s, velocity_dt_max_s]` nhưng chỉ khi giá trị > 0 |
| `._extrapolate(pose, dt)` | Bỏ qua nếu tốc độ < `min_velocity_mm_s` (5 mm/s) để không khuếch đại nhiễu lúc ngồi yên |
| `._prediction_seconds()` | Chân trời = `max(prediction_frames/60, độ trễ đo được)`, chặn trên bởi `max_prediction_ms` |
| `.set_measured_latency_ms(ms)` | Nhận độ trễ đo từ `Profiler` để dự đoán thích nghi |
| `_quaternion_delta(a, b)` | Quaternion đưa a về b |
| `_quaternion_to_axis_angle(q)` | Đổi quaternion sang vector trục·góc |
| `_integrate_rotation(q, ω, dt)` | Tích phân vận tốc góc thành quaternion mới |

### Ba knob vận tốc để 0.0 là **chủ ý, không phải sót**

`velocity_smoothing`, `velocity_dt_min_s`, `velocity_dt_max_s` nối dây đầy đủ
schema → pipeline → predictor nhưng **xuất xưởng ở trạng thái tắt**, và
`assets/config/default.json` **cố tình không đặt** chúng. Đừng "sửa" bằng cách
bật lên.

**Bối cảnh.** Chúng ra đời khi truy hiện tượng giật khung, theo giả thuyết rằng
`dt` dao động 0.5–50 ms là thứ phải chặn. Giả thuyết đó **sai chiều**. Dao động
ấy là *triệu chứng*: `_update_velocity` chia cho khoảng thời gian của **vòng
lặp tracking** (tới 120 Hz) thay vì khoảng thời gian **chụp ảnh** của camera
(60 fps). Hệ quả là |v| lúc ngồi yên vọt lên p99 = 84 mm/s so với cổng 5 mm/s,
tức **84.9% số frame bị ngoại suy oan**.

**Chỗ sửa thật** nằm ở `src/engine/tracking_worker.py`: dùng dấu thời gian chụp
của chính packet, và bỏ qua cập nhật vận tốc khi pose lặp lại.

| Số đo | Trước | Sau |
|---|---|---|
| p99 độ dịch pixel, ngồi yên | 3.41 px | **1.14 px** |
| Độ trễ khi đang chuyển động | — | **+0.0 ms** (không đổi) |

**Vì sao vẫn không bật.** Kẹp `dt` chồng lên một nguyên nhân đã bị loại bỏ thì
không thêm giá trị — nó chỉ thêm một tầng che khuất. Lần sau có lỗi mốc thời
gian khác, `dt` sẽ là thứ báo hiệu; kẹp nó đi thì con số đẹp lên còn nguyên
nhân vẫn nằm đó và sẽ lộ ra ở chỗ khác, khó tìm hơn. Giữ 0.0 để `dt` đi tới
`_update_velocity` là `dt` thật.

**Điều kiện xem xét lại.** Nếu p99 độ dịch pixel lúc ngồi yên leo trở lại trên
khoảng **2 px**, hãy kiểm **nguồn `dt`** trước — dấu thời gian chụp, khử trùng
lặp theo `frame_index`, FPS camera so với `track_max_fps`. Chỉ khi nguồn đã
sạch mới cân nhắc ba knob này, và phải kèm số đo p99 trước/sau, không quyết
theo cảm giác.

> Đây là vết thứ ba cùng loại trong dự án. Hai vết trước: `layer_count` bị bỏ
> qua khi adaptive K bật, và năm knob mà `select_optimal_k` nhận rồi vứt. Cả
> hai đều tốn thời gian mới phát hiện, nên lần này lý do được ghi ngay tại
> `src/config/schema.py::FilterSettings` — nơi người đọc gặp ba knob đầu tiên.

### `pipeline.py` — Ghép chuỗi lọc

| Hàm | Công dụng |
|---|---|
| `FilterStageInfo` | Mô tả chuỗi lọc đang hoạt động (để hiển thị lên HUD) |
| `PoseFilterPipeline.update(pose, *, new_measurement=True)` | **Điểm vào chính**: làm mượt → (tuỳ chọn) dự đoán. `new_measurement=False` khi pose bị lặp lại (chưa có frame mới) → bỏ qua cập nhật vận tốc để không chia cho `dt` giả |
| `.cycle_smoother()` | Xoay vòng one_euro → kalman → ema → none (phím F2) |
| `.toggle_prediction()` | Bật/tắt dự đoán (phím F3) |
| `.extrapolate_for_display(pose, age)` | Ngoại suy tại thời điểm render |
| `.set_measured_latency_ms(ms)` | Chuyển độ trễ đo xuống predictor |
| `.set_smoother(type)` / `.reset()` | Đặt thuật toán / xoá trạng thái |

---

## 3.7. `src/math/` — Toán học

> `coordinates.py` và `frustum.py` **đã xoá** (2026-09-22, commit `fb78ce8`).
> Cả hai không có caller và không có test, và mỗi file là một bản sao thứ hai
> của một quy ước đã có chỗ đứng duy nhất: `screen_mm_to_world` **trừ**
> `webcam_offset_mm` trong khi đường chạy thật
> (`head_pose._opencv_to_hcpd_translation`) **cộng** — đúng dạng lỗi 206 mm ở
> mục 3; `opencv_point_to_opengl` dùng `diag(1,−1,−1)`, quy ước thứ ba;
> `frustum.py` dựng chiếu lệch trục theo **mm** trong khi `projection.py`
> làm việc theo **mét**. Phép đổi hệ duy nhất là
> `src.pose.head_pose.R_CV_TO_HCPD`. Đừng viết lại chúng.

### `projection.py` — Chiếu lệch trục Kooima (**trái tim của HCPD**)

| Hàm | Công dụng |
|---|---|
| `off_axis_projection(l, r, b, t, near, far)` | Dựng ma trận chiếu phối cảnh **lệch trục** tổng quát từ các mặt biên tại near plane |
| `compute_off_axis_projection(pa, pb, pc, pe, near, far, depth_offset_m)` | **Hàm chính.** Từ 3 góc màn hình (trái-dưới, phải-dưới, trái-trên) và vị trí mắt: dựng hệ trục màn hình `vr, vu, vn`, tính khoảng cách vuông góc `d`, suy ra `l/r/b/t`, trả `(projection, view)` |
| `ComputeOffAxisProjection(...)` | Bí danh kiểu C++, giữ để đồng bộ tên gọi |
| `_screen_view_matrix(vr, vu, vn, eye)` | Dựng ma trận view từ hệ trục màn hình và vị trí mắt |
| `_normalize(v)` | Chuẩn hoá vector, báo lỗi nếu độ dài ~0 |

Công thức:
```
vr = norm(pb − pa);  vu = norm(pc − pa);  vn = norm(vr × vu)
va = pa − pe;  vb = pb − pe;  vc = pc − pe
d  = −dot(va, vn) + depth_offset
l = near·dot(vr,va)/d    r = near·dot(vr,vb)/d
b = near·dot(vu,va)/d    t = near·dot(vu,vc)/d
```

### `homogeneous.py` — Ma trận 4×4

| Hàm | Công dụng |
|---|---|
| `identity4()` | Ma trận đơn vị 4×4 |
| `translation_matrix(x,y,z)` | Ma trận tịnh tiến |
| `scale_matrix(sx,sy,sz)` | Ma trận co giãn |
| `multiply(a,b)` | Nhân hai ma trận (`a @ b`) |
| `transform_point(m, p)` | Biến đổi điểm 3D (có tịnh tiến) |
| `transform_direction(m, d)` | Biến đổi vector hướng (bỏ qua tịnh tiến) |
| `invert(m)` | Nghịch đảo |
| `to_column_major_flat(m)` | Duỗi phẳng theo **cột** thành float32 để nạp lên OpenGL |

### `quaternion.py`

| Hàm | Công dụng |
|---|---|
| `identity_quat()` | Quaternion đơn vị `[w,x,y,z]` |
| `normalize(q)` | Chuẩn hoá về độ dài 1 |
| `multiply(a,b)` | Tích Hamilton |
| `conjugate(q)` | Liên hợp |
| `from_axis_angle(axis, rad)` | Dựng từ trục và góc |
| `to_rotation_matrix(q)` | Quaternion → ma trận xoay 4×4 |
| `from_rotation_matrix(m)` | Ma trận xoay → quaternion |
| `rotate_vector(q, v)` | Xoay vector bằng quaternion |

### `pose_utils.py`

| Hàm | Công dụng |
|---|---|
| `Pose6DoF` | **Kiểu dữ liệu trung tâm**: translation (mm), rotation (quaternion), timestamp |
| `.as_matrix()` | Đổi pose thành ma trận biến đổi 4×4 |
| `compose_poses(parent, child)` | Ghép hai pose |
| `invert_pose(pose)` | Nghịch đảo pose |
| `interpolate_pose(a, b, alpha)` | Nội suy tuyến tính translation + **SLERP** cho quaternion |
| `identity_pose(timestamp)` | Pose gốc toạ độ |

---

## 3.8. `src/engine/` — Điều phối

### `app.py` — `HcpdApplication` (lớp điều phối chính)

| Hàm | Công dụng |
|---|---|
| `__init__(config, project_root)` | Dựng `PoseBus`, chuỗi lọc, `Profiler`, adaptive quality; **nạp screen.json trước** để `camera_offset_mm` kịp tới `TrackingWorker`; tạo camera worker, tracking worker, cửa sổ GLFW, `VirtualCamera` |
| `_load_screen_calibration(root)` | Đọc `screen.json` (đơn vị **mét**), quy `camera_offset` sang mm. Nếu thiếu file thì lấy mặc định từ `default.json` |
| `_log_calibration()` | Log kích thước màn hình và offset webcam đã nạp |
| `_load_ui_settings(root)` | Nạp `ui.json` phủ lên `UiSettings`, trả `WebcamPreviewSettings` |
| `_load_content(root)` | Đọc `content.json` rồi uỷ quyền cho `ContentSettings.from_dict`. Chỉ lo phần I/O: thiếu file, JSON hỏng, root không phải object — cả ba đều rơi về mặc định thay vì làm chết lúc khởi động |
| `_build_scene()` | Chọn cảnh: `build_video_window_scene` (chế độ video) hoặc `build_room_scene` |
| `_try_load_media(path)` | Nạp media từ hộp thoại/kéo-thả, hiện toast, ghi lại đường dẫn vào `content.json` |
| `run()` | **Vòng lặp render chính**: bắt đầu đo frame → đọc phím → lấy pose từ bus → ngoại suy → cập nhật camera ảo → vẽ → swap buffer → cập nhật profiler và adaptive quality |
| `_build_metrics(pose, pitch, yaw, roll)` | Gom số liệu cho HUD gỡ lỗi F1 |
| `_shutdown()` | Dừng tracking, camera, giải phóng renderer, huỷ cửa sổ |

### `camera_worker.py`

| Hàm | Công dụng |
|---|---|
| `CameraWorker.start()/.stop()` | Khởi động/dừng luồng thu hình |
| `.pop_frame()` | Lấy frame mới nhất cho tracking (xoá queue) |
| `.peek_frame()` | Xem frame mới nhất cho HUD preview (không xoá) |
| `._loop()` | Vòng lặp nền chỉ để cập nhật `capture_latency_ms` |

### `tracking_worker.py` — `TrackingWorker` (luồng 2)

| Hàm | Công dụng |
|---|---|
| `__init__(...)` | Dựng nội tham số qua **`resolve_intrinsics`** (số đo nếu có, không thì FOV giả định), lấy `face_scale` qua **`resolve_face_scale(intrinsics)`** — không phải `load_face_calibration`, vì scale chỉ hợp lệ với fx nó được đo cùng — rồi tạo `HeadPoseEstimator`, `EyePositionEstimator` (có `fy` và `webcam_offset_mm`), `FaceTracker` |
| `.start()/.stop()` | Khởi động/dừng luồng |
| `.preview_snapshot()` | Bản sao an toàn luồng của dữ liệu landmark cho HUD PiP |
| `.set_tracking_scale(scale)` | Nhận hệ số giảm mẫu từ adaptive quality (kẹp 0.25–1.0) |
| `.apply_input(mx, my, mz, dt)` | Cộng dồn dịch chuyển thủ công bằng phím WASD/QE (180 mm/s) |
| `.head_angles_deg()` | Trả pitch/yaw/roll của lần ước lượng gần nhất |
| `._loop()` | Vòng lặp: bắt ngoại lệ từng vòng để một frame lỗi không giết luồng |
| `._loop_once()` | Một chu kỳ: lấy frame → ước lượng pose → cộng input phím → **chặn biên** X,Y ±2000, Z [250,4000] → lọc → publish lên `PoseBus` → tự giới hạn tốc độ theo `track_max_fps`. Khi `tracking.image_width` khác `_intrinsics.width` thì **cả hai** bộ ước lượng được cập nhật: `_head_pose.update_intrinsics(...)` kèm `face_scale` vừa resolve lại, và `_eye_estimator.update_intrinsics(..., fy=...)`. Trước đây chỉ bộ thứ hai được cập nhật |
| `._estimate_pose(packet, ts)` | Chạy MediaPipe → head pose → eye position; xử lý **giữ pose khi độ tin cậy thấp** (< 0.25, giữ tối đa 8 frame) |
| `._hold_or_simulate(ts)` | Mất mặt: giữ pose cũ nếu trước đó có, không thì mô phỏng |
| `._simulated_pose(ts)` | Pose giả khi không có camera/MediaPipe (yaw suy từ X mắt cuối) |

### `pose_bus.py` — Kênh truyền pose giữa luồng

| Hàm | Công dụng |
|---|---|
| `PoseSnapshot.age_ms(now)` | Số ms kể từ lúc pose được công bố |
| `PoseBus.publish(pose)` | Luồng tracking ghi pose mới (có khoá) |
| `PoseBus.latest()` | Luồng render đọc pose mới nhất |
| `PoseBus.pose_age_ms()` | Tuổi của pose mới nhất |

### `profiler.py`, `latency_budget.py`, `adaptive_quality.py`

| Hàm | Công dụng |
|---|---|
| `FrameProfile` | Bảng phân rã độ trễ một frame |
| `Profiler.begin_frame()` | Đánh dấu mốc bắt đầu frame |
| `Profiler.end_frame(start)` | Cập nhật FPS cuộn (cửa sổ 120 frame) và gọi `LatencyBudget.update` |
| `LatencySnapshot` | Ảnh chụp độ trễ đã làm mượt |
| `LatencyBudget.update(cap, track, render, age)` | Làm mượt EMA (α=0.15) từng chặng, tính `compensation = min(pipeline + 0.5·pose_age, 2·target)` — con số này nuôi chân trời dự đoán |
| `LatencyBudget._smooth(prev, sample)` | Một bước EMA |
| `QualityState` | Bậc chất lượng hiện tại và hệ số co |
| `AdaptiveQualityController.update(fps)` | Hạ bậc 100→75→50→35 khi fps < `min_fps` (45); chỉ nâng lại sau **60 frame ổn định** trên mục tiêu |
| `.reset()` | Về bậc chất lượng cao nhất |

---

## 3.9. `src/renderer/` — Kết xuất OpenGL

### `window.py` — Cửa sổ GLFW

| Lớp / Hàm | Công dụng |
|---|---|
| `InputState` | Ảnh chụp input một frame (dịch chuyển, các cờ toggle, thoát) |
| `GlfwWindow.create()` | Khởi tạo GLFW, context GL 3.3 core, MSAA, vsync, đăng ký callback phím và kéo-thả, bật depth test + cull face |
| `.poll_input()` | Đọc sự kiện, gom phím giữ (WASD/QE) và các **cờ một-lần** do callback đặt, rồi xoá cờ |
| `._on_key(...)` | Callback: đặt cờ cho F1/F2/F3/F4/F6/F7/F8/F10 và +/− |
| `._on_drop(window, paths)` | Callback kéo-thả file, xử lý cả chuỗi bytes |
| `.pop_dropped_path()` | Lấy đường dẫn đã thả tiếp theo |
| `.swap_buffers()` / `.destroy()` | Trình bày back buffer / huỷ cửa sổ |

### `camera.py` — `VirtualCamera`

| Hàm | Công dụng |
|---|---|
| `__post_init__()` | Làm sạch vị trí mắt ban đầu, khởi tạo cache ma trận |
| `.set_eye_mm(pos)` | Cập nhật vị trí mắt (có làm sạch) |
| `._sanitize_eye(pos)` | **Lá chắn an toàn**: loại giá trị không hữu hạn, kẹp X,Y ±2000 mm và Z vào `[center_z+150, 4000]`. Một pose rác lọt vào đây sẽ làm sụp biên frustum |
| `.adjust_zoom(delta_mm)` | Cộng offset vào `d` của Kooima — **không** dời vị trí mắt được theo dõi (phím +/−) |
| `.pa / .pb / .pc / .pd` | Bốn góc màn hình vật lý suy từ kích thước và tâm |
| `.projection_matrix()` / `.view_matrix()` | Ma trận chiếu / view |
| `.view_projection_flat()` | Trả bản column-major float32 **có cache**, chỉ tính lại khi mắt dịch > 0.05 mm hoặc zoom đổi |
| `.debug_frustum_lines()` | Sinh đoạn thẳng thế giới: khung màn hình, hình chữ nhật near/far, tia từ mắt (vẽ khi bật F1) |
| `._needs_recompute()` | Quyết định có phải tính lại ma trận không |
| `.eye_position_m` | Vị trí mắt theo **mét** cho không gian OpenGL |

### `shader.py`

| Hàm | Công dụng |
|---|---|
| `ShaderProgram.__init__(vert, frag)` | Biên dịch + link cặp shader, tạo cache vị trí uniform |
| `.use()` | Kích hoạt chương trình |
| `.set_mat4/.set_vec3/.set_float` | Nạp uniform. **Lưu ý:** `set_mat4` dùng `GL_TRUE` nên ma trận được chuyển vị khi nạp |
| `._location(name)` | Tra vị trí uniform (có cache) |
| `._build(...)` / `._compile(...)` | Biên dịch shader và báo lỗi biên dịch/link |

### `mesh.py`, `textured_mesh.py`

| Lớp | Công dụng |
|---|---|
| `MeshData` / `Mesh` | Dữ liệu và mesh GPU **stride 6** (vị trí + pháp tuyến) — dùng cho vật thể Phong |
| `TexturedMeshData` / `TexturedMesh` | Dữ liệu và mesh GPU **stride 5** (vị trí + UV) — dùng cho mặt phẳng nội dung |
| `.draw()` / `.release()` | Vẽ (glDrawElements) / giải phóng VAO-VBO-EBO |

### `renderer.py` — `SceneRenderer`

| Hàm | Công dụng |
|---|---|
| `__init__(...)` | Nạp shader Phong và line, tạo `DebugGeometryRenderer`, `DebugOverlay`, `WebcamPreviewOverlay`, `ToastOverlay`, và `ContentPlaneRenderer` (nếu bật chế độ video) |
| `.initialize()` | Khởi tạo tài nguyên GPU của các thành phần con |
| `.render(metrics, frame, preview, dt)` | **Vẽ một frame**: xoá màn → nạp uniform ánh sáng Phong → duyệt scene graph → vẽ content plane (`update` → `take_build_notice` → `draw`, PB-076) → vẽ frustum gỡ lỗi → HUD → PiP → toast |
| `.load_content_file(path)` | Đổi media nóng, tự tạo content plane nếu chưa có |
| `._ensure_content_plane()` | Dựng cảnh video + content plane khi đang ở chế độ phòng |
| `.toggle_atmospheric()` / `.toggle_mist()` / `.toggle_relief()` | Chuyển tiếp toggle F8/F7/F10 xuống content plane |
| `.show_toast(msg)` | Hiện thông báo ngắn |
| `.layer_grade_lines` | Dòng trạng thái từng lớp cho HUD |
| `.release()` | Giải phóng toàn bộ tài nguyên GPU |

### `content_plane.py` — `ContentPlaneRenderer` (**file lớn nhất, ~800 dòng**)

| Hàm | Công dụng |
|---|---|
| `__init__(shader_dir, content, root)` | Tạo `DepthWorker`, mesh phẳng, shader; nạp media ban đầu, có dự phòng procedural |
| `.initialize()` | Tạo texture GPU sau khi context GL sẵn sàng; nạp shader `video_rgba` và `mist`; khởi động `DepthWorker` |
| `.load_file(path, strict)` | Đổi ảnh/video đang hiển thị, trả `(thành công, thông báo)` |
| `.update(dt_s)` | Mỗi frame: lấy frame video mới (nếu là video) và **kiểm tra depth đã xong chưa** |
| `.draw(view, proj)` | Vẽ: nếu có lớp thì gọi `_draw_layers`, không thì vẽ quad phẳng |
| `._draw_layers(...)` | Vẽ các lớp **xa → gần**, bật blend, **tắt ghi depth** để biên mềm không đục lỗ lớp gần hơn; chèn quad sương mù sau lớp xa nhất |
| `._prime_and_upload(reason)` | Lấy frame đầu, nạp texture, và yêu cầu tính depth |
| `._request_depth(rgb)` | Gửi ảnh cho `DepthWorker` (bỏ qua nếu đã có cache) |
| `._poll_depth_mesh()` | Kiểm tra depth đã sẵn sàng → gọi `_apply_layers` (Mode B) hoặc `_apply_depth_mesh` (Mode A). **PB-076**: Mode B **hoãn đúng một khung** — khung đầu chỉ đặt toast `LAYER_BUILD_TOAST`, khung sau mới dựng, nên lời nhắn nằm trên màn suốt lúc đóng băng |
| `.take_build_notice()` | **PB-076**: lấy và xoá lời nhắn của khung hoãn; `SceneRenderer.render` gọi ngay sau `update` nên toast lên đúng khung cuối trước lúc đóng băng |
| `._apply_depth_mesh(depth, key)` | **Mode A**: dựng một lưới 96×96 dịch chuyển từng đỉnh theo depth |
| `._apply_layers(depth, key)` | **Mode B — hàm điều phối chính**: chọn K → phân cụm → phân loại đường nối → dựng lưới/texture từng lớp → tạo sương mù → log bảng thời gian nạp |
| `._create_rgba_texture(rgba)` | Tạo texture RGBA (có `flipud` vì OpenGL gốc toạ độ dưới-trái) |
| `._release_layers()` / `._release_mist()` | Giải phóng texture và mesh của lớp / sương mù |
| `._mist_quad_size(z)` | Tính kích thước quad sương mù theo tỉ lệ độ sâu và `mist_margin` |
| `._build_mist_renderable(...)` / `._draw_mist(...)` | Dựng / vẽ quad sương mù bán trong suốt |
| `._build_layer_shadows()` | **PB-073**: dựng texture bóng một lần cho mỗi ảnh, nằm sau đúng một cổng `if layer_shadow` |
| `layer_geometry(...)` | Nguồn sự thật duy nhất cho hình học quad. **PB-034**: `surround == "box"` thì hệ số phóng lấy từ `box_layer_scale`, **vẫn dời tâm theo mắt thiết kế** như đường cũ |
| `._edge_params()` / `._semantic_params()` / `._staging_params()` | Đóng gói `ContentSettings` thành các dataclass tham số cho pipeline |
| `.toggle_atmospheric()` | Bật/tắt phối màu và **dựng lại lớp từ depth đã cache** (không suy luận depth lại) — dùng để chụp ảnh so sánh A/B |
| `.toggle_mist()` / `.toggle_relief()` | Bật/tắt sương mù / relief (relief phải dựng lại lưới) |
| `.layer_grade_lines` | Sinh dòng trạng thái: trạng thái A/B, từng lớp, báo cáo semantic, báo cáo adaptive K |
| `._swap_source(path, kind)` | Đổi nguồn media (ImageSource ↔ VideoPlayer) |
| `._resize_plane(aspect)` | Tính lại kích thước mặt phẳng theo tỉ lệ ảnh |
| `._upload_rgb(rgb)` | Nạp ảnh RGB lên texture (`glTexImage2D` lần đầu, `glTexSubImage2D` các lần sau) |
| `._build_flat_mesh` / `._invalidate_texture` | Tiện ích dựng mesh phẳng / đánh dấu texture cần nạp lại |
| `.release()` | Dừng depth worker, đóng nguồn media, xoá texture và shader |

### `debug_overlay.py`, `debug_geometry.py`, `webcam_preview.py`, `toast_overlay.py`

| Lớp / Hàm | Công dụng |
|---|---|
| `DebugMetrics` | Dataclass gom mọi số liệu hiển thị trên HUD F1 |
| `DebugOverlay.initialize()` | Thử ImGui; không có thì dựng HUD dự phòng bằng quad + texture OpenCV |
| `.toggle()` / `.draw(metrics)` | Bật/tắt (F1) / vẽ HUD |
| `._draw_imgui` / `._draw_fallback` | Hai đường vẽ HUD |
| `._format_lines(metrics)` | Định dạng các dòng text ⚠️ **có code chết sau `return`** — xem Phần V |
| `DebugGeometryRenderer.upload_segments(segs)` | Nạp các đoạn thẳng thế giới (khung frustum) |
| `.draw(view, proj)` | Vẽ đoạn thẳng bằng shader line |
| `WebcamPreviewSettings` / `TrackingPreviewSnapshot` | Cấu hình bố cục PiP / dữ liệu landmark an toàn luồng |
| `WebcamPreviewOverlay.draw(...)` | Vẽ khung webcam nhỏ ở góc (F4) |
| `._prepare_preview_image(...)` | Thu nhỏ, lật gương ảnh preview |
| `._draw_landmarks(...)` | Vẽ chấm landmark và mống mắt lên preview |
| `._draw_quad(...)` / `._upload_texture(...)` | Vẽ quad màn hình / nạp texture |
| `_preview_rect_ndc(...)` | Tính hình chữ nhật NDC theo góc và tỉ lệ đã cấu hình |
| `ToastOverlay.show(msg, is_error)` | Đặt thông báo hiển thị ~2 giây |
| `.draw(w, h)` | Vẽ toast ở giữa-trên, tự ẩn khi hết giờ |
| `._render_texture(is_error)` | Vẽ text ra texture bằng OpenCV |

### `video_player.py`, `gpu_camera.py`, `obj_loader.py`

| Hàm | Công dụng |
|---|---|
| `VideoPlayer.__init__(path, loop)` | Mở video; nếu hỏng thì chuyển sang chế độ procedural |
| `.prime_first_frame()` | Giải mã và trả frame đầu ngay (không cần dt) |
| `.tick(dt_s)` | Tiến playback theo thời gian thực, trả frame RGB mới nhất |
| `._read_next()` | Đọc frame tiếp; tua về đầu nếu bật `loop` |
| `._procedural_frame()` | Sinh dải màu động khi không có file video |
| `.replace(path, loop)` | Đóng decoder cũ, mở file mới |
| `.decoded_frames` | Số khung đã giải mã — **chỉ để đo** (PB-059 (3)). Bộ đếm chỉ chạy khi đặt `HCPD_VIDEO_STATS=1`; mặc định tắt thì không thêm một phép tính hay một dòng log nào. Cùng lối móc sau biến môi trường như PB-058 |
| `GpuCameraBuffer.initialize()` | Tạo Uniform Buffer Object cho ma trận camera |
| `.update(state)` | Nạp trạng thái camera lên UBO |
| `.bind_block(program, name)` | Gắn UBO với uniform block của shader |
| `._write_cpu_cache(state)` / `._pack_matrix(...)` | Đóng gói theo layout std140 |
| `load_obj(path)` | Đọc file OBJ mặt tam giác thành `MeshData` |
| `load_obj_mesh(path)` | Đọc OBJ và nạp thẳng lên GPU |

---

## 3.10. `src/scene/` — Đồ hoạ cảnh

### `node.py` — Scene graph

| Lớp / Hàm | Công dụng |
|---|---|
| `Material` | Màu và độ bóng Phong |
| `SceneNode` | Nút biến đổi: tên, mesh, vật liệu, tịnh tiến, xoay quanh Y, co giãn, các nút con |
| `.world_matrix(parent)` | Tính ma trận thế giới tích luỹ |
| `.traverse(parent)` | Generator duyệt sâu, sinh cặp `(node, world_matrix)` |

### `primitives.py` — Sinh hình học thủ tục

| Hàm | Công dụng |
|---|---|
| `create_plane(w, d, y)` | Mặt phẳng ngang trên mặt XZ |
| `create_box(w, h, d)` | Hộp vuông góc quanh gốc |
| `create_sphere(r, segments, rings)` | Hình cầu UV |
| `create_vertical_plane(w, h, z)` | Quad đứng trong XY hướng +Z (có pháp tuyến, cho Phong) |
| `create_vertical_plane_at(cx, cy, w, h, z)` | Quad đứng đặt tại tâm chỉ định |
| `create_window_frame_bezel(iw, ih, b, z)` | Khung viền rỗng (4 thanh) quanh ô nội dung |
| `create_open_box(w, h, z_front, z_back)` | **PB-034**: hộp **năm mặt** (sàn, trần, hai vách, vách sau), mở về phía người xem, pháp tuyến hướng **vào trong**. Mặt trước cố ý không có — đó là miệng hộp, phải trùng đúng hình chữ nhật màn hình |
| `create_vertical_plane_uv(w, h, z)` | Quad đứng **có UV** cho texture video |
| `create_displaced_vertical_plane_uv(...)` | **Mode A**: lưới chia nhỏ, mỗi đỉnh dịch Z theo `depth × depth_scale` |
| `create_layer_relief_vertical_plane_uv(...)` | **Relief P1**: lưới chia nhỏ cho một lớp, phồng Z **quanh độ sâu trung bình của lớp**, làm mượt Gaussian rồi **kẹp ở ±0.4·khoảng cách giữa lớp** để đỉnh không xuyên qua lớp kề. Log hai dòng chẩn đoán: biên độ mm thực tế, đỉnh trước/sau làm mượt |
| `sample_depth_map(depth, u, v)` | Lấy mẫu song tuyến depth map tại UV (v=0 là đáy) |
| `sample_mask_map(mask, u, v)` | Lấy mẫu song tuyến mask lớp tại UV |
| `_pack(positions, normals, indices)` | Đóng gói thành `MeshData` |
| `_merge_mesh_parts(parts)` | Gộp nhiều phần mesh thành một draw call |
| `create_mesh(data)` | Nạp `MeshData` lên GPU |

Công thức relief:
```
delta  = local_depth_norm − mean_norm
relief = delta · (relief_amplitude · gap) · relief_depth_scale(depth_norm) · mask_weight
       → GaussianBlur(relief_smooth_sigma)
       → clip(±0.4 · gap)
```

### `room.py`, `video_scene.py`

| Hàm | Công dụng |
|---|---|
| `build_room_scene()` | Dựng phòng 3D có sàn, tường, đồ nội thất (hộp/cầu) — chế độ demo thị sai |
| `build_video_window_scene(content)` | Dựng "phòng rỗng": backdrop Z=−2.8, hai tấm giữa Z=−1.5, khung viền Z=−0.58. Mặt phẳng nội dung Z=−0.75 do `ContentPlaneRenderer` vẽ riêng |
| `farthest_content_z_m(content)` | Z của **mặt nội dung xa nhất** = `plane_position_m.z + layer_z_far_m` (tắt tách lớp thì chỉ là plane Z) |
| `behind_content_z_m(z, content)` | Đẩy khung viền / tấm giữa / tấm nền ra sau lớp xa nhất 0,05 m, **giữ nguyên** giá trị nào đã đủ sâu. Thiếu nó thì `layer_z_far_m` = −0,60 làm khung viền −0,58 che mất ảnh (chữ U đen, 31/10) |

### `window_box.py` — hộp quanh các lớp (PB-034, thêm 2026-10-30)

| Hàm | Công dụng |
|---|---|
| `build_scene_for(content, screen_w_m, screen_h_m)` | **Một chỗ quyết định duy nhất** cho cả `app.py` lẫn `renderer.py`: `surround == "box"` thì dựng hộp, không thì gọi `build_video_window_scene` y như cũ |
| `build_window_box_scene(...)` | Hộp mở năm mặt, miệng ở z = 0 đúng kích thước màn, màu vách lấy `backdrop_color` |
| `box_cross_section_m(w, h, aspect)` | Bề rộng quad lớp để phủ kín tiết diện hộp: `max(W, H × tỉ lệ ảnh)` |
| `back_wall_z_m(content)` | Z vách sau = `layer_z_far_m − 0.05 m` |
| `box_layer_scale(t, ...)` | Hệ số phóng quad ở chế độ hộp = **max của hai ràng buộc**: khớp phối cảnh `(Dd + t)/Dd`, và phủ kín lòng ống hộp **sau khi đã dời tâm** theo mắt thiết kế (`W/2 + abs(ox)`, `H/2 + abs(oy)`). Bỏ ràng buộc thứ hai thì lớp xa hụt 2,3 mm trên demo; bỏ việc dời tâm thì ảnh rách dọc |

---

## 3.11. `src/content/` — Độ sâu và tách lớp (vùng nghiên cứu chính)

### `depth_estimator.py`

| Lớp / Hàm | Công dụng |
|---|---|
| `normalize_depth_map(raw, h, w)` | Chuẩn hoá về [0,1] (**1 = gần**) bằng cắt phân vị 2–98% để vài pixel ngoại lai không nén hết tương phản, rồi resize |
| `DepthEstimator.__init__(model_name, backend)` | Lưu tên model và engine (`"torch"` mặc định / `"onnxruntime"`), **chưa nạp** (lazy) |
| `.estimate(rgb)` | Trả depth map float32 HxW trong [0,1] |
| `._ensure_loaded()` | Nạp model lần đầu; thất bại thì ghi nhớ lỗi và trả False mãi mãi |
| `._create_backend()` | Chọn backend theo tên model + `depth_backend`; ORT lỗi/thiếu file .onnx thì rơi về torch, Depth Anything lỗi thì rơi về MiDaS |
| `_MiDaSBackend` | Nạp MiDaS small qua `torch.hub` |
| `_DepthAnythingBackend` | Nạp Depth Anything Small qua `transformers` |
| `_OnnxRuntimeBackend` | Chạy Depth Anything Small bằng ONNX Runtime CPU từ file `artifacts/pb060_onnx/models/da_v1_small_fp32.onnx` (PB-060); cùng processor HF, 2 intra / 1 inter thread |
| `_DepthBackend` | Protocol định nghĩa giao diện `estimate()` |
| `ONNX_DEPTH_RELPATH` | Đường dẫn file .onnx so với gốc repo; file trong `artifacts/` nên gitignored, thiếu thì rơi về torch |

### `depth_worker.py`

| Hàm | Công dụng |
|---|---|
| `DepthWorker.start()/.stop()` | Khởi động/dừng luồng suy luận nền |
| `.submit(media_key, rgb)` | Xếp hàng một ảnh (bỏ qua nếu đã cache hoặc đang xử lý) |
| `.pop_ready(media_key)` | Lấy depth map đã tính xong |
| `.has_ready(media_key)` | Kiểm tra đã có kết quả chưa |
| `.media_key(path)` | Sinh khoá cache từ đường dẫn tuyệt đối |
| `.last_inference_ms_for(key)` | Thời gian suy luận của lần gần nhất (cho bảng log) |
| `._loop()` | Vòng lặp nền: lấy việc → chạy model → lưu kết quả + thời gian |
| `._take_pending()` | Lấy việc đang chờ (an toàn luồng) |

### `adaptive_k.py` — Chọn số lớp tự động

| Hằng / Lớp / Hàm | Công dụng |
|---|---|
| `GVF_THRESHOLD` = 0.90 | Ngưỡng quyết định |
| `_MAX_SAMPLE_PIXELS` = 10_000 | Số mẫu depth tối đa để chấm điểm |
| `_N_INIT` = 4 | Số lần khởi động lại k-means |
| `LOG_SILHOUETTE_COMPARISON` = False | Cổng bật silhouette (chỉ để so sánh, mặc định tắt) |
| `AdaptiveKReport` | Báo cáo: K đã chọn, `gvf_variance` từng K, ngưỡng, danh sách lý do loại, số mẫu, số lần fit, thời gian |
| `.gvf` | Bí danh của `gvf_variance` (tương thích ngược cho HUD) |
| `_KMeansFitCounter` | Bộ đếm số lần gọi k-means trong một lần chọn |
| `_downsample_depth_values(depth)` | Giảm còn ~10k mẫu bằng `cv2.resize` INTER_AREA |
| `_assign_labels_1d(values, centroids)` | Gán mỗi giá trị về tâm gần nhất |
| `_wcss(values, centroids, labels)` | Tổng bình phương sai số trong cụm |
| `_compute_gvf_variance(...)` | **GVF thật** = `1 − WCSS/TSS` |
| `_best_gvf_fit(...)` | Lấy GVF tốt nhất qua `n_init` lần khởi động lại |
| `_fit_kmeans(...)` | Gọi `_kmeans_1d` và đếm |
| `_assert_gvf_variance_monotonic(gvf)` | **Chỉ ghi log cảnh báo** nếu GVF không đơn điệu tăng theo K — không kẹp giá trị |
| `_choose_k_gvf_threshold(...)` | **Luật Jenks**: chọn K nhỏ nhất có `gvf ≥ threshold`, không có thì `k_max` |
| `_silhouette_1d(values, labels)` | Hệ số silhouette (O(n²)) — đã **gỡ khỏi đường nóng**, chỉ chạy khi bật cổng |
| `_log_adaptive_k(report)` | In bảng kết quả và lý do loại |
| `select_optimal_k(depth_map, ...)` | **Hàm chính**. Có nhánh dự phòng khi depth phẳng (std < 1e-4 hoặc TSS ~ 0) → trả `k_min`. ⚠️ **Bỏ qua 5 tham số** — xem Phần V |

**Lịch sử:** luật cũ dùng silhouette, tốn **40–111 giây/ảnh** khiến app treo
("Not Responding"). Số liệu đo được lúc chuyển sang luật GVF:

| Ảnh | K=2 | K=3 | K=4 | → chọn ở 0.90 |
|---|---|---|---|---|
| t15 | 0.844 | 0.937 | 0.969 | K=3 |
| t14 | 0.875 | 0.950 | 0.971 | K=3 |
| t5 | 0.882 | 0.947 | 0.978 | K=3 |
| t1 | 0.924 | 0.968 | 0.985 | K=2 |

### `semantic_depth.py` — Điều chỉnh độ sâu theo ngữ nghĩa

**Vấn đề:** depth đơn ảnh đặt các vật thể khác nhau về ngữ nghĩa (bầu trời vs
ngôi làng) ở độ sâu gần như nhau → k-means gộp chúng thành một lớp phẳng → mất
thị sai. **Cách làm** (theo 3D Ken Burns 2019, SLIDE 2021): phân đoạn bằng
OpenCV thuần CPU, không cần huấn luyện, không cần SAM.

| Lớp / Hàm | Công dụng |
|---|---|
| `SemanticDepthParams` | Tham số: số dải, ngưỡng tương đồng, độ dời tách, phương sai tối đa, kích thước segment tối thiểu, giới hạn kích thước làm việc |
| `SemanticDepthReport` | Báo cáo: số segment, số bị làm phẳng, số cặp bị tách, dải depth trước/sau |
| `_SegmentInfo` | Thông tin một segment: median, phương sai, trọng tâm Y, số pixel |
| `_resize_work(rgb, depth, max_dim)` | Thu nhỏ về cạnh dài ≤ 640 để phân đoạn nhanh (nhãn được phóng lại sau) |
| `_combined_gradient(rgb, depth)` | Gradient tổ hợp `0.72·Sobel(RGB) + 0.28·Sobel(depth)` — làm "rào chắn" cho watershed |
| `_depth_band_labels(depth, n)` | Chia depth thành các dải theo phân vị |
| `_watershed_subdivide(...)` | Tách blob lớn bằng watershed; nếu không tìm được marker thì rơi về chia theo phân vị hàng (trên/giữa/dưới) |
| `segment_image(rgb, depth, params)` | **Phân đoạn**: dải depth → connected components → watershed cho blob lớn |
| `_merge_small_segments(segments, min_px)` | Gộp segment quá nhỏ vào nhãn kề lớn nhất |
| `_spatial_depth_grid_segments(depth, min_px)` | Dự phòng: lưới depth × vị trí dọc khi watershed chỉ ra một blob |
| `_collect_segments(depth, segments)` | Thu thập thống kê từng segment |
| `_adjacent_segment_pairs(segments)` | Tìm các cặp segment kề nhau (dịch mảng 1 pixel theo 2 hướng) |
| `_flatten_segments(depth, segments, max_var)` | **Làm phẳng**: segment phương sai thấp bị ép về median (giữ mặt + thân người trên một mặt phẳng) |
| `_separate_similar_adjacent_segments(...)` | **Tách**: dùng union-find gom các segment kề nhau cùng độ sâu nhưng khác vị trí Y, rồi rải chúng ra theo `separation_offset`. Sắp xếp theo trọng tâm Y (trên = xa hơn, theo ngữ nghĩa phong cảnh: trời > làng > đường) |
| `adjust_depth_semantic(rgb, depth, params)` | **Hàm chính**, chạy 3 bước trên và trả `(depth đã chỉnh, báo cáo)` |

### `layer_vote.py` — "Lớp đệm": bầu nhãn theo vùng (PB-070b / SPIKE-15)

Thêm 2026-10-29. Copy từ `D:\seg_lab\seg_lab\vote.py` @ `2058b3b`, **không
import**. Bật bằng key `layer_method = "kmeans_vote"`; mặc định `"kmeans"` thì
không hàm nào ở đây được gọi.

| Hàm / hằng | Công dụng |
|---|---|
| `TAU_F` = 0.6 | Tỷ lệ tối thiểu của nhãn đa số trong một vùng để được dồn cả vùng |
| `MIN_PX` = 64 | Vùng nhỏ hơn số điểm này bị bỏ qua |
| `centroid_gaps(centroids)` | Khoảng cách từ mỗi tâm cụm tới tâm gần nhất khác nó; **K = 1 trả vô cực** (không có lớp cạnh tranh nên cửa `s ≤ g` không được chặn) |
| `vote_labels(labels, centroids, depth, masks)` | `(labels mới, relabel_frac)`. Xét vùng **lớn → nhỏ**; đổi cả vùng sang nhãn đa số khi `f* ≥ 0.6` **và** `s ≤ g`. `ℓ*` luôn tính trên nhãn k-means **gốc**, không phải nhãn đã bị vùng trước sửa |
| `vote_on_regions(labels, centroids, depth, rgb)` | Bọc: lấy vùng P10 rồi bầu; lỗi ở bước vùng thì **giữ nguyên** nhãn k-means và chỉ ghi cảnh báo |

**Không bao giờ cụm lại** sau khi bầu — tâm cụm giữ nguyên của k-means. Đó là
điều kiện làm nên kết quả SPIKE-15: P2/P5/P10 thay depth rồi cụm lại nên mất
4–6 điểm thứ tự che khuất, còn cách này chỉ mất 0,97 điểm.

### `fog_complement.py` — Xám lớp xa bằng màu bù (PB-071)

Thêm 2026-10-30. Chép (không import) công thức từ `scripts/_fog_complement.py`
@ `188df40`. Bật bằng key `far_fog_mode = "complement"`; mặc định `"atmo"` thì
không hàm nào ở đây được gọi và đầu ra giống hôm nay **từng byte**.

Công thức, trộn trong linear RGB: `c' = (max(c) + min(c)) − c` rồi
`out = c^(1−t)·c'^t` (trung bình **hình học** ≈ trộn bột màu, vừa xám vừa tối).

| Hàm / hằng | Công dụng |
|---|---|
| `T_MAX` = 0.5 | Trần của `t`. Bão hoà theo `t` có dạng **chữ U** — `t = 1` là màu bù nguyên chất, mà bù của màu rực thì cũng rực (đỏ → lơ) — nên chỉ nửa `[0, 0.5]` đơn điệu giảm và dò được bằng chia đôi |
| `BISECT_STEPS` = 24 | Số bước chia đôi |
| `SAMPLE_CAP` = 60 000 | Số điểm tối đa đem đi chia đôi; lớp đông hơn thì lấy thưa đều. Bản đo ở `scripts/` trộn **cả ảnh** mỗi bước (16,4 s một lớp ở 900×1200); ở đây chỉ trộn điểm của lớp |
| `complement_mix(rgb, t)` | Trộn ảnh với màu bù của chính nó |
| `mean_saturation(rgb, mask)` | Bão hoà trung bình (HSV của OpenCV) trên mask |
| `solve_t_for_saturation(rgb, mask, target)` | Chia đôi tìm `t` cho bão hoà bằng `target`; trả đúng `T_MAX` khi ngay điểm xám nhất vẫn còn rực hơn đích |
| `solve_t_for_layer(base, atmo_out, mask, cap)` | Bọc: lấy mẫu thưa, đo đích trên **đúng tập mẫu đó**, rồi gọi hàm trên |

### `lama_fill.py` — lấp khe bằng big-lama ONNX (PB-074 Phase 5b)

| Hàm | Công dụng |
|---|---|
| `inpaint_hole(rgb, hole, radius, method)` (ở `disocclusion.py`) | Một chỗ chọn thuật toán lấp: `telea` · `fsr` · `shiftmap` (PB-075, chạy trong **Lab**, mask **ngược** với `cv2.inpaint`) · `lama` |
| `inpaint_lama(rgb, hole, use_cache=True)` | Lấp một lớp bằng **một lượt suy luận** 512×512; cache ra đĩa theo băm nội dung (`artifacts/lama_cache/`). Trả `None` nếu không chạy được → bên gọi dùng Telea |
| `_get_session()` | Phiên `onnxruntime` dùng chung, dựng **đúng một lần** (~42 s cho model 208 MB; kiểm hai lần quanh `threading.Lock` — thiếu khoá thì luồng nạp nền và luồng GL cùng dựng **hai** phiên, PB-076 đo 80,0 s + 63,3 s) |
| `preload_session_async()` | **PB-076**: nạp phiên ở luồng nền **ưu tiên thấp nhất**, gọi **sau khi khung đầu đã lên màn** (`app.py`, chỉ khi có khoá đang để `lama`). Đo được: suốt 81,9 s nạp, FPS vẫn **40,00** trung vị |
| `_fit_into_tile(rgb, hole)` | Thu ảnh cỡ làm việc vào ô 512 **giữ tỉ lệ**, đệm `BORDER_REPLICATE` (không phải đen: lớp FFC đọc toàn ảnh làm ngữ cảnh) |
| `_run(tile_rgb, tile_hole)` | Chuẩn hoá /255, nhân `(1 − mask)` rồi chạy; đầu ra đã ở thang 0–255 |

Model **không nằm trong repo** (208 MB, đã vào `.gitignore`). Nguồn, sha256 và
giấy phép ghi ở `task_done\PB-074_look3d\README.md` mục Phase 5b.

### `layer_regions.py` — Bản đồ vùng P10 (RGB-D Felzenszwalb)

Thêm 2026-10-29, copy phần P10 từ `scripts/seg_regions.py` + `seg_lab/regions_c.py`
(nhánh `spike11b-seg` @ `f834a41`). Chỉ dùng cho bước bầu.

| Hàm / hằng | Công dụng |
|---|---|
| `WORK_MAX_DIM` = 640 | Cạnh dài của ảnh làm việc; vùng tính ở cỡ này |
| `P10_LAMBDA` = 3.0 | Trọng số kênh depth trong `[L, a, b, λ·D]`, **đóng băng** |
| `SegParams` / `FROZEN` | sigma 0.6, k 220, min_size 180, alpha 1.0, beta 0.6, gamma 0.8, tau_p2 0.35 — copy y nguyên giá trị đã dò ở SPIKE-11 |
| `work_image(rgb, max_dim)` | Thu ảnh về cạnh dài ≤ `max_dim` |
| `small_pair(rgb, depth)` | `(rgb thu nhỏ, depth thu nhỏ cùng cỡ)` |
| `p10_labels(small, dep)` | Felzenszwalb trên ảnh 4 kênh `[L, a, b, λ·D]` |
| `region_masks(rgb, depth)` | Mask từng vùng ở cỡ ảnh **gốc**; phóng nhãn bằng `INTER_NEAREST` nên vùng trùng đúng vùng P10 đã được chấm trên 765 ảnh |

### `region_merge.py` — Gộp vùng tham lam

Copy y nguyên `scripts/_seg_merge.py` @ `f834a41`. Gộp cặp vùng kề rẻ nhất theo
`C = α|Δd| + β‖Δc‖ + γE`. `relabel`, `region_pixels`, `merge_regions` là ba hàm
`layer_regions.py` dùng; phần còn lại là nội bộ.

### `layer_clustering.py` — Tách lớp và phối màu (**840 dòng**)

| Hằng số | Ý nghĩa |
|---|---|
| `ATMOSPHERE_FALLBACK` | Xám xanh nhạt #B8C4D0, dùng khi lớp xa nhất quá tối |
| `DARK_LUMA_THRESHOLD` = 25.0 | Ngưỡng luma coi là "quá tối" |
| `_MAX_FIT_SAMPLES` = 40_000 | Số mẫu tối đa cho k-means |

| Lớp | Công dụng |
|---|---|
| `LayerEdgeParams` | Viền sáng, độ rộng feather biên, color bleed, ngưỡng gradient EDGE |
| `LayerStagingParams` | Bật/tắt, pha khí quyển tối đa, giảm bão hoà lớp xa, tăng/giảm tương phản |
| `LayerGradeReport` | Ghi lại chính xác grade đã áp cho một lớp (cho HUD F1) |
| `DepthLayer` | Một lớp: chỉ số, depth trung bình/lo/hi, Z thế giới, độ phủ, mask, texture RGBA, `depth_norm`, báo cáo grade |
| `ClusterIntoLayersResult` | Kết quả: danh sách lớp, báo cáo semantic, màu sương mù, bảng thời gian |

| Hàm | Công dụng |
|---|---|
| `_kmeans_1d(values, k, seed, iters)` | k-means 1-D **tất định** (seed k-means++ rồi lặp Lloyd), trả tâm đã sắp xếp |
| `cluster_depth_layers(depth, n, seed)` | Phân cụm depth map, trả `(nhãn HxW, tâm)`; lấy mẫu nếu ảnh quá lớn |
| `classify_silhouette_edges(mask, rgb, threshold)` | Đánh dấu biên tương phản cao (cây, người vs trời) để **chỉ nới rộng feather ở chỗ cắt lộ liễu** |
| `_silhouette_band(mask, width)` | Dải pixel quanh biên mask (cả trong lẫn ngoài) |
| `feather_mask_alpha(mask, dilate, blur, edge_mask, edge_blur)` | Biến mask nhị phân thành kênh alpha mềm: dilate lấp lỗ nhỏ → Gaussian blur → thay bằng bản blur rộng hơn ở dải biên EDGE. **Đây là thứ xoá đường ghép đen giữa các lớp** |
| `apply_color_bleed(rgb, alpha, behind, px)` | Pha RGB tiền cảnh về phía lớp phía sau trong vùng alpha mềm, tránh gãy màu đột ngột ở biên |
| `extract_layer_rgba(rgb, mask, ...)` | Dựng texture RGBA HxWx4. **RGB giữ ở mọi pixel** (không xoá ngoài cụm) để biên mềm pha đúng màu thật |
| `estimate_atmosphere_color(rgb, mask, ...)` | Lấy màu khí quyển = RGB trung bình lớp xa nhất; quá tối thì dùng fallback |
| `_srgb_to_linear(x)` / `_linear_to_srgb(x)` | Giải mã/mã hoá sRGB — grading chạy trong ánh sáng tuyến tính để giống sương mù thật |
| `staging_factors(depth_norm, params)` | Ánh xạ độ xa → `(pha khí quyển, bão hoà, tương phản)`. Neo sao cho **lớp giữa trung tính**: gần thì tăng tương phản, xa thì giảm tương phản + giảm bão hoà |
| `stage_layer_color(rgb, depth_norm, atmo, params, ..., mask)` | **Grade một lớp** theo thứ tự: tương phản quanh pivot → pha khí quyển (tuyến tính) → giảm bão hoà (HSV). Trả ảnh + `LayerGradeReport`. `enabled=False` là nhánh "OFF" của so sánh A/B. PB-071: khi `params.far_fog_mode = "complement"` thì hai bước sau được thay bằng `complement_fog()`; `mask` chỉ dùng ở đó |
| `complement_fog(lin_contrast, atmo_out, mask)` | PB-071: xám hoá bằng **màu bù** thay vì tấm xám. `atmo_out` chỉ làm **đích bão hoà**. Trả `(ảnh, t)` |
| `_layer_luminance_pivot(rgb, mask)` | Luma tuyến tính trung bình của pixel thuộc lớp — dùng làm pivot tương phản |
| `apply_atmospheric_perspective(...)` | Đường **cũ** (legacy) khi không truyền `staging` |
| `add_rim_light(rgb, mask, width, intensity, color)` | Làm sáng dải mỏng bên trong biên — chỉ dùng cho lớp tiền cảnh |
| `compute_mist_color(far_layer)` | Màu sương mù = RGB trung bình lớp xa nhất, làm sáng 15% về trắng |
| `layer_feather_scales(depth_norm, base_blur, edge_px)` | Nội suy bán kính feather theo độ xa (lớp xa mềm hơn một chút) |
| `inter_layer_gap_m(z_near, z_far, count)` | Khoảng cách đều giữa các mặt phẳng lớp |
| `assign_layer_z(rank, count, z_near, z_far)` | Ánh xạ thứ hạng xa→gần thành Z thế giới **cách đều**, đảm bảo lớp luôn tách bạch dù cụm depth có xúm lại |
| `relief_depth_scale(depth_norm, far_falloff)` | Hệ số relief theo độ xa: `1 − depth_norm·(1−far_falloff)` |
| `cluster_into_layers(rgb, depth, n_clusters, ..., layer_method)` | **Hàm điều phối chính**: semantic → k-means → (PB-070b: bầu theo vùng nếu `layer_method = "kmeans_vote"`, **không cụm lại**) → với từng lớp (grade → viền sáng → feather+RGBA+bleed → cập nhật ảnh composite) → tính màu sương mù. Trả danh sách **xa → gần** để vẽ trực tiếp |

### `disocclusion.py` — Lấp khe hở khi nghiêng đầu (C1)

Tách lớp cứng, không hàn → nghiêng đầu làm lớp gần trượt qua lớp xa và **lộ ra
vùng vốn bị che**, nơi ảnh gốc không có pixel nào. Không xử lý thì thấy xuyên
qua tới màu void. Theo Shih et al., *3D Photography using Context-aware Layered
Depth Inpainting* (CVPR 2020), bản nhẹ: Telea inpaint trên dải có giới hạn,
không cần model học sâu, chạy một lần lúc nạp ảnh.

| Hàm | Công dụng |
|---|---|
| `DisocclusionParams` | `enabled`, `fill_px` (64), `inpaint_radius` (5), `feather_px` (8), `max_work_dim` (640) |
| `DisocclusionReport` | Số pixel dải, số pixel nguồn, thời gian — để ghi log |
| `max_reveal_px(...)` | Tính **từ quang học** ra số pixel texture bị lộ: chiếu hai mặt phẳng lớp lên màn hình, lấy chênh lệch thị sai, quy về pixel của lớp xa. Dùng để chọn `fill_px` thay vì đoán. Ở cấu hình hiện tại: **38 px cho mỗi 100 mm nghiêng đầu** |
| `disocclusion_band(own, nearer, fill_px)` | Dải duy nhất có thể bị lộ. Dùng **distance transform** (`DIST_MASK_5`) chứ không `cv2.dilate`: OpenCV không phân rã kernel ellipse nên ở `fill_px=64` nó tích chập phần tử 129×129 và **một mình chiếm ~2 giây** trên texture 2912×1632; distance transform còn 71 ms |
| `_band_alpha(band, feather_px)` | Alpha đục trong dải, vuốt xuống ở rìa ngoài để không kết thúc bằng nhát cắt |
| `fill_layer_disocclusion(...)` | Hàm chính. **Mask inpaint = toàn bộ vùng bị lớp gần che**, không chỉ dải — đó là điểm C3: `extract_layer_rgba` giữ RGB ở mọi pixel nên màu tiền cảnh vẫn nằm trong mảng, nếu không che hết thì Telea sẽ hút chính nó vào lấp khe |

Chi phí đo được: +897 ms cho t15 (K=3), inpaint ở `max_work_dim=640` thay vì 960
tiết kiệm gần một nửa với sai lệch 4.35/255 trong dải — không nhìn thấy được.

### `boundary_classifier.py` — Phân loại đường nối (Phase 2a)

Với mỗi pixel trên đường nối giữa hai lớp, sinh trọng số hàn `w ∈ [0,1]`:
`w = 1` → hàn Z hoàn toàn (chuyển tiếp mượt), `w = 0` → bậc cứng (che khuất).
**Gradient depth quyết định**; biên RGB chỉ *xác nhận* bậc cứng khi depth đã có
đứt gãy. Dùng depth **gốc** (trước semantic).

| Hằng số | Giá trị |
|---|---|
| `DEPTH_EDGE_LOW / HIGH` | 0.05 / 0.20 (tuyệt đối, không co giãn theo ảnh → so sánh được giữa các ảnh) |
| `RGB_EDGE_THRESHOLD` | 0.04 |

| Lớp / Hàm | Công dụng |
|---|---|
| `SeamStats` | Thống kê một đường nối: hai lớp, độ dài px, tỉ lệ cứng/mềm/pha, `w` trung bình |
| `BoundaryClassifierResult` | Bản đồ `weld_weight` (NaN ngoài đường nối), gradient depth, danh sách `SeamStats` |
| `depth_gradient_magnitude(depth)` | Độ lớn Sobel tuyệt đối của depth |
| `weld_weight_from_depth_grad(g, low, high)` | Ánh xạ gradient → trọng số hàn (dốc tuyến tính giữa low và high) |
| `_rgb_edge_map(rgb, threshold)` | Mask biên Sobel toàn ảnh (dự phòng/test) |
| `_seam_mask(mask_a, mask_b)` | Pixel của A∪B chạm vào lớp kia (dilate 3×3) |
| `_seam_stats(a, b, seam, weld)` | Tổng hợp thống kê một đường nối |
| `classify_layer_seams(depth, masks, rgb, ...)` | **Hàm chính**: duyệt mọi cặp lớp, tính trọng số hàn, ghi log từng đường nối |
| `render_weldmap(rgb, result, alpha)` | Vẽ overlay gỡ lỗi: **đỏ = cứng, xanh lá = hàn mềm, vàng = vùng dốc** |

> ⚠️ **Chưa hàn gì cả.** Kết quả chỉ được lưu vào
> `ContentPlaneRenderer._boundary_report` và ghi log. Áp trọng số vào hình học
> là công việc của Phase 2b.

### `image_source.py`, `formats.py`, `cv_io.py`, `persistence.py`

| Hàm | Công dụng |
|---|---|
| `ImageSource.__init__(path)` | Nạp một ảnh tĩnh, tính tỉ lệ khung |
| `.tick(dt)` | Trả cùng một frame RGB mỗi lần gọi (đồng nhất API với video) |
| `._load(path)` | Đọc ảnh, đổi BGR→RGB, báo lỗi rõ nếu OpenCV không giải mã được |
| `detect_content_type(path)` | Trả `"image"`, `"video"`, hoặc `None` theo đuôi file |
| `is_supported_media(path)` | Có hỗ trợ định dạng này không |
| `imread_bgr(path)` | Đọc ảnh **an toàn Unicode**: `cv2.imread` hỏng khi đường dẫn có ký tự tiếng Việt, nên dùng `np.fromfile` + `cv2.imdecode` |
| `open_video_capture(path)` | Mở video, thử dạng URI an toàn Unicode trên Windows |
| `media_path_for_config(path, root)` | Ưu tiên đường dẫn tương đối gốc project khi lưu config |
| `save_content_settings(...)` | Ghi đường dẫn và loại media trở lại `content.json`. ⚠️ **Chỉ ghi một phần** — xem Phần V |

---

## 3.12. `src/gui/` — Giao diện

### `calibration_wizard.py` — Wizard 5 bước

| Bước | Nội dung |
|---|---|
| 1/5 | Kích thước màn hình vật lý (mm) và độ phân giải (px) |
| 2/5 | Chọn monitor |
| 3/5 | Vị trí webcam so với tâm màn hình (mm) |
| 4/5 | **Calibration `face_scale`** — thu trực tiếp tại khoảng cách đo bằng thước |
| 5/5 | Xem lại và lưu vào `user.json` |

| Hàm | Công dụng |
|---|---|
| `CalibrationWizard.__init__(config, output)` | Sao chép cấu hình, **nạp calibration đã có** để lúc lưu không làm mất |
| `.run()` | Vòng lặp wizard: `+/−` chỉnh, `N/Enter` tiếp, `P` lùi, `C` chạy capture, `S` lưu, `Q/Esc` thoát |
| `._capture_face_scale()` | **Hàm quan trọng**: mở camera + tracker, thu 60 frame, lọc theo `presence ≥ 0.5` và `rmse ≤ 6`, gọi `compute_face_scale`. Dùng model **chưa calibrate** vì `face_scale` chính là hiệu chỉnh giữa model chung và người dùng này. Khi lỗi thì in kèm phân bố RMSE quan sát được để biết chỉnh ngưỡng nào |
| `._draw_capture_progress(i, valid, rejected)` | Vẽ tiến trình thu |
| `._draw_face_scale_step(canvas)` | Vẽ bước 4: ô khoảng cách, kết quả hoặc lỗi, phần trợ giúp |
| `._refresh_fields()` | Nạp danh sách trường có thể chỉnh cho bước hiện tại |
| `._draw_step()` | Vẽ canvas theo bước |
| `._adjust_field(direction)` | Tăng/giảm giá trị trường đang chọn (khoảng cách calibration bước 10 mm) |
| `._apply_field(key, value)` | Ghi giá trị vào đối tượng tương ứng |
| `._next_step()` / `._previous_step()` | Điều hướng bước |
| `._build_config()` | Dựng `AppConfig` từ giá trị đang chỉnh |
| `._save()` | `save_config` rồi **gộp lại** khối `face_calibration` (vì `save_config` ghi đè toàn bộ file) |
| `._put_text(...)` | Vẽ text có viền đen cho dễ đọc |
| `_wrap(text, width)` | Ngắt dòng thông báo lỗi dài |
| `run_calibration_wizard(config, output)` | Hàm tiện ích khởi chạy wizard |

### `file_dialog.py`

| Hàm | Công dụng |
|---|---|
| `pick_media_file()` | Mở hộp thoại chọn file gốc của hệ điều hành bằng tkinter (F6) |

---

## 3.13. `assets/` — Tài nguyên

### `assets/config/`

| File | Ai đọc | Công dụng |
|---|---|---|
| `default.json` | `merge_user_config` → `AppConfig` | Cấu hình gốc: camera, screen, filter, tracking, optimization, renderer, ui, **debug** |
| `user.json` | phủ lên default | Ghi đè cá nhân + khối `face_calibration` |
| `screen.json` | `_load_screen_calibration` | Mặt phẳng màn hình **vật lý**, đơn vị **mét**. **Thắng `default.json`** |
| `content.json` | `_load_content` | Toàn bộ tham số nội dung/depth/lớp |
| `ui.json` | `_load_ui_settings` | Bố cục PiP |
| `camera_calibration.json` | *không ai đọc lúc chạy* | Nội tham số còn sót lại |
| `default.yaml` | — | Bản YAML tương đương |

Giá trị `screen.json` hiện tại (laptop 15.6"):
```json
{ "screen_width": 0.344, "screen_height": 0.194,
  "screen_center": [0,0,0],
  "camera_offset": [0.0, 0.103, 0.005],
  "eye_offset":    [0.0, 0.0, 0.0] }
```
Webcam nằm cao hơn tâm màn hình 103 mm và nhô ra trước 5 mm.

### `assets/shaders/`

| File | Công dụng |
|---|---|
| `phong.vert` / `phong.frag` | Đổ bóng Phong cho vật thể scene graph |
| `video.vert` | Vertex shader chung cho mọi quad có texture |
| `video.frag` | Fragment shader RGB đơn giản (mặt phẳng phẳng) |
| `video_rgba.frag` | Fragment shader **cho lớp**: loại bỏ fragment có `alpha < uAlphaThreshold`, còn lại blend mềm |
| `mist.frag` | Sương mù: độ mờ giảm dần theo bán kính từ tâm |
| `line.vert` / `line.frag` | Vẽ đoạn thẳng gỡ lỗi (khung frustum) |
| `hud.vert` / `hud.frag` | Vẽ quad màn hình cho HUD/toast/preview |
| `basic.vert` / `basic.frag` | Shader tối giản |

### `assets/models/`
`face_landmarker.task` — model MediaPipe Tasks (bắt buộc cho Tasks API).

---

## 3.14. `tests/` — 26 file, 177 test

Chạy `.\.venv\Scripts\python.exe -m pytest -q`. **Không cần** context OpenGL hay
webcam — các test nội dung/toán chạy trên mảng NumPy.

| File | Kiểm tra |
|---|---|
| `test_face_calibration.py` | **(mới)** đơn vị `FACE_MODEL_MM`, `compute_face_scale`, cổng `[0.6,1.6]`, lưu/đọc `user.json`, đường CALIBRATED, và **bất biến của đường legacy** trước phép đổi tỉ lệ model |
| `test_adaptive_k.py` | Luật ngưỡng GVF, nhánh dự phòng depth phẳng |
| `test_boundary_classifier.py` | Trọng số hàn theo gradient depth, thống kê đường nối |
| `test_layer_clustering.py` | k-means, feather, viền sáng, staging, bleed |
| `test_layer_relief.py` | Biên độ và kẹp của lưới relief |
| `test_semantic_depth.py` | Tách trời/làng, giữ chân dung liền khối |
| `test_depth_mesh.py` | Chế độ lưới dịch chuyển (Mode A) |
| `test_content_plane_load.py`, `test_content_formats.py`, `test_video_content.py`, `test_unicode_image_load.py` | Nạp/đổi media, nhận dạng định dạng, đường dẫn không ASCII |
| `test_math_projection.py`, `test_math_quaternion.py`, `test_math_homogeneous.py`, `test_math_pose.py` | Kooima, quaternion, ma trận |
| `test_pose.py`, `test_tracking.py` | solvePnP, tiện ích landmark, **bất biến của `R_CV_TO_HCPD`** (det = +1, trực giao, giữ độ dài) và **dấu thị sai** (tx > 0 → world_x < 0; Z luôn dương; tinh chỉnh mống mắt cùng dấu với translation) |
| `test_disocclusion.py` | Dải khe hở, mask inpaint che toàn bộ lớp gần, feather |
| `test_content_settings_loader.py` | `ContentSettings.from_dict()`: ép kiểu theo default, bí danh key, cảnh báo key lạ |
| `test_filters.py`, `test_prediction.py`, `test_optimization.py` | Bộ lọc, dự đoán, ngân sách độ trễ |
| `test_camera.py`, `test_config.py`, `test_webcam_preview.py` | Đo thời gian thu, round-trip config, bố cục PiP |
| `test_layer_vote.py` | PB-070b: key `layer_method`, bầu nhãn theo vùng, ngoài vùng được nhận thì nhãn y k-means |
| `test_fog_complement.py` | PB-071: key `far_fog_mode`; bản `"atmo"` ra **đúng từng byte** như trước khi thêm key (so với vân tay chụp ở `bef606a`), bản `"complement"` khớp bão hoà và để yên lớp gần |
| `test_static_video_notice.py` | FR-DEPTH-03: chỉ báo khi vừa là video vừa ở chế độ lớp; toast gộp vào một dòng |
| `test_layer_build_toast.py` | PB-076: khung hoãn **không** dựng gì và đặt toast; khung sau dựng với **đúng** bản đồ depth cũ (so từng byte); chế độ lưới (Mode A) không hoãn |

## 3.15. `scripts/` — Công cụ offline

| Script | Công dụng |
|---|---|
| `capture_landmarks.py` | **(mới)** Ghi landmark thật ra `artifacts/landmarks.npz`. Mang từ `kooima_mesh_test` về, chỉ đổi đường dẫn. Cần webcam và người cử động |
| `compare_solvers.py` | **(mới)** Phát lại `.npz` qua EPNP / ITERATIVE / ITERATIVE+seed / SQPNP, in **cả hàng** chỉ số cho từng biến thể. Cố ý in cả hàng — xem §4.6 |
| `z_diag_report.py` | Tổng hợp `artifacts/z_diag.csv`: min/max/std từng cột, mốc `reference_depth_raw`, số frame `presence < 0.25` (**cảnh báo: cột này là hằng 1.0, xem PHẦN V mục 9**), 20 dòng đầu |
| `verify_gvf_threshold_ab.py` | Kiểm chứng luật GVF; lưu ảnh ghép so sánh K=2 vs K=3 → `artifacts/gvf_ab/` |
| `report_gvf_table.py` | In bảng gvf_var / silhouette kèm thời gian nạp |
| `phase2a_weldmaps.py` | Sinh overlay weldmap → `artifacts/phase2/` |
| `measure_real_images.py` | PB-059: chạy pipeline app trên ảnh thật, không mở cửa sổ → `real_images.csv`. `load_to_ready_ms` = nạp ảnh + depth + `_apply_layers`, cộng y hệt nhau cho mọi `layer_method`; model được hâm một lượt không tính giờ |
| `video_gate.py` + `_video_gate_parts.py` + `_video_gate_contend.py` | SPIKE-12: đo đường producer video 2.5D. `--load <giây>` chạy tải nền theo nhịp phát; `--contend-report <csv>` ra FPS **đếm theo cửa sổ 5 giây** (không phải EMA của F1) |
| `rebuild_wait_table.py` | SPIKE-12: dựng lại `gate_wait_table.csv` với `p` đã gồm depth khung khoá chia đều cho `k`; chỉ đọc `gate_summary.csv` |
| `measure_fog_ab.py`, `measure_fog_complement.py`, `measure_layer_method_ab.py` | PB-069 / PB-071 / SPIKE-15: dựng ảnh A/B và bảng số, **không đổi file config nào** |
| `diagnostic_kpolicy_relief.py` | Chẩn đoán K-policy và relief cho từng ảnh |
| `diagnose_gvf_zero.py` | Chẩn đoán trường hợp GVF = 0. Đã sửa: bỏ import `_score_clustering` (hàm này đã bị xoá khỏi `adaptive_k.py`) |

Ảnh thử nghiệm nằm **ngoài** repo tại `D:\PycharmProjects\test_images\`
(`t1, t5, t14, t15, t16, t17`…). `t2.png` (cảnh ngôi làng) là phép thử chuẩn để
kiểm tra trời / làng / đường có rơi vào ba lớp riêng biệt hay không.

---

# PHẦN IV — TRA CỨU NHANH

## 4.1. Phím tắt (nguồn chuẩn: `src/renderer/window.py::_on_key`)

| Phím | Chức năng |
|---|---|
| **F1** | HUD gỡ lỗi + khung frustum |
| **F2** | Xoay vòng bộ lọc: one_euro → kalman → ema → none |
| **F3** | Bật/tắt dự đoán chuyển động |
| **F4** | Bật/tắt khung PiP webcam |
| **F6** | Mở hộp thoại chọn media |
| **F7** | Bật/tắt **sương mù** (A/B) |
| **F8** | Bật/tắt **phối màu khí quyển** (A/B) |
| **F9** | Bật/tắt **lấp khe hở C1** (A/B) |
| **F10** | Bật/tắt **relief trong lớp** (A/B) |
| **G** | Đo mắt nghỉ 2 giây rồi **đặt vật ẩn** (mở/đặt lại ván chơi) |
| **T** | Bật/tắt **tracking** (đứng yên pose hiện tại) |
| **+ / −** | Offset zoom (cộng vào `d` của Kooima, ±5 mm mỗi lần nhấn) |
| **[ / ]** | Dịch Z mặt phẳng nội dung (cần gạt chính) |
| **, / .** | Dịch Z lớp xa |
| **; / '** | Đổi `overscan_e_max_mm` |
| **W A S D / Q E** | Dịch pose thủ công (hỗ trợ phát triển) |
| **Esc** | Thoát |

> Chế độ `--demo` (PB-055) chỉ cho **G, T, F6, kéo-thả, F4, Esc**; mọi phím A/B
> và phím chỉnh tham số bị chặn.

> ⚠️ `README.md` ghi F7 = atmospheric là **sai**. F7 là sương mù, F8 là khí quyển.

## 4.2. Quy trình chỉnh tham số

Sửa `content.json` → nạp lại media bằng **F6** (hoặc kéo-thả).
**Không có hot-reload cấu hình.** Chỉ ba toggle F7/F8/F10 dựng lại lớp từ depth
đã cache (không suy luận depth lại).

| Mục tiêu | Chỉnh gì |
|---|---|
| Nhiều lớp hơn / thị sai mịn hơn | Tăng `adaptive_k_max`, hoặc tắt `adaptive_k_enabled` và đặt `layer_count` |
| Tách làng khỏi trời | `semantic_depth_enabled: true`, tăng `semantic_depth_separation` |
| Bớt quầng trắng quanh cây | `rim_light_intensity: 0` hoặc `0.15`, tăng `edge_feather_px` |
| Chiều sâu mạnh hơn | `layer_z_far_m` âm hơn |
| Màu phẳng / thô | F8 hoặc `atmospheric_enabled: false` |
| Nạp depth nhanh hơn | `semantic_max_work_dim: 480`, ảnh nhỏ hơn |

## 4.3. Đo hiệu năng

- `Profiler`: cửa sổ FPS 120 frame.
- `LatencyBudget`: EMA α=0.15, `compensation = min(pipeline + 0.5·pose_age, 2·target)`.
- `AdaptiveQualityController`: 100→75→50→35 khi FPS < 45, hồi phục sau 60 frame ổn định.
- `TrackingWorker` tự giới hạn ở `track_max_fps = 120`.
- Log thời gian nạp mỗi ảnh:
  `depth_inference | k_selection | semantic_adjust | clustering | mesh_build | total`.

## 4.4. Chẩn đoán trục Z (Phase A)

```bash
# 1. đặt "debug": {"z_diagnostics": true} trong assets/config/default.json
.\.venv\Scripts\python.exe main.py       # nghiêng người ra/vào ~20 s rồi Esc
.\.venv\Scripts\python.exe scripts\z_diag_report.py
# 2. TẮT cờ lại
```

10 cột của `artifacts/z_diag.csv`: `frame_idx, tvec_z_raw, depth_raw,
reference_depth_raw, z_after_scale, z_after_iris, z_after_clamp, z_published,
reproj_rmse_px, presence`.

Cách đọc: cột nào biến thiên mà cột kế tiếp phẳng → lỗi nằm giữa hai chặng đó.
Lưu ý `z_after_iris = z_after_scale + 5.0 mm` là **bình thường** (offset Z của
webcam), vì `EyePositionEstimator` không đụng vào Z.

## 4.5. Ba ca chẩn đoán — cùng một khuôn mẫu

Ba lỗi nặng nhất của dự án đều có chung hình dạng: **triệu chứng ở tầng cuối,
nguyên nhân ở tầng đầu, và giả thuyết trực giác ban đầu đều sai.** Cách duy
nhất phân định được là đo, không phải đọc code kỹ hơn. Ghi lại vì đây là phương
pháp, không chỉ là ba sự cố rời rạc.

| Ca | Triệu chứng | Giả thuyết ban đầu | Nguyên nhân thật | Số đo phân định |
|---|---|---|---|---|
| **Trục Z** | Z "đứng yên ở 605 mm" | Z bị kẹt/bị đóng băng | `FACE_MODEL_MM` lớn gấp 4.71 lần, `_monocular_scale` che mất vì tỉ lệ tự triệt tiêu | Z thật biến thiên 249–817 mm, std 124.74 → triệu chứng báo cáo **không tồn tại** |
| **Weldmap** | Viền làng bị đánh là mềm | Bộ phân loại sai ngưỡng | `render_weldmap` **dilate** trọng số; dilate = lấy cực đại cục bộ → lệch xanh toàn cục | Đổi sang `distanceTransformWithLabels` + LUT thì bản đồ đổi hẳn; bộ phân loại vốn đúng |
| **Giật khung** | Ngồi yên mà khung hình lâu lâu giật | Bộ lọc kém / `dt` dao động cần kẹp | Vi phân theo khoảng thời gian **vòng lặp** thay vì **chụp ảnh** | p99 lúc ngồi yên **3.41 → 1.14 px**, độ trễ khi chuyển động **+0.0 ms** |

Ba hệ quả rút ra:

1. **Đo trước khi sửa.** Cả ba lần, giả thuyết nghe hợp lý đều bị số đo bác bỏ.
   Ca trục Z còn bác bỏ cả *mô tả triệu chứng*.
2. **Tiêu chí nghiệm thu phải có hai nửa.** Chỉ đo lúc ngồi yên thì cách "sửa"
   dễ nhất là tắt dự đoán — mượt hơn nhưng trễ hơn. Vì vậy ca giật khung đo cả
   *ngồi yên* (p99 giảm) lẫn *đang chuyển động* (độ trễ không tăng).
3. **Chặn triệu chứng khác với sửa nguyên nhân.** Kẹp `dt` sẽ làm p99 đẹp lên
   mà mốc thời gian vẫn sai, và lỗi kế tiếp cùng loại sẽ lộ ra ở chỗ khó tìm
   hơn. Xem §3.6 để biết vì sao ba knob kẹp `dt` giữ ở 0.0.

Một biến thể thứ tư đáng ghi vì nó ngược chiều: giả thuyết **"bộ lọc Z chưa
được tinh chỉnh"** cũng bị bác bỏ — đo sáu cấu hình thì `min_cutoff=1.0,
beta=0.007` đang dùng cho nhiễu dư thấp nhất (0.82 mm), còn phương án định thay
thế sẽ **tăng gấp ba** lên 2.38 mm. Không phải giả thuyết nào cũng dẫn tới một
bản sửa; đôi khi kết quả đo là "giữ nguyên".

## 4.6. `reproj` MỘT MÌNH không phải cổng chất lượng

Ghi lại vì nó **sửa một khuyến nghị trong đề cương** (§3.10 đề xuất dùng sai số
tái chiếu thay cho `presence`) bằng số đo, chứ không bằng lập luận.

Bảng so solver trên 1200 khung thật cho một kết quả nghịch lý nếu chỉ đọc một
cột:

| biến thể | `reproj` trung vị | `tz<0` | độ lệch chuẩn độ sâu |
|---|---|---|---|
| ITERATIVE | **10.07 px — thấp nhất** | **18.6% — hỏng nhiều nhất** | **229 mm** |
| SQPNP | 10.42 px | 0.0% | 29 mm |
| EPNP | 14.08 px | 0.0% | 19 mm |

Biến thể có sai số tái chiếu **thấp nhất** cũng là biến thể **hỏng nhiều
nhất**. Lý do là hình học, không phải trùng hợp: `FACE_MODEL_MM` gần đồng
phẳng (tỉ lệ phẳng 0.26), nên bài toán PnP có **nghiệm gương** — lật tấm phẳng
theo chiều ngược lại cho ra ảnh chiếu gần như giống hệt. Nghiệm gương khớp ảnh
rất đẹp; nó chỉ sai về **chiều sâu**, thứ mà reproj không nhìn thấy.

**Kết luận dùng được:** `reproj` thấp là điều kiện **cần**, không phải điều
kiện **đủ**. Một cổng chất lượng dùng được phải kết hợp ít nhất hai vế:

```
tin cậy  ⟺  reproj thấp  VÀ  khoảng cách nằm trong dải vật lý hợp lý
```

Vế thứ hai chính là bộ đếm nghiệm rác trong `compare_solvers.py`. Xem PHẦN V
mục 10 về việc chọn cận dưới cho dải đó.

### Cổng nhất quán hình học — đã thiết kế, đã đo, đã BỎ

Ứng viên thay cho `presence` chết: so `tz` của solvePnP với một ước lượng độ
sâu độc lập dựng từ bề rộng hai đuôi mắt. **Kết luận: không triển khai.** Ghi
lại vì con đường dẫn tới kết luận đó là chất liệu báo cáo, và vì nó ngăn người
sau làm lại đúng vòng này.

**Ba lần đo, ba lần kết luận đổi.**

*Lần 1 — sai vì chọn nhầm lớp đối chứng.* Lấy độ tản từ khung tốt của
ITERATIVE. Nhưng ITERATIVE hỏng đúng ở tư thế khó, nên lớp "tốt" của nó đã bị
lọc sạch các ca khó, làm độ tản chặt giả tạo. Ra 82% bắt được — **sai**. Lớp
tốt thật của SQPNP gồm cả yaw ±60°, độ tản rộng gấp 5 lần, và cổng chỉ bắt
3.1%.

*Lần 2 — tìm ra một lỗi công thức thật.* `z_geom` đo độ sâu tới **mặt phẳng
hai mắt**, `tz` đo tới **gốc model là chóp mũi**; hai thứ lệch một hằng số
~30 mm nên tỉ số là `1 + const/tz`, **trôi theo khoảng cách**. Sửa thành so với
độ sâu của chính mặt phẳng hai mắt thì dữ liệu tổng hợp model-khớp-hoàn-hảo cho
**đúng 1.0000** ở cả ba khoảng cách. Lỗi này do một test hỏng lộ ra, không phải
do đọc code.

*Lần 3 — lớp lỗi đang đo ĐÃ bị chặn sẵn.* Sau khi sửa công thức, cổng bắt
**100%** khung hỏng của ITERATIVE. Nhưng toàn bộ khung đó là `tz <= 0`, mà
`head_pose.estimate` **đã** trả `detected=False` cho `tz <= 0` từ trước. Cổng
chỉ phát hiện lại một lỗi đã bắt.

**Phép thử quyết định.** Lớp nguy hiểm thật là nghiệm sai mà `tz` vẫn dương —
plausible nhưng xoay sai, không cổng nào đang chặn. Ép nó xuất hiện bằng nhiễu
Gauss vào landmark:

| Nhiễu | Số nghiệm hỏng (>50 mm, `tz`>0) | Cổng bắt | Báo động giả |
|---|---|---|---|
| 1 px | **0** | — | 0.2% |
| 2 px | **0** | — | 0.1% |
| 4 px | **0** | — | 0.3% |
| 8 px | 6 | 16.7% | 0.5% |

Nhiễu landmark **thật** đo trên 151 khung người ngồi yên là **2.8 px**. Ở mức
đó SQPNP sinh **0** nghiệm hỏng. Cổng sẽ trả phí 0.1–0.3% báo động giả để bắt
một lớp lỗi **không xảy ra**.

**Vì sao vẫn nên biết:** nếu sau này đổi solver, đổi model mặt, hoặc chất lượng
landmark tụt (camera kém hơn, thiếu sáng), lớp lỗi này có thể xuất hiện. Lúc đó
công thức đúng đã có sẵn ở đây, cùng cách dựng lớp đối chứng — **đừng dùng
khung `tz<=0` làm lớp dương**, nó đã bị chặn sẵn.

Đây cũng là lý do `compare_solvers.py` in **cả hàng** chỉ số cho mọi biến thể
thay vì chỉ cột đang quan tâm. Một phiên trước ở dự án ablation đã cải thiện rò
rỉ pitch 17.4 → 3.7 mm rồi trả giá bằng `tz<0` đi từ 0% lên 73.9% — chỉ số nó
không hề đo — và tự tạo ra hiện tượng đứng hình mà hai lượt sau phải đi chẩn
đoán.

## 4.7. Hiệu chỉnh `face_scale` từ phiên quay thước (Phase D)

> ### GIÁ TRỊ ĐANG DÙNG — MỤC NÀY LÀ BẢN LƯU DUY NHẤT
>
> ```
> face_scale = 0.9555 ± 0.0003   (4 lần chạy)
> ```
>
> Nằm trong `assets/config/user.json`, mà file đó **trong `.gitignore`**. Đúng
> về nguyên tắc — nó là hiệu chỉnh riêng của một khuôn mặt và một webcam,
> không phải của dự án. **Mặt trái: không ai ngoài máy này tái lập được kết
> quả**, và mất file là mất số. Nên mục này ghi cả giá trị lẫn điều kiện sinh
> ra nó, để tài liệu làm bản lưu.
>
> | Điều kiện | |
> |---|---|
> | Nguồn | `tracking_benchmark/data/ruler_session/session_20260714T140310Z` |
> | Người / máy | cùng người, cùng webcam, quay bằng `ruler_test.py` |
> | Solver | `SOLVEPNP_SQPNP`, `fast=True` |
> | Intrinsics | `CameraIntrinsics.from_image_size(1280, 720)` → `fx = 1044.4` |
> | Model mặt | `FACE_MODEL_MM` (90.00 mm mắt↔mắt, 67.27 mm mũi→cằm) |
> | Mốc tham chiếu | `ax_600` (`\|tz\|` = 628.0 mm, std 3.8, 151 khung) |
> | Tái lập | `.\.venv\Scripts\python.exe scripts\calibrate_from_ruler_session.py` |
>
> **Vì sao có dải ±0.0003 thay vì một số duy nhất:** `face_tracker.py:150` cấp
> dấu thời gian cho MediaPipe bằng `time.perf_counter()` — **đồng hồ tường**.
> Ở chế độ video, MediaPipe lọc theo thời gian dựa vào dấu này, mà khi phát lại
> offline thì tốc độ chạy không liên quan gì tới thời gian thật của video, nên
> mỗi lần chạy cho landmark hơi khác. Đã kiểm: **hai tracker mới trên cùng một
> khung lệch đúng 0.0 px** (mô hình tất định), còn **cùng một tracker gọi hai
> lần trên cùng khung lệch 0.51 px** (trạng thái thời gian). Biên độ ±0.03%
> tương đương ±0.18 mm ở 600 mm — nhỏ hơn cả sai số dư 0.5 mm, nên ghi lại chứ
> không sửa.

`face_scale` **đã hiệu chỉnh** ngày 2026-09-20. Không chạy wizard — dùng một
phiên quay có sẵn của dự án anh em `d:\tracking_benchmark`,
`data/ruler_session/session_20260714T140310Z`, cùng người cùng webcam.

**Vì sao chọn nó thay vì wizard:** wizard cho **một** khoảng cách, phiên quay
cho **ba** (500 / 600 / 700 mm) đã đo thước, cộng sáu mốc ngang. Ba điểm mới
tách được *sai số tỉ lệ* khỏi *sai số gốc toạ độ*; một điểm thì không.

> ### CẢNH BÁO — ĐỪNG TÁI DÙNG SỐ CỦA `tracking_benchmark`
>
> | | `fx` dùng để giải PnP |
> |---|---|
> | `tracking_benchmark` | `max(w, h)` = **1280 px** |
> | `test_true3D` | **1044.4 px** (suy từ giả định FOV 63°) |
>
> Chênh **22.6%**. `tvec` **tỉ lệ thuận** với `fx`, nên `|tz|` của hai dự án
> lệch đúng chừng đó — mà **cả hai con số đều trông hợp lý**, không có gì báo
> động. Đây là loại bẫy sinh ra sai số 22% một cách âm thầm.
>
> **Luôn giải lại từ video bằng intrinsics của dự án này.** Không chép `|tz|`,
> không chép `face_scale`, không chép MAE. `scripts/calibrate_from_ruler_session.py`
> luôn tự giải lại và in `fx` nó đang dùng ở dòng đầu, chính vì lý do này.

Giải lại bằng đúng chuỗi của dự án này (SQPNP, `fx = 1044.4`).

| Mốc | đo thước | `\|tz\|` | `face_scale` hàm ý |
|---|---|---|---|
| ax_500 | 500 mm | 522.9 mm | 0.9562 |
| ax_600 | 600 mm | 628.0 mm | **0.9555** |
| ax_700 | 700 mm | 733.4 mm | 0.9545 |

451 khung, **0 khung trượt**, std 1.9–3.7 mm, `reproj` 6.8–8.9 px.

**Kết luận đọc theo đúng tiêu chí Phase D.** Sai số trước hiệu chỉnh là
**+4.6% / +4.6% / +4.7%** — phần trăm **phẳng** qua cả ba khoảng cách, tức
**sai số tỉ lệ thuần**, sửa được bằng một vô hướng. (Nếu phần trăm thay đổi
theo khoảng cách thì đã phải ghi là hạn chế phương pháp, không được thêm hệ số
bù.) Sau hiệu chỉnh: 499.7 / 600.0 / 700.4 mm, lệch tối đa **0.4 mm**.

**Kiểm chéo độc lập — trục ngang.** Cùng vô hướng đó, suy ra hoàn toàn từ trục
Z, áp lên sáu mốc ngang: MAE **6.7 mm** cả sáu, **2.3 mm** khi bỏ hai mốc ±200
(benchmark cũng loại hai mốc này), hệ số góc khớp 0.9453. Một số rút từ Z lại
sửa đúng X — đó là tỉ lệ vật lý, không phải khớp đường cong.

### Hai điều KHÔNG được trích dẫn sai

**1. Mơ hồ +5 mm, cố ý không bù.** Chạy qua `HeadPoseEstimator` thật cho sai số
**+4.7 / +5.3 / +5.9 mm** — hằng số +5 mm, đúng bằng `camera_offset` Z trong
`screen.json`. Hai cách đọc đều hợp lệ: nếu thước đo tới **camera** thì +5 mm
là *đúng*; nếu đo tới **màn hình** thì `face_scale` phải là 0.9472.
`ruler_test.py` **không ghi gốc toạ độ** ở bất kỳ đâu. Chênh lệch 0.8%, nhỏ hơn
độ chính xác của chính phép đo thước, nên giữ 0.9555 và ghi lại thay vì thêm
một hệ số bù cho quy ước không ai ghi.

**2. Con số 74.5 mm KHÔNG phải khoảng cách đuôi mắt của người dùng.**
`eye_px · 600 / fx` cho 74.5 mm, nhưng `fx` là **giả định** FOV 63°. Muốn ra
90 mm thì FOV phải là 72.6°. `face_scale` nuốt cả sai số kích thước mặt lẫn sai
số `fx`, nên Z đúng dù nguyên nhân là cái nào — nhưng **đừng trích 74.5 mm như
một số đo cơ thể**. Tách được hai thứ này cần đo `fx` riêng (chụp vật có kích
thước chuẩn, ví dụ cạnh dài A4 = 297 mm, ở khoảng cách đã đo thước).

---

# PHẦN V — LỖI VÀ CẠM BẪY ĐÃ BIẾT

Đã kiểm chứng trên code hiện tại. Sửa có chủ đích, đừng sửa vô tình.

1. **`save_content_settings` ghi đè một tập con khoá.** Nó **đọc file cũ trước**
   rồi `payload.update(...)`, nên mọi khoá lớp/semantic/relief đã có trong
   `content.json` **được giữ nguyên** — không mất như từng lo. Hai giới hạn
   thật: (a) các toggle runtime F7/F8/F10 không được ghi lại, và (b) nếu
   `content.json` chưa tồn tại thì chỉ tập con được ghi ra.

2. **`layer_count` bị bỏ qua** khi `adaptive_k_enabled: true`.
   Đặt `adaptive_k_enabled: false` mới dùng được.

3. **Depth của video chỉ lấy frame đầu.** `DepthWorker` cache theo đường dẫn;
   chưa có depth theo từng frame (TODO đã ghi trong code, chi phí CPU cao).

4. **Nội tham số camera là ước lượng** từ giả định FOV 63°.
   `camera_calibration.json` có tồn tại nhưng đường chạy thật không đọc.

5. **`CALIBRATION_MAX_RMSE_PX` đã nâng 6.0 → 15.0** (lệch có chủ đích so với
    spec gốc). Đo thực tế trên máy này: RMSE tái chiếu min=7.07, median=13.18,
    max=33.42 px — với ngưỡng 6.0 thì **mọi frame đều bị loại** và calibration
    không bao giờ thành công. RMSE **bất biến với `face_scale`** (đó là sai lệch
    *hình dạng* giữa mặt người dùng và model chung, không phải kích thước), nên
    calibrate xong con số này vẫn nguyên. Lý do ghi trong comment tại
    `src/pose/calibration.py`. Nếu vẫn thiếu frame hợp lệ, nâng
    `CALIBRATION_FRAMES` thay vì nới tiếp ngưỡng.

6. **`nyu_depth_v2_labeled.mat` nặng ~2.97 GB** ở gốc project, cùng với
    `nyu_all/`, `nyu_test/` và vài PNG (`atmo_ab.png`, `hybrid_ab.png`…).
    Đây là tàn dư nghiên cứu, **không** phải phụ thuộc lúc chạy.

7. **`hybrid_regions.py` / `layer_split.py` / `adaptive_layers.py` không tồn tại.**
    Ghi chú từ các phiên cũ có nhắc tới chúng. Thứ thực sự có trong repo là
    `adaptive_k.py` (chọn K tự động) và `boundary_classifier.py` (phân tích
    đường nối).

8. **Ba knob vận tốc nối dây đầy đủ nhưng không chạy — có chủ ý.**
    `velocity_smoothing`, `velocity_dt_min_s`, `velocity_dt_max_s` = 0.0 nghĩa
    là tắt, và `default.json` cố tình không đặt. Đây là **quyết định**, không
    phải sót. Lý do và điều kiện bật lại nằm ở §3.6 và ở comment trong
    `src/config/schema.py::FilterSettings`. Cùng họ với mục 2 (`layer_count`)
    nhưng khác bản chất: mục 2 là bẫy thật, mục này là bẫy đã được gỡ ngòi bằng
    tài liệu tại chỗ.

9. **`FaceTracker.confidence` LUÔN bằng đúng 1.0 — cổng tin cậy là code chết.**
    Model face mesh xuất một điểm số presence cho **cả khuôn mặt**, không xuất
    theo **từng landmark**. Thư viện gán `presence = c_struct.presence if
    c_struct.has_presence else None`, nên `lm.presence` luôn `None`;
    `face_tracker.py` lọc rỗng rồi rơi về hằng `1.0`. Backend Solutions còn
    `return pts, 1.0` cứng. Kiểm trên 1200 khung thật trải yaw ±60° và
    142–840 mm: `min == max == 1.0`.

    **Hệ quả:** nhánh `confidence < 0.25` → giữ pose 8 khung trong
    `tracking_worker.py` **chưa từng chạy một lần nào**. Ngưỡng
    `CALIBRATION_MIN_PRESENCE = 0.5` trong wizard cũng không bao giờ loại được
    khung nào. Cột `presence` trong `z_diag.csv` đang ghi một hằng số.

    **Đã sửa một nửa** (2026-09-19): đường solvePnP hỏng giờ đi vào
    `_hold_or_simulate` — trước đó nó nhảy về `reference_distance` và ghi cú
    nhảy đó vào `_last_confident_eye`.

    **Nửa còn lại: đã thiết kế, đã đo, đã BỎ.** Ứng viên thay thế — cổng nhất
    quán hình học — được đo trên dữ liệu thật và **không đáng triển khai**: ở
    mức nhiễu landmark thật (2.8 px), SQPNP sinh **0** nghiệm hỏng mà `tz` vẫn
    dương, tức lớp lỗi nó nhắm tới **không xảy ra**. Toàn bộ quá trình và ba
    lần đổi kết luận nằm ở §4.6 — đọc trước khi định làm lại, để khỏi lặp đúng
    vòng đó.

    **Hiện trạng thật:** đường bảo vệ duy nhất đang chạy là cổng `tz <= 0`.
    Không có cổng chất lượng nào khác, và đó là **quyết định có bằng chứng**
    chứ không phải thiếu sót. Nhánh `confidence < 0.25` vẫn nằm đó và vẫn
    chết — giữ lại vì xoá nó đòi dọn cả `_hold_frames_remaining` và
    `_last_confident_eye`, mà hai thứ đó là ngân sách giữ pose có thể còn dùng.

10. **Cận dưới `DIST_SANE_MM = 200 mm` bắt nhầm khung sát thật.**
    Bản ghi 1200 khung chạm 142 mm, và đó là khoảng cách **thật**: 58 khung đó
    gom thành 5 đoạn liên tục, bề rộng mắt trên ảnh đạt 625 px, tam giác đồng
    dạng `fx·90/625` cho 150 mm, tương quan `|t|` với nghịch đảo bề rộng mắt là
    0.91. Dùng ngưỡng 200 mm lúc chạy sẽ báo động giả 4.8%.

    **ĐÃ SỬA** ngày 2026-09-20: cận dưới hạ xuống **120 mm**, và hằng số giờ
    nằm ở **một chỗ duy nhất** — `src/pose/head_pose.py::DIST_SANE_MM` —
    `scripts/compare_solvers.py` import về thay vì giữ bản sao. Sau khi hạ,
    cột nghiệm rác về **0.0% trên cả bốn biến thể**; toàn bộ 4.8% cũ là báo
    động giả.

    **Nhưng 120 mm thừa hưởng một giả định**: nó suy ra từ FOV 63° mà
    `CameraIntrinsics.from_image_size` **ước lượng**, không phải từ hiệu chuẩn
    camera thật. Có hiệu chuẩn thật thì phải tính lại.

    Kèm theo là **bộ đếm nghiệm rác** trong `head_pose.estimate` (Phase C, thu
    hẹp): `tz` dương nhưng `|t|` ngoài dải → tăng bộ đếm và ghi log, **không
    chặn khung**. Quan sát chứ không can thiệp, vì SQPNP sinh 0/1200 và 0/451
    nghiệm rác nên chưa có gì để chặn; nhưng dự án anh em từng thấy một khung
    `dist = 451.488.954 mm` mà **cơ chế tới giờ vẫn chưa giải thích được**, nên
    nếu nó tái xuất thì lần này có dấu vết. Log chỉ ghi khi kích hoạt, chi phí
    khi không có gì xảy ra bằng 0.

    **Bốn cột `tz<0` của Phase C gốc thì KHÔNG làm.** Đường `tz<=0` đã được xử
    lý đúng ở Việc 1 và không còn là điểm nóng; thêm đo đạc cho một đường đã
    yên là trả phí không đổi lấy gì.

11. **`obj_loader.load_obj` im lặng vứt hình học — nhưng là code chết.**
    Vòng đọc face lấy `parts[1:4]`, tức ba corner đầu của mọi face. Quad mất 1
    corner, n-gon mất nhiều hơn, **không cảnh báo gì** và ảnh vẫn trông có vẻ
    đúng. Trên một model thử ở dự án anh em (7388 quad + 41 n-gon / 10869 face)
    sẽ mất hơn nửa hình học.

    **Không ai gọi.** `load_obj` được export trong `src/renderer/__init__.py`,
    `load_obj_mesh` có gọi nó, nhưng `load_obj_mesh` cũng không nơi nào dùng.
    Lỗi tiềm ẩn trong code chết, không phải lỗi đang chạy → **ghi cảnh báo tại
    chỗ, không sửa**. Muốn dùng lại thì phải tam giác hoá face, hoặc tối thiểu
    báo lỗi khi gặp face > 3 corner.

12. **`default.json` ghi 531 × 299 mm và con số đó VÔ HIỆU.**
    `app.py::_load_screen_calibration` cho `screen.json` thắng, nên kích thước
    thật đang dùng là **344 × 194 mm**. Đã thêm khoá `_comment` vào khối
    `screen` của `default.json` nói rõ điều đó ngay tại chỗ — `_merge_dataclass`
    bỏ qua khoá lạ nên an toàn (đã kiểm: nạp lại bình thường). Giữ 531 × 299 vì
    `ScreenSettings` vẫn cần giá trị mặc định khi `screen.json` vắng mặt.

13. **Lệch tỉ lệ khung hình 0.26%.**
    Frustum dựng từ màn hình vật lý 344 × 194 mm (tỉ lệ **1.7732**), viewport
    là 1280 × 720 px (tỉ lệ **1.7778**). Ảnh bị kéo ngang 0.26%.

    **Nguyên nhân gọn hơn nó trông:** để khớp đúng 16:9 thì chiều cao phải là
    **193.50 mm**, mà `screen.json` ghi **194.0** — lệch **0.5 mm**, tức đúng
    sai số làm tròn của một ước lượng "15.6 inch 16:9". Chính `screen.json` đã
    tự ghi `TODO: measure your specific model with a ruler`. Nên đây **không
    phải lỗi code**; đo màn hình thật bằng thước là hết. Dưới ngưỡng nhìn thấy
    ở 1280 × 720, nhưng sẽ lớn hơn nếu đổi độ phân giải.

14. **Bóng cành cây tối vắt qua nền trời (vấn đề cấu trúc mảnh).**
    Cành cây mảnh bị cắt vào một lớp GẦN hơn, để lại lỗ trên lớp trời. Telea
    lấp lỗ đó bằng màu lấy từ **biên của lỗ** — mà biên chính là màu cành tối.
    Kết quả là những vệt tối uốn lượn nằm **lệch** so với cành thật, và chúng
    trượt theo thị sai của lớp trời chứ không theo lớp cành.

    Đây đúng là bài toán **cấu trúc mảnh** mà SLIDE (ICCV 2021) nhắm tới: tách
    lớp cứng không xử lý nổi vật thể mảnh hơn một lớp. Sửa nó cần biên mềm dẫn
    theo gradient **độ sâu**, không phải vá inpainting. **Chưa sửa.**

15. **Lỗ đen lớn khi ghé sát — vượt dải disocclusion 64 px.**
    `DisocclusionParams.fill_px = 64`, tương đương khoảng 170 mm dịch đầu.
    Ghé gần hơn thì vùng lộ ra vượt dải đã lấp và thành lỗ đen. Đây là **hạn
    chế phương pháp đã biết**, đã ghi trong README mục tiêu 5; nới `fill_px`
    làm Telea phải bịa ra nhiều hơn và chất lượng tụt nhanh. **Chưa sửa.**

16. **`_plane_height` là trạng thái phụ thuộc thời điểm — đừng đọc nó để tính.**
    Trường này bằng `plane_width_m / aspect`, mà `aspect` lúc là tỉ lệ ảnh
    thật, lúc là **16:9 dự phòng**, tuỳ `_resize_plane` đã chạy với giá trị
    nào. Đo thật: hệ số overscan cho cùng một lớp `t = 1100 mm` ra **1.191**
    ở ba lần chạy và **1.525** ở một lần — vì chiều cao quad lúc 0.96 m
    (4:3 thật) lúc 0.72 m (16:9 dự phòng).

    `_overscan_scale` giờ **nhận kích thước quad qua tham số**, lấy từ biến
    `width`/`height` mà `_apply_layers` đã tính từ kích thước texture thật.
    Ai thêm phép tính mới dựa trên kích thước quad thì làm y như vậy, đừng đọc
    `self._plane_height`.

## Đã sửa — đừng đảo ngược

Sửa ngày 2026-09-04. Ghi lại vì phần lớn đều "trông như code đúng" nếu đọc lướt.

**0. Dọn dẹp đợt 1.**
- `DebugOverlay._format_lines` từng `return` thẳng list literal, biến khối
  `layer_grade_lines` phía sau thành code chết → dòng trạng thái lớp không bao
  giờ hiện ở HUD dự phòng. Đã gán vào `lines` rồi mới `return`.
  `_draw_fallback` cũng từng gọi hàm này **hai lần**, nay dùng lại kết quả.
- **5 knob adaptive-K đã bị xoá khỏi toàn chuỗi** (`content.json` → `app.py` →
  `ContentSettings` → `content_plane.py` → `select_optimal_k`):
  `adaptive_k_tolerance`, `adaptive_k_flat_threshold`, `min_layer_coverage`,
  `min_depth_separation`, `min_gvf_gain`. Chúng được nhận rồi vứt bằng
  `_ = (...)`, nên đặt giá trị trong config **không có tác dụng gì**. Xoá hẳn
  thay vì để làm bẫy. Luật GVF không đổi. Nhánh dự phòng depth phẳng dùng
  `_FLAT_DEPTH_STD` chứ không dùng `flat_threshold`.
- `scripts/diagnose_gvf_zero.py` import `_score_clustering` đã bị xoá → sửa
  import và cập nhật phần in cho khớp GVF thật. Không xoá file vì project
  **không có git**, xoá là mất vĩnh viễn.
- `README.md` thiếu F4/F6/F7/F8/F10/`+`/`−` và còn ghi lệnh test là
  `unittest discover` trong khi dự án dùng `pytest`.
- `one_euro.py::filter` tạo `z_config`/`z_filter` mỗi mẫu rồi không dùng —
  7 lần cấp phát thừa mỗi pose mỗi frame. Đã xoá, đầu ra không đổi.
- `save_content_settings` thêm `ensure_ascii=False` để đường dẫn tiếng Việt
  không bị escape thành `\uXXXX`.

**0b. Loader `ContentSettings` viết lại (dọn đợt 2).**
`app.py::_load_content` từng parse **189 dòng** `payload.get(...)` bằng tay cho
70 field. Thêm field vào dataclass mà quên sửa danh sách đó thì giá trị trong
`content.json` **bị bỏ qua im lặng** — chính là cách 5 knob adaptive-K chết mà
không ai biết. Nay thay bằng `ContentSettings.from_dict()` duyệt
`dataclasses.fields()`, ép kiểu theo **kiểu của giá trị mặc định** (không dùng
`get_type_hints` vì module bật postponed annotations nên annotation chỉ là
chuỗi). `_load_content` còn 22 dòng, chỉ lo I/O.

Hai lưới an toàn mới: giá trị sai kiểu **rơi về mặc định kèm cảnh báo** thay vì
ném lỗi làm chết lúc khởi động; và **khoá lạ bị báo ra** nên gõ sai tên
(`mist_opactiy`) hiện cảnh báo thay vì im lặng không làm gì. Khoá bắt đầu bằng
`_` được coi là chú thích, không cảnh báo. Bí danh cũ giữ trong
`CONTENT_KEY_ALIASES` (`rim_light_intensity` ← `layer_rim_intensity`).

Test `test_content_settings_loader.py::test_every_field_is_reachable_from_json`
là chốt chặn: nó perturb **từng** field rồi kiểm tra giá trị có tới nơi không,
nên lớp lỗi này không thể quay lại mà test vẫn xanh.

Ba lỗi độc lập của khung PiP webcam:

**A. Lật dọc (ảnh lộn ngược).**
`glTexImage2D` coi hàng 0 của mảng (đỉnh ảnh) là `v=0`, nhưng `_draw_quad` gán
`uv(0,0)` cho đỉnh **dưới**-trái. Sửa bằng cách đảo toạ độ V trong
`webcam_preview.py::_draw_quad` và `toast_overlay.py::draw`.
`debug_overlay.py` **vốn đã đúng** — nó bù bằng cách liệt kê đỉnh từ trên xuống,
nên **không đụng vào**. Cách khác là `np.flipud` mỗi lần upload như
`content_plane.py` làm, nhưng tốn một lần copy mảng mỗi frame.

**B. Nhấp nháy.**
Luồng tracking dùng `pop_latest()` — hàm này **xoá sạch** hàng đợi — trong khi
render dùng `get_latest()` và nhận `None` khi hàng đợi rỗng. Vì camera 60 Hz còn
tracking chạy tới 120 Hz nên hàng đợi rỗng phần lớn thời gian, và
`draw()` thoát sớm khiến khung PiP **biến mất hoàn toàn** frame đó (framebuffer
bị `glClear` mỗi frame). Sửa: khi không có frame mới thì **vẽ lại texture đã
cache** thay vì thoát. Không đổi hành vi của luồng tracking — việc nó luôn lấy
frame mới nhất là cố ý để giảm trễ.

**C. Skeleton ngược ngang.**
`_draw_landmarks::pt()` tự lật `x = pw-1-x`, rồi `_prepare_preview_image` lật
tiếp cả canvas bằng `cv2.flip(small, 1)` → landmark bị lật **hai lần** nên lệch
gương so với khuôn mặt, và nhãn confidence thành chữ ngược. Sửa bằng cách **lật
ảnh trước, vẽ overlay sau**; giữ nguyên bù trừ trong `pt()`.
Lưu ý khi kiểm thử: mặt người gần đối xứng nên lỗi này **không thấy được khi
ngồi thẳng** — phải nghiêng hẳn đầu sang một bên mới lộ.

---

# PHẦN VI — TRẠNG THÁI HIỆN TẠI

**Đã hoàn thành và chạy được:**

- Chiếu lệch trục Kooima điều khiển bằng MediaPipe + solvePnP + lọc 1€, có ngân
  sách độ trễ, dự đoán chuyển động, và chất lượng thích nghi.
- Content plane tách lớp (Mode B) với điều chỉnh semantic, phối màu khí quyển
  trong ánh sáng tuyến tính, biên mềm, color bleed, viền sáng, và quad sương mù.
- **Chọn K tự động theo ngưỡng GVF 0.90**, đã gỡ silhouette khỏi đường nóng
  (trước đây treo 40–111 giây mỗi lần nạp).
- **Relief trong lớp (P1)** với chẩn đoán từng lớp và kẹp theo tỉ lệ khoảng cách.
- **Phân loại đường nối Phase 2a** — trọng số hàn đã tính và ghi log, weldmap đã
  vẽ. **Chưa áp vào hình học.**
- **Chẩn đoán trục Z (Phase A)** — CSV 10 cột, gated, mặc định tắt.
- **Calibration `face_scale`** — `FACE_MODEL_MM` đã quy về mm thật (chia 5.0),
  cổng tỉnh táo `[0.6, 1.6]`. **ĐÃ HIỆU CHỈNH** ngày 2026-09-20:
  `face_scale = 0.9555`, `Z path: CALIBRATED`. Sai số Z còn **0.4 mm** trên dải
  500–700 mm đo thước. Nguồn không phải wizard mà là phiên quay thước có sẵn —
  xem §4.7.
- **Lấp khe hở C1** — `src/content/disocclusion.py`, inpaint Telea trong dải
  giới hạn 64 px, mask che toàn bộ lớp gần (C3) để Telea không hút màu tiền
  cảnh vào khe.
- **Sửa giật khung** — truy về mốc thời gian sai, không phải bộ lọc kém.
  p99 lúc ngồi yên 3.41 → 1.14 px, độ trễ khi chuyển động +0.0 ms.
- **Sửa thị sai ngang bị ngược** — gom về một hằng `R_CV_TO_HCPD` định thức +1;
  ba quy ước cũ mâu thuẫn nhau, trong đó bản translation là phép phản chiếu.

**Phase 2b (hàn Z giữa các lớp) đã huỷ**, không phải hoãn: hướng hiện tại là
tách lớp trực tiếp, không có vùng chuyển tiếp, nên trọng số hàn không cần thiết.
Weldmap giữ lại như công cụ chẩn đoán.

**Việc còn treo:**
- Xác nhận bằng mắt rằng chiều thị sai ngang đã đúng sau bản sửa.
- Tách độ phân giải hai đường: tính toán ở 1024, texture RGB giữ nguyên. Đã đo
  là tiết kiệm ~5.3 s mỗi lần nạp, **chưa làm** vì nó đổi ảnh hiển thị ra.

**Project anh em:** `d:\tracking_benchmark` là benchmark độc lập có chủ đích
(MediaPipe vs OpenFace 2.0 vs 6DRepNet). Nó **sao chép** chứ không import code
HCPD, và không bao giờ được sửa project này.
