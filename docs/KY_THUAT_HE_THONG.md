# HCPD — Tài liệu kỹ thuật hệ thống

**Head-Coupled Perspective Display** — biến màn hình thành cửa sổ 3D theo dõi đầu người xem.

| Thuộc tính | Giá trị |
|---|---|
| Ngôn ngữ | Python 3 + OpenGL 3.3 |
| Entry point | `main.py` |
| Render | GLFW + PyOpenGL |
| Tracking | MediaPipe Face Landmarker + `cv2.solvePnP` |
| Depth 2.5D | Depth Anything / MiDaS (tùy chọn) |

---

## Mục lục

1. [Tổng quan kiến trúc](#1-tổng-quan-kiến-trúc)
2. [Pipeline hệ thống](#2-pipeline-hệ-thống)
3. [Hệ quy chiếu & công thức toán học](#3-hệ-quy-chiếu--công-thức-toán-học)
4. [Module theo dõi đầu (Tracking & Pose)](#4-module-theo-dõi-đầu-tracking--pose)
5. [Bộ lọc & dự đoán chuyển động](#5-bộ-lọc--dự-đoán-chuyển-động)
6. [Projection & Virtual Camera](#6-projection--virtual-camera)
7. [Rendering & Shader](#7-rendering--shader)
8. [Depth 2.5D: ước lượng, mesh, phân lớp](#8-depth-25d-ước-lượng-mesh-phân-lớp)
9. [Scene graph & phân tầng Z](#9-scene-graph--phân-tầng-z)
10. [Hệ thống cấu hình](#10-hệ-thống-cấu-hình)
11. [Phím tắt & điều khiển](#11-phím-tắt--điều-khiển)
12. [Chỉ mục hàm theo file](#12-chỉ-mục-hàm-theo-file)

---

## 1. Tổng quan kiến trúc

### 1.1 Nguyên tắc cốt lõi

> **Đối tượng scene không di chuyển.** Chỉ camera ảo (`VirtualCamera`) thay đổi vị trí theo mắt người xem mỗi frame.

Webcam chỉ dùng nội bộ (thread tracking); cửa sổ duy nhất là phòng 3D OpenGL.

### 1.2 Sơ đồ luồng dữ liệu

```
main.py
  └─ merge_user_config(default.json + user.json)
  └─ HcpdApplication.run()                    [Main thread — OpenGL]

Thread CameraWorker
  └─ AsyncCameraCapture → FramePacket queue

Thread TrackingWorker
  └─ FaceTracker.process()
  └─ HeadPoseEstimator.estimate()  (solvePnP)
  └─ EyePositionEstimator.estimate()
  └─ PoseFilterPipeline.update()
  └─ PoseBus.publish(PoseSnapshot)

Main loop (mỗi frame):
  └─ PoseBus.read() → extrapolate (optional)
  └─ VirtualCamera.set_eye_mm(translation + eye_offset)
  └─ SceneRenderer.render()
       ├─ Phong scene graph (frame, backdrop, room)
       ├─ ContentPlaneRenderer (ảnh/video + depth)
       ├─ Debug overlay / Webcam PiP / Toast

Thread DepthWorker (tùy chọn, async):
  └─ DepthEstimator.estimate(rgb)
  └─ ContentPlaneRenderer._poll_depth_mesh()
       ├─ [A] Displacement mesh (per-pixel Z)
       └─ [B] Layer clustering (N lớp RGBA)
```

### 1.3 Cấu trúc thư mục

```
d:\test_true3D\
├── main.py                          # Entry point
├── convert_nyu.py                   # Chuyển NYU Depth V2 .mat → PNG
├── assets/
│   ├── config/                      # JSON/YAML cấu hình
│   ├── shaders/                     # GLSL vertex/fragment
│   ├── models/                      # face_landmarker.task
│   └── video/                       # Media mặc định
├── src/
│   ├── engine/                      # Điều phối app, workers, profiler
│   ├── camera/                      # Capture webcam async
│   ├── tracking/                    # MediaPipe face landmarks
│   ├── pose/                        # solvePnP, head pose, eye refine
│   ├── filters/                     # One Euro, Kalman, prediction
│   ├── math/                        # Projection Kooima, quaternion, pose, ma trận
│   ├── content/                     # Depth, layer clustering, media I/O
│   ├── scene/                       # Scene graph, primitives, room
│   ├── renderer/                    # OpenGL renderer, shaders, content plane
│   ├── config/                      # Schema + loader
│   └── gui/                         # Calibration wizard, file dialog
└── tests/                           # 76+ unit tests
```

---

## 2. Pipeline hệ thống

### 2.1 Khởi động — `main.py`

| Hàm | Công dụng |
|-----|-----------|
| `parse_args()` | Đọc `--config`, `--calibrate`, `-v` |
| `main()` | Load config → `HcpdApplication` hoặc calibration wizard |

**Luồng:**
1. `merge_user_config(default.json, user.json)` — deep merge cấu hình
2. `--calibrate` → `run_calibration_wizard()` (GUI tkinter)
3. Ngược lại → `HcpdApplication(config, PROJECT_ROOT).run()`

### 2.2 Điều phối — `HcpdApplication` (`src/engine/app.py`)

| Hàm / thành phần | Công dụng |
|------------------|-----------|
| `__init__` | Tạo `PoseBus`, `PoseFilterPipeline`, `Profiler`, `AdaptiveQualityController`; load `screen.json`, `content.json`, `ui.json`; khởi tạo workers |
| `_load_screen_calibration()` | Kích thước màn hình vật lý (m), `camera_offset_mm`, `eye_offset_mm` |
| `_load_content()` | `ContentSettings` từ `content.json` |
| `_build_scene()` | `build_video_window_scene()` hoặc `build_room_scene()` |
| `run()` | Vòng lặp GLFW: input → pose → camera → render → swap buffers |
| `_try_load_media()` | F6 / drag-drop → `renderer.load_content_file()` |

**Vòng lặp render (mỗi frame):**
```python
pose = pose_bus.read()
if render_extrapolation and prediction_enabled:
    pose = filters.extrapolate_for_display(pose, pose_age_s)
virtual_camera.set_eye_mm(pose.translation_mm + eye_offset_mm)
renderer.render(metrics, frame, tracking_preview, dt_s)
```

### 2.3 Camera Worker — `src/engine/camera_worker.py`

| Hàm | Công dụng |
|-----|-----------|
| `CameraWorker.start()` | Khởi thread capture |
| `CameraWorker.poll()` | Lấy `FramePacket` mới nhất (non-blocking) |

Dùng `AsyncCameraCapture` (`src/camera/async_capture.py`) — bounded deque, không block render thread.

**`FramePacket`:** `bgr`, `timestamp`, `width`, `height`, `frame_id`

### 2.4 Tracking Worker — `src/engine/tracking_worker.py`

| Hàm | Công dụng |
|-----|-----------|
| `_loop()` | Vòng lặp thread: poll frame → track → pose → filter → publish |
| `_estimate_pose()` | Chuỗi: FaceTracker → HeadPose → EyePosition → clamp → filter |
| `_clamp_translation()` | Giới hạn vị trí mắt trong vùng hợp lý |

**Output:** `PoseSnapshot` qua `PoseBus` (thread-safe slot đơn).

### 2.5 Pose Bus — `src/engine/pose_bus.py`

| Hàm | Công dụng |
|-----|-----------|
| `PoseBus.publish(snapshot)` | Ghi pose mới (tracking thread) |
| `PoseBus.read()` | Đọc pose mới nhất (render thread) |

### 2.6 Profiler & Adaptive Quality

| File | Công dụng |
|------|-----------|
| `profiler.py` | `Profiler` — FPS, breakdown latency (capture/track/render) |
| `latency_budget.py` | `LatencyBudget` — EMA latency → compensation ms |
| `adaptive_quality.py` | `AdaptiveQualityController` — giảm `tracking_scale` khi FPS < target |

---

## 3. Hệ quy chiếu & công thức toán học

> Sai chỗ này là hỏng hết, và đã hỏng thật nhiều lần. Mục này được viết lại
> 2026-09-22 cho khớp code; bản trước ghi lại **đúng hai bug đã sửa**.

### 3.1 Ba hệ tọa độ

| Hệ | Trục | Đơn vị | Dùng ở đâu |
|----|------|--------|------------|
| **OpenCV camera** | X phải trên ảnh, Y **xuống**, Z tới trước vào cảnh | mm | Output `solvePnP` |
| **HCPD world** | X phải, Y **lên**, Z **về phía người xem** | mm | Pose, `VirtualCamera` |
| **OpenGL world** | Trùng HCPD | **mét** | Kooima, scene graph, shader |

**Màn hình vật lý:** mặt phẳng Z = 0, tâm tại `screen_center` (mặc định gốc tọa độ).

**Đổi mm ↔ mét xảy ra đúng một chỗ:** `VirtualCamera.eye_position_m` (`× 0.001`).
Pose đi trong mm tới tận `set_eye_mm`; mọi thứ từ Kooima trở đi là mét.

**Depth map** chuẩn hóa ngược chiều với `depth_norm` của lớp:
`depth` — `1.0 = gần`, `0.0 = xa`; `depth_norm` của lớp là **độ xa** — `0.0 = gần nhất`.

### 3.2 Chuyển đổi tọa độ — CHỈ CÓ MỘT

Toàn hệ thống có **đúng một** phép đổi hệ, trong `head_pose.py`:

```
R_CV_TO_HCPD = diag(-1, -1, 1)        # định thức +1, trực giao, giữ độ dài
```

Suy ra chứ không chọn: webcam nhìn về phía người xem nên `Z_cam` đã trùng world
`+Z`; camera dựng thẳng nên `Y_cam = -Y_world`; hệ thuận tay phải khi đó **ép**
`X_cam = Y_cam × Z_cam = (-1, 0, 0)`. Kiểm bằng trực giác: nhìn ảnh không lật
gương của một người đối diện, tay phải họ nằm bên **trái** ảnh.

| Hàm | Công thức |
|-----|-----------|
| `_opencv_to_hcpd_translation(tvec, offset)` | `R_CV_TO_HCPD @ tvec + webcam_offset_mm` |
| `_opencv_rot_to_hcpd(R)` | `R_CV_TO_HCPD @ R @ R_CV_TO_HCPDᵀ` (trả 4×4) |
| `_euler_from_rotation(R_hcpd[:3,:3])` | pitch/yaw/roll (độ), phân rã ZYX nội tại |

**Cộng `webcam_offset_mm`, không trừ.** Offset là vị trí webcam trong world, nên
một điểm đo trong hệ camera phải **cộng** nó để về gốc tâm màn hình. Từng có bug
trừ thay vì cộng, lệch 206 mm.

**`abs(tvec_z)` đã bỏ.** `tvec_z <= 0` giờ là **nghiệm PnP hỏng** → `detected=False`
→ caller giữ pose cũ. Đếm trên 1077 khung thật: **0 khung** có `tz <= 0`, nên
`abs` chưa từng cứu gì — nó chỉ che một tín hiệu lỗi.

<details>
<summary>Năm quy ước cũ đã xoá (giữ lại để không ai dựng lại)</summary>

| Chỗ | Quy ước cũ | Hỏng thế nào |
|-----|-----------|--------------|
| translation | `[tx, -ty, abs(tz)]` = `diag(1,-1,1)` | định thức **−1**, là **phản chiếu** chứ không phải chuyển hệ rigid → **thị sai ngang ngược dấu**, cảnh có cảm giác xoay theo người xem |
| rotation | `diag(1,-1,-1)` | phép quay hợp lệ nhưng quanh **trục khác** — chỉ vào HUD F1 nên sống rất lâu không ai thấy |
| tinh chỉnh iris | đảo Y, **không** đảo X | *khuếch đại* lỗi thị sai thay vì bù |
| `coordinates.py::opencv_point_to_opengl` | `diag(1,-1,-1)` | quy ước thứ tư; **không caller, không test** |
| `coordinates.py::screen_mm_to_world` | `point_mm − webcam_offset_mm` | **ngược dấu** với đường sống — đúng phép trừ đã gây lệch 206 mm |

Cả module `coordinates.py` đã **xoá** 2026-09-22: sửa dấu vẫn để lại hai bản của
một quy ước, và hai bản của một quy ước là cái bẫy dự án này đã dính nhiều lần.
`frustum.py` xoá cùng lý do — nó là phép chiếu lệch trục **thứ hai**, tính bằng
**mm** (`near_mm = 1.0`) trong khi đường sống tính bằng **mét**; cũng không
caller, không test. Bia mộ nằm trong docstring của `src/math/__init__.py`.

Xác nhận bằng **đo thước vật lý**, không phải suy luận: header kết quả ruler của
`tracking_benchmark` ghi `pred' = (-t_x, t_y, -t_z)` khớp vạch thật, MAE ngang
3.71 mm trên các vạch ±100 mm. Đó là phép đo đầu-cuối trên đúng webcam đó, nên nó
loại trừ luôn khả năng **driver lật gương** — thứ mà đọc code không loại trừ được.

</details>

### 3.3 Ma trận đồng nhất — `src/math/homogeneous.py`

| Hàm | Công dụng |
|-----|-----------|
| `identity4()`, `translation_matrix`, `scale_matrix` | Dựng ma trận 4×4 |
| `multiply(a, b)` | `a @ b` |
| `transform_point` / `transform_direction` | Biến đổi điểm (có tịnh tiến) / vector hướng (bỏ tịnh tiến) |
| `invert(m)` | `np.linalg.inv` |
| `to_column_major_flat(m)` | ⚠ **Tên sai — nó trả về theo HÀNG** |

**Bẫy `to_column_major_flat`:** `m.T` đọc theo `order='F'` chính là `m` đọc theo
`order='C'`, tức **row-major**. Kiểm:
`to_column_major_flat(np.arange(16).reshape(4,4))` cho `[0,1,2,…,15]`, không phải
`[0,4,8,12,…]`.

Tổng thể **vẫn đúng** vì `shader.py::set_mat4` là nơi upload ma trận **duy nhất**
và nó dùng `glUniformMatrix4fv(..., GL_TRUE, ...)` — OpenGL tự transpose. Thêm bất
kỳ upload nào với `GL_FALSE` sẽ transpose âm thầm.

### 3.4 Quaternion — `src/math/quaternion.py`

Format Hamilton: `[w, x, y, z]`

| Hàm | Công dụng |
|-----|-----------|
| `identity_quat()` / `normalize(q)` / `conjugate(q)` | Cơ bản |
| `from_axis_angle(axis, angle_rad)` | Trục-góc → quaternion |
| `from_rotation_matrix(m)` | R → quaternion (Shepperd, 4 nhánh theo trace) |
| `to_rotation_matrix(q)` | Quaternion → ma trận **4×4** |
| `multiply(a, b)` | Tích Hamilton |
| `rotate_vector(q, v)` | `q · v · q*` |

**Không có `slerp` trong file này.** SLERP nằm ở `pose_utils.interpolate_pose`, và
hàm đó *có* xử lý dấu bán cầu (`dot < 0 → q_b = -q_b`).

**Hạn chế đã biết, CHƯA sửa:** `from_rotation_matrix` **không** chuẩn hóa dấu (`w`
có thể âm ở các nhánh `elif`), còn `PoseSmoother.smooth` lọc **4 thành phần độc
lập** rồi mới normalize. Nếu dấu quaternion lật giữa hai khung, bộ lọc trung bình
`q` với `−q` → nội suy xuyên gốc → `normalize` trả `identity_quat()`, tức pose
nhảy về không xoay. Hiếm, và rotation **không** vào phép chiếu (Kooima chỉ dùng
**vị trí** mắt), nên tác động giới hạn ở HUD. Sửa đúng là ép cùng bán cầu với mẫu
trước khi lọc — đúng mẹo mà `interpolate_pose` đang dùng.

---

## 4. Module theo dõi đầu (Tracking & Pose)

### 4.1 Face Tracker — `src/tracking/face_tracker.py`

| Hàm | Công dụng |
|-----|-----------|
| `FaceTracker.process(bgr, process_scale)` | MediaPipe Tasks API (ưu tiên) hoặc Solutions fallback |

Downscale frame theo `process_scale` (kẹp `[0.25, 1.0]`, từ adaptive quality), rồi
**chia lại `scale`** để landmark trả về theo pixel của ảnh **gốc** — nên intrinsics
không phải đổi theo. (Vòng downscale → nhân → chia này thừa: MediaPipe vốn trả
tọa độ chuẩn hóa `[0,1]`, nhân thẳng với kích thước gốc là xong và không có sai
số làm tròn của `int()`.)

**Output:** `FaceTrackingResult(detected, landmarks, confidence, image_width, image_height)`
— `image_width/height` là kích thước ảnh **gốc**.

⚠ **`confidence` luôn bằng 1.0.** Tasks API để `lm.presence = None` cho face
landmarker, nên danh sách lọc ra rỗng và hàm rơi vào mặc định `1.0`; nhánh
Solutions hard-code `1.0`. Hệ quả: cổng `_confidence_threshold = 0.25` trong
`tracking_worker` **không bao giờ kích hoạt**, và `_hold_max_frames` /
`_last_confident_eye` là code chết.

Thay tín hiệu chết này bằng tín hiệu thật đã **thiết kế, đo qua ba vòng, rồi BỎ**
— xem `docs/document.md` PHẦN V mục 9. Lý do: nhiễu landmark thật đo được **2.8 px**,
còn SQPNP chỉ bắt đầu sinh nghiệm hỏng từ **8 px**, nên cổng sẽ trả phí báo động
giả để bắt một lớp lỗi không xảy ra.

**Đừng đề xuất dùng sai số tái chiếu làm độ tin cậy.** §4.6 của `document.md` đã
đo: ITERATIVE có reproj **thấp nhất** (10.07 px) đồng thời hỏng **nhiều nhất**
(18.6%). Nghiệm gương khớp ảnh rất đẹp — reproj thấp không chứng minh nghiệm đúng.

### 4.2 Landmarks — `src/tracking/landmarks.py`

| Hằng số | Giá trị / Công dụng |
|---------|---------------------|
| `PNP_LANDMARKS` | `(1, 152, 33, 263, 61, 291)` — mũi, cằm, hai khóe mắt ngoài, hai khóe miệng |
| `FACE_MODEL_RAW` | Model 6 điểm **đơn vị tùy ý** như ship ban đầu. Tên "mm" lịch sử là sai: đo 2026-08-20 cho `tvec.z = 2824 mm` khi người ngồi thật 600 mm, tức lớn ~4.7× |
| `FACE_MODEL_UNITS_PER_MM` | `5.0` — neo nhân trắc: khoảng hai khóe mắt ngoài của model là 450 đơn vị, bề rộng ectocanthion người lớn trung bình ~90 mm |
| `FACE_MODEL_MM` | `FACE_MODEL_RAW / 5.0`. Đo lại: **90.0 mm** hai khóe mắt, **66.0 mm** mũi→cằm — đều trong dải nhân trắc, nên `face_scale` của một người nằm quanh 1.0 và cổng `[0.6, 1.6]` mới có nghĩa |
| `LandmarkSet` | `image_points` (6×2), `left_iris`, `right_iris`, `all_landmarks` |
| `normalized_to_pixels(lm, w, h)` | `[0,1]` → pixel |

Đường `_monocular_scale` **bất biến** với phép chia 5.0 này (nó chia `depth_raw`
cho `reference_depth_raw`, cả hai co cùng hệ số), nên hành vi khi chưa hiệu chuẩn
không đổi.

### 4.3 solvePnP — `src/pose/solve_pnp.py`

| Hàm | Công dụng |
|-----|-----------|
| `solve_head_pose(model_mm, image_px, intrinsics, fast)` | `fast=True` → **`SOLVEPNP_SQPNP`**; `fast=False` → `SOLVEPNP_ITERATIVE` (giữ để còn A/B được) |

Đổi từ `SOLVEPNP_EPNP` ngày 2026-09-19. Lý do gốc: `FACE_MODEL_MM` gần **đồng
phẳng** — phân rã SVD 6 điểm sau khi dời về trọng tâm cho giá trị kỳ dị
89.21 / 76.49 / 23.35, tỉ lệ phẳng `s3/s1 = 0.2617`, và phần lớn bề dày đó chỉ do
chóp mũi nhô 20.2 mm. Dữ liệu gần phẳng sinh **nghiệm gương** và điều kiện xấu;
EPnP yếu đúng ở cấu hình này vì 4 điểm điều khiển ảo của nó suy biến theo. Bảng đo
1200 khung landmark thật nằm ngay trong file.

**ITERATIVE là bẫy** dù reproj của nó thấp nhất: 18.6% khung trả `tz <= 0`, và độ
lệch chuẩn độ sâu 229 mm so với 19.5 mm của EPNP.

### 4.4 Head Pose Estimator — `src/pose/head_pose.py`

| Hàm | Công dụng |
|-----|-----------|
| `.estimate(landmarks, timestamp)` | **Hàm chính** — xem chuỗi bên dưới |
| `.update_intrinsics(intrinsics, face_scale)` | Nhận K mới giữa phiên; xoá `_reference_depth_raw` nếu fx đổi |
| `._ensure_depth_baseline(depth_raw)` | Khung đầu tiên: lưu `reference_depth_raw` |
| `._monocular_scale()` | `reference_distance_mm / reference_depth_raw` |
| `._count_garbage_solve(tvec)` | **Chỉ đếm**, không chặn khung. `DIST_SANE_MM = (120, 1500)` |

**Chuỗi xử lý:**

```
solvePnP → không success            → detected=False
         → tvec_z <= 0              → detected=False (nghiệm lật; KHÔNG abs nữa)
         → Rodrigues → R_cv
         → đếm nghiệm rác (quan sát, không can thiệp)
         → rẽ nhánh Z:

  [CALIBRATED]  có face_scale → model đã đúng mm thật
      xy = tvec_xy ;  z = tvec_z         (BỎ QUA _monocular_scale hoàn toàn —
                                          hai cơ chế sửa CÙNG một lỗi, chồng
                                          lên nhau là sửa hai lần)

  [LEGACY]      chưa hiệu chuẩn
      scale = reference_distance_mm / reference_depth_raw   (chốt ở khung đầu)
      xy = tvec_xy × scale
      z  = reference_distance × (depth_raw / reference_depth_raw)
         ≡ depth_raw × scale                ← hai nhánh trong code cho CÙNG một
                                              giá trị; nhánh else chỉ chạy khi
                                              baseline chưa chốt, lúc đó scale=1

         → translation_mm = R_CV_TO_HCPD @ [xy, z] + webcam_offset_mm
         → rotation_hcpd  = _opencv_rot_to_hcpd(R_cv)
         → quaternion VÀ Euler đều lấy từ rotation_hcpd    ← MỘT hệ
```

**Hạn chế của đường LEGACY:** `reference_depth_raw` chốt ở khung **đầu tiên**, bất
kể lúc đó người ngồi ở đâu, rồi *khẳng định* đó là `reference_distance_mm` (600 mm).
Sai số tuyệt đối bằng tỉ số khoảng-cách-thật / 600, và nó nhân vào **cả X, Y lẫn
Z**. Đường CALIBRATED không có vấn đề này.

**Euler** (`_euler_from_rotation`): phân rã ZYX nội tại
`R = Rz(roll)·Ry(yaw)·Rx(pitch)` quanh trục **HCPD**, có nhánh suy biến (gimbal
lock `sy < 1e-6`, tức yaw ±90° → roll ghim 0, phần còn lại dồn vào pitch — đó là
quy ước suy biến chuẩn, không phải phép cứu). **Chỉ để hiển thị** — Kooima dựng
view từ hệ trục màn hình và **vị trí** mắt; hướng đầu không vào phép chiếu. Dấu
tuyệt đối của từng góc thừa hưởng từ trục của `FACE_MODEL_MM`, chưa ai đối chiếu
với thước đo góc thật; chỉ *tính nhất quán* với quaternion là được bảo đảm.

> **Bug đã sửa 2026-09-22.** Trước đó `estimate` gọi `_euler_from_rotation` với ma
> trận **OpenCV thô** trong khi quaternion lấy từ ma trận **đã đổi hệ** — một
> `HeadPoseResult` báo cùng một phép xoay trong hai hệ khác nhau. Liên hợp bởi
> `diag(-1,-1,1)` đưa phép xoay quanh `n` về quanh `M·n`, nên **pitch và yaw đảo
> dấu**, roll thì không: HUD F1 hiện "ngẩng lên" cho đầu chúi xuống, "quay trái"
> cho đầu quay phải. Đo trên `R_cv = Rz(5)Ry(30)Rx(-12)`: cũ `(-12, +30, 5)` →
> mới `(+12, -30, 5)`.
>
> Test `EulerMatchesQuaternionTest` dựng lại `Rz·Ry·Rx` từ ba góc và so với
> `to_rotation_matrix(quaternion)` trên 25 phép xoay — khớp tới **2.2e-16**. Kiểm
> theo cách đó bắt được cả hệ, cả thứ tự, cả lẫn độ/radian, thay vì ghim một ví dụ.

**`HeadPoseResult`:** `pose.rotation`, `pitch/yaw/roll_deg` và `translation_mm`
đều ở hệ **HCPD**. `rotation_matrix` là output **OpenCV thô**, giữ nguyên có chủ
đích làm chẩn đoán — không consumer nào đọc nó.

### 4.5 Eye Position — `src/pose/eye_position.py`

| Hàm / hằng | Công dụng |
|-----|-----------|
| `.__init__(ipd_mm, fx, cx, cy, fy=None, webcam_offset_mm=None)` | `fy` mặc định `= fx`. `webcam_offset_mm` **bắt buộc trên đường sống** |
| `.estimate(head_pose, landmarks, timestamp)` | Nội suy từ neo PnP về phía ước lượng theo mống mắt |
| `.update_intrinsics(fx, cx, cy, fy=None)` | Khi độ phân giải frame đổi |
| `.set_webcam_offset_mm(offset)` | Cập nhật vị trí webcam (mm, trục HCPD) |
| `IRIS_BLEND = 0.12` | **Phân số nội suy** thật: 0 giữ PnP, 1 đáp đúng iris |

solvePnP neo ở **chóp mũi** (gốc của `FACE_MODEL_MM`); người xem thật nhìn từ
**giữa hai mắt**. Ba bước:

```
# 1) un-project mống mắt — trong gốc của CAMERA
d = max(translation_mm[2] - webcam_offset_mm[2], 250)     # Z hệ camera, KHÔNG phải world
iris_cam = [ -(u - cx)·d/fx , -(v - cy)·d/fy ]            # đảo dấu CẢ HAI → đã ở trục HCPD

# 2) về cùng gốc TÂM MÀN HÌNH với translation_mm
iris_world = iris_cam + webcam_offset_mm[:2]

# 3) NỘI SUY, không cộng
eye[:2] += 0.12 · (iris_world - translation_mm[:2])       # = 0.88·PnP + 0.12·iris
```

**Chỉ sửa X và Y — Z truyền nguyên vẹn.** Phép un-project cần một độ sâu để chạy
được, nên nó không thể đồng thời sinh ra một độ sâu.

**Bất biến định nghĩa:** mống mắt rơi đúng tia của điểm PnP ⇒ hai ước lượng trùng
⇒ hiệu chỉnh **bằng 0**. Đo được 3.4e-15 mm trên 10 tư thế. Một phép cộng không
bao giờ bằng 0 được — đó chính là cách cái lỗi dưới đây ẩn mình.

> **Bug đã sửa 2026-09-22.** Bước 3 trước đây là `eye[:2] += 0.12 · iris_cam`,
> tức **cộng 12% của một vị trí tuyệt đối vào 100% của một vị trí tuyệt đối
> khác** — không phải blend. Hai lỗi chồng lên nhau:
>
> | | sai số |
> |---|---|
> | X: hệ số **1.12×** | +12.1 mm tại 100 mm thật; +24.2 mm tại 200 mm |
> | Y: `iris_cam` gốc **thấu kính**, `translation_mm` gốc **tâm màn hình** | hằng số `−0.12 × 103` = **−12.4 mm** ở mọi vị trí |
>
> Bước 1 còn chia `dy` cho `fx` thay vì `fy` — vô hại khi dùng `from_image_size`
> (`fy = fx`) nhưng là lỗi âm thầm đúng `fy/fx − 1` ngay khi có hiệu chuẩn thật.
>
> Test cũ `test_iris_refinement_agrees_with_translation` assert
> `refined_x < base_x` — **chính chữ ký vượt ngưỡng của phép cộng**. Có test, và
> test canh đúng cái sai; đó là lý do lỗi sống sót. Thay bằng `IrisBlendTest`,
> ghim bất biến-0 và tỉ lệ nội suy.

### 4.6 Camera Intrinsics — `src/pose/calibration.py`

| Hàm | Công dụng |
|-----|-----------|
| `CameraIntrinsics.from_image_size(w, h, fov_deg=63)` | **Ước lượng** từ FOV giả định: `fx = fy = w / (2·tan(fov/2))` = **0.8159·w**, `cx = w/2`, `cy = h/2`. Với 1280×720 → **fx = 1044.4** |
| `.from_dict(data, w, h)` | Parse block **đã đo**; **bắt buộc** có `width`/`height` là độ phân giải lúc đo, rồi tỉ lệ hóa về độ phân giải đang chạy. Aspect khác → `ValueError` |
| `load_camera_intrinsics(w, h, path)` | Đọc `camera_calibration.json`; `None` khi không dùng được, và **log lý do** |
| `resolve_intrinsics(w, h)` | Đo được thì dùng, không thì `from_image_size`. **Nguồn duy nhất** cho cả `TrackingWorker` và calibration wizard |
| `resolve_face_scale(intrinsics, path)` | `face_scale` hợp lệ cho **chính K này**, hoặc `None` → về LEGACY |

**Tại sao fx quan trọng, và không chỉ với Z:** solvePnP phục hồi hướng góc của mặt
qua `(u − cx)/fx`. `face_scale` hấp thu được sai số fx trên **Z** (nó là một vô
hướng trên độ sâu), nhưng **không gì** hấp thu được theo phương ngang — với Z đã
đúng thì `x_est = x_true × fx_true/fx_assumed`. Tức fx sai 8% là thị sai ngang sai
8%, đúng cái đại lượng mà cả dự án này tồn tại để làm cho đúng.

**Trạng thái hiện tại:** `assets/config/camera_calibration.json` trong repo ghi
`fx = 960` nhưng **bị từ chối** — nó không khai `width`/`height`, `dist_coeffs`
toàn 0, `cx/cy` đúng tâm ảnh: dấu hiệu số viết tay chứ không phải một lần chạy
`cv2.calibrateCamera` thật. Đường sống vẫn chạy trên **fx = 1044.4 ước lượng từ
FOV 63° giả định**. Thay một fx đoán-bằng-FOV bằng một fx đoán-bởi-người-khác
không phải tiến bộ. Chạy hiệu chuẩn bàn cờ thật, ghi `width`/`height` vào block,
là nó bắt đầu có hiệu lực.

> `DIST_SANE_MM[0] = 120` trong `head_pose.py` **thừa hưởng chính giả định này**
> (dưới 120 mm thì mặt không còn lọt khung ở FOV 63°). Có hiệu chuẩn thật thì
> phải tính lại con số đó.

**Khóa liên động `face_scale ↔ fx`.** `face_scale = D / median(|tvec.z|)` và
`tvec.z` tỉ lệ với fx, nên một `face_scale` đo ở fx này thì sai ở fx khác đúng
theo tỉ số đó — và **sai âm thầm**, vì cổng `[0.6, 1.6]` quá rộng để bắt một cú
lệch 8%. `FaceCalibration.intrinsics_fx` ghi lại fx đã đo cùng (`0.0` = block
trước 2026-09-22).

Đo 2026-09-22 (chiếu `FACE_MODEL_MM` ở `tvec = [-40, 30, 628]`, thêm nhiễu
landmark 2.8 px, giải lại bằng SQPNP):

| fx | `tvec_z` | sau khi × `fx₀/fx` | sai số còn lại |
|---|---|---|---|
| 1044.4 | 624.37 | 624.37 | 0.000% ← mốc |
| 960.0 | 575.90 | 626.53 | **0.345%** |
| 800.0 | 484.10 | 631.99 | **1.220%** |
| 1200.0 | 713.80 | 621.25 | −0.501% |

Nên phép di trú `face_scale × fx_cũ/fx_mới` **không chính xác** — còn dư 0.3–1.2%
(2–7 mm ở 600 mm), vì điểm tối ưu tái chiếu xê dịch dưới nhiễu landmark. Vẫn áp
dụng, và đây là sự đánh đổi có chủ đích: **từ chối** sẽ đẩy người dùng về
`_monocular_scale`, thứ chốt tỉ lệ theo chỗ họ tình cờ ngồi ở khung đầu — một sai
số **không có biên**. 1% có biên thắng vô biên. Hiệu chuẩn lại thì hết cả 1%.

Hai thứ phép nhân vô hướng **không** phủ được, nên chúng **từ chối** thay vì đoán:

- **`fx ≠ fy`** — phép di trú mù với fy. Đo được: 6% lệch fy làm depth sau di trú
  lệch **3.3%** (20 mm ở 600 mm).
- Lệch `cx/cy` thì **không** chặn: ±15 px chỉ làm depth lệch 0.2%.

**`face_scale` đang lưu:** `0.9555` trong `assets/config/user.json`, nguồn
`tracking_benchmark ruler_session_20260714T140310Z`, tái lập bằng
`scripts/calibrate_from_ruler_session.py`, 4 lần chạy cho `0.9555 ± 0.0003`. Block
đó có trước `intrinsics_fx` nên được **giả định** là đo với `from_image_size` —
đó là đường code duy nhất từng tồn tại. Nói ra vì đó là một giả định.

---

## 5. Bộ lọc & dự đoán chuyển động

### 5.1 Pose Filter Pipeline — `src/filters/pipeline.py`

| Hàm | Công dụng |
|-----|-----------|
| `PoseFilterPipeline.update(pose)` | Lọc translation (3 trục) + rotation |
| `extrapolate_for_display(pose, age_s)` | Dự đoán vị trí tương lai theo latency |
| `cycle_filter_type()` | F2: one_euro → kalman → ema → none |

**Các loại filter** (`filter.type` trong config):

| Loại | File | Đặc điểm |
|------|------|----------|
| `one_euro` | `one_euro.py` | Giảm jitter, ít lag khi chuyển động nhanh |
| `kalman` | `kalman.py` | Kalman 1D per axis |
| `ema` | `ema.py` | Exponential moving average |
| `none` | — | Không lọc |

### 5.2 One Euro Filter — `src/filters/one_euro.py`

Thuật toán Casiez et al. (2012):

```
dt = t - t_prev
dx = (x - x_prev) / dt
edx = smooth(dx, dx_prev, dt, d_cutoff)
cutoff = min_cutoff + beta * |edx|
filtered = smooth(x, x_prev, dt, cutoff)

smooth(value, prev, dt, cutoff):
  tau   = 1 / (2π * cutoff)
  alpha = 1 / (1 + tau / dt)
  return alpha * value + (1 - alpha) * prev
```

**Tham số mặc định** (`default.json`): `min_cutoff=1.0`, `beta=0.007`

### 5.3 Motion Predictor — `src/filters/prediction.py`

| Hàm | Công dụng |
|-----|-----------|
| `MotionPredictor.predict(pose, delta_s)` | Extrapolate translation theo velocity |

**Công thức:**
```
velocity = (pose_t - pose_{t-1}) / dt
predicted = translation + velocity * delta_seconds

horizon = min(max(prediction_frames/60, measured_latency_ms/1000), max_prediction_ms/1000)
```

Bù độ trễ pipeline (capture + track + render) để giảm cảm giác lag.

---

## 6. Projection & Virtual Camera

### 6.1 Off-Axis Projection (Robert Kooima)

**File:** `src/math/projection.py`

**Đầu vào (meters):**
- `pa` = góc dưới-trái màn hình
- `pb` = góc dưới-phải
- `pc` = góc trên-trái
- `pe` = vị trí mắt (eye)

**Vector cơ sở màn hình:**
```
vr = normalize(pb - pa)           // hướng phải
vu = normalize(pc - pa)           // hướng lên
vn = normalize(cross(vr, vu))     // pháp tuyến (hướng ra viewer)
```

**Khoảng cách mắt–màn hình:**
```
d_tracked = -dot(pa - pe, vn)
d_effective = max(d_tracked + depth_offset_m, 1e-3)
// depth_offset_m = zoom_offset_mm * 0.001  (phím +/-)
```

**Frustum tại near plane:**
```
left   = near * dot(vr, pa - pe) / d_effective
right  = near * dot(vr, pb - pe) / d_effective
bottom = near * dot(vu, pa - pe) / d_effective
top    = near * dot(vu, pc - pe) / d_effective
```

**Ma trận projection** (`off_axis_projection`):
```
P[0,0] = 2*near / (right - left)
P[1,1] = 2*near / (top - bottom)
P[0,2] = (right + left) / (right - left)
P[1,2] = (top + bottom) / (top - bottom)
P[2,2] = -(far + near) / (far - near)
P[2,3] = -2*far*near / (far - near)
P[3,2] = -1
```

**View matrix** (`_screen_view_matrix`):
```
R = [vr; vu; vn]    // 3×3 rotation
T = translation(-eye)
view = R @ T
```

### 6.2 Virtual Camera — `src/renderer/camera.py`

| Hàm | Công dụng |
|-----|-----------|
| `set_eye_mm(eye)` | Cập nhật vị trí mắt, invalidate cache projection |
| `view_projection_flat()` | Trả `(view, projection)` float32 column-major (có cache) |
| `adjust_zoom(delta_mm)` | Thay đổi `zoom_offset_mm` — chỉ ảnh hưởng Kooima `d`, không đổi pose tracked |
| `debug_frustum_lines()` | 12 cạnh frustum cho debug overlay (F1) |
| `eye_position_m` | Vị trí mắt hiện tại (meters) |

### 6.3 ~~Frustum đơn giản — `src/math/frustum.py`~~ — ĐÃ XOÁ

`src/math/frustum.py` đã xoá 2026-09-22. Nó là phép chiếu lệch trục **thứ hai**,
tính bằng **mm** (`near_mm = 1.0`, `far_mm = 5000`) trong khi đường sống ở §6.1
tính bằng **mét** — trộn hai cái là sai `near/far` âm thầm. Không caller, không
test. Xem §3.2 cho lý do xoá thay vì sửa.

**Đường duy nhất** để dựng frustum là `compute_off_axis_projection` (§6.1).

---

## 7. Rendering & Shader

### 7.1 Scene Renderer — `src/renderer/renderer.py`

**Thứ tự vẽ mỗi frame (`render()`):**

| Bước | Thành phần | Shader |
|------|------------|--------|
| 1 | `glClear` (màu `void_clear_color`) | — |
| 2 | Traverse `SceneNode` → Phong draw | `phong.vert/frag` |
| 3 | `ContentPlaneRenderer.draw()` | `video.vert/frag` hoặc `video_rgba.frag` |
| 4 | Debug frustum wireframe (F1) | `line.vert/frag` |
| 5 | `DebugOverlay` — metrics HUD | `hud.vert/frag` |
| 6 | `WebcamPreviewOverlay` — PiP (F4) | `hud.vert/frag` |
| 7 | `ToastOverlay` — thông báo load media | `hud.vert/frag` |

### 7.2 Shader pipeline

**Thư mục:** `assets/shaders/`

| Cặp shader | Dùng cho | Vertex attributes | Uniforms chính |
|------------|----------|-------------------|----------------|
| `phong.vert/frag` | Scene graph (room, frame, backdrop) | pos(0), normal(1) | uModel, uView, uProjection, uBaseColor, uViewPos, lights, uShininess, uAmbient |
| `video.vert/frag` | Content plane RGB | pos(0), uv(1) | uModel, uView, uProjection, sampler uVideo |
| `video.vert/video_rgba.frag` | Depth layers RGBA | pos(0), uv(1) | + uAlphaThreshold |
| `line.vert/frag` | Debug frustum | pos(0) | uView, uProjection |
| `hud.vert/frag` | HUD overlay | screen-space | — |

**Transform chain (Phong / Video):**
```glsl
gl_Position = uProjection * uView * uModel * vec4(position, 1.0);
```

**Phong lighting (`phong.frag`):**
```
diff = max(dot(N, L), 0)
spec = pow(max(dot(V, reflect(-L, N)), 0), shininess)
color = ambient*base + diff*base*lightColor + spec*lightColor*0.35
// + point light: attenuation = 1 / (1 + 0.09*d + 0.032*d²)
```

**Layer RGBA (`video_rgba.frag`):**
```glsl
vec4 c = texture(uVideo, vUV);
if (c.a < uAlphaThreshold) discard;
FragColor = vec4(c.rgb, c.a);   // alpha blend back-to-front
```

### 7.3 Content Plane Renderer — `src/renderer/content_plane.py`

| Hàm | Công dụng |
|-----|-----------|
| `initialize()` | Tạo GL texture, khởi DepthWorker, prime first frame |
| `load_file(path)` | Hot-swap ảnh/video (F6 / drag-drop) |
| `update(dt_s)` | Tick video frame, poll depth mesh/layers |
| `draw(view, proj)` | Vẽ flat mesh HOẶC N layer RGBA |
| `_prime_and_upload()` | Upload texture GPU + submit depth job |
| `_poll_depth_mesh()` | Nhận depth map → displacement HOẶC clustering |
| `_apply_depth_mesh()` | Tạo `create_displaced_vertical_plane_uv` |
| `_apply_layers()` | Tạo N RGBA texture + flat quad per layer |
| `_draw_layers()` | Alpha blend back-to-front, depth writes off |

### 7.4 Mesh types

| Class | File | Vertex layout | Dùng cho |
|-------|------|---------------|----------|
| `Mesh` | `mesh.py` | pos(3) + normal(3) | Phong scene objects |
| `TexturedMesh` | `textured_mesh.py` | pos(3) + uv(2) | Content plane, video |

### 7.5 Quy ước GPU upload

- Ma trận: **column-major** float32 (`GL_TRUE` transpose flag)
- Texture RGB: `np.flipud()` trước upload (OpenGL origin bottom-left)
- MSAA: `glfw.window_hint(SAMPLES, msaa_samples)` — mặc định 4×

---

## 8. Depth 2.5D: ước lượng, mesh, phân lớp

### 8.1 Pipeline depth tổng quát

```
ContentPlaneRenderer._prime_and_upload()
  → DepthWorker.submit(media_key, rgb)        [background thread]
  → DepthEstimator.estimate(rgb)
      → Depth Anything small HOẶC MiDaS small
      → normalize_depth_map()
  → ContentPlaneRenderer._poll_depth_mesh()
      IF layer_clustering_enabled:
          cluster_into_layers() → N RGBA textures + flat quads
      ELSE:
          create_displaced_vertical_plane_uv() → displaced mesh
```

**Lưu ý:** Video chỉ ước lượng depth frame đầu; chưa có per-frame temporal depth.

### 8.2 Depth Estimator — `src/content/depth_estimator.py`

| Hàm / class | Công dụng |
|-------------|-----------|
| `DepthEstimator(model_name)` | Lazy-load backend |
| `estimate(rgb)` | Trả depth map H×W float32, [0, 1], 1=gần |
| `normalize_depth_map(raw, h, w)` | Percentile clip 2–98%, resize, normalize 0–1 |
| `_MiDaSBackend` | `torch.hub` MiDaS_small |
| `_DepthAnythingBackend` | `transformers` LiheYoung/depth-anything-small-hf |

**Công thức normalize:**
```
lo, hi = percentile(depth, 2), percentile(depth, 98)
depth_norm = clip((depth - lo) / (hi - lo), 0, 1)
// 1.0 = gần viewer, 0.0 = xa
```

### 8.3 Displacement Mesh — `src/scene/primitives.py`

| Hàm | Công dụng |
|-----|-----------|
| `create_displaced_vertical_plane_uv(w, h, depth_map, scale, sx, sy, z_base)` | Lưới sx×sy, mỗi đỉnh offset Z theo depth |
| `sample_depth_map(depth, u, v)` | Bilinear sample depth tại UV (v=0 bottom, v=1 top) |
| `create_vertical_plane_uv(w, h, z)` | Quad phẳng 4 đỉnh + UV |
| `create_vertical_plane_at(cx, cy, w, h, z)` | Quad dịch chuyển (mid panels, frame) |
| `create_window_frame_bezel(...)` | 4 thanh bezel quanh cửa sổ |

**Công thức displacement:**
```
z_vertex = z_base + sample_depth_map(depth, u, v) * depth_scale_meters
// depth=1.0 → đỉnh tiến về viewer (+Z)
```

### 8.4 Layer Clustering — `src/content/layer_clustering.py`

#### Pipeline phân lớp

```
depth map (float 0-1)
  → cluster_depth_layers()     // 1-D k-means
  → labels H×W (0=far, k-1=near)
  → per cluster:
      estimate_atmosphere_color()     // auto từ lớp xa nhất
      apply_atmospheric_perspective() // haze + desaturate
      add_rim_light()                 // chỉ foreground
      extract_layer_rgba()            // feather alpha
  → DepthLayer list (back-to-front)
```

#### Các hàm chính

| Hàm | Công dụng |
|-----|-----------|
| `cluster_depth_layers(depth, n_clusters, seed)` | 1-D k-means trên giá trị depth phẳng |
| `cluster_into_layers(rgb, depth, n_clusters, ...)` | Pipeline đầy đủ → `list[DepthLayer]` |
| `estimate_atmosphere_color(rgb, mask)` | Mean RGB lớp xa nhất — không hardcode |
| `apply_atmospheric_perspective(rgb, depth_norm, atmo, ...)` | Haze + desaturate theo khoảng cách |
| `add_rim_light(rgb, mask, width_px, intensity)` | Viền sáng silhouette foreground |
| `feather_mask_alpha(mask, dilate_px, blur_px)` | Dilate → Gaussian blur → alpha mềm |
| `extract_layer_rgba(rgb, mask, ...)` | RGB đầy đủ + alpha feathered |
| `assign_layer_z(rank, count, z_near, z_far)` | Gán Z world đều theo rank |

#### Công thức k-means 1D

```
centroids = kmeans_1d(depth_flat, k, seed)    // k-means++ init + Lloyd
labels[i] = argmin_j |depth[i] - centroid[j]|
// centroids sorted: 0=far, k-1=near
```

#### Gán Z world

```
t = rank / (layer_count - 1)     // rank 0 = xa nhất
z = z_far + (z_near - z_far) * t
```

#### Atmospheric perspective

```
depth_norm = 1 - rank/(count-1)    // 1.0 = xa, 0.0 = gần

color = lerp(rgb, atmosphere_color, depth_norm * atmosphere_blend)
S_hsv *= (1 - depth_norm * desaturation)
V_hsv += depth_norm * haze_lift * 255
```

**Mặc định:** `atmosphere_blend=0.4`, `desaturation=0.5`, `haze_lift=0.08`

#### Rim light (foreground only)

```
rim = mask - erode(mask, kernel)
out[rim] = lerp(rgb[rim], white, intensity)
```

**Mặc định:** `width_px=2`, `intensity=0.6`

#### Feather alpha

```
alpha = mask * 255
alpha = dilate(alpha, ellipse_kernel)    // đóng khe hở 1-2px
alpha = GaussianBlur(alpha, kernel=5)    // làm mềm biên
```

#### Render layers

```
GL_BLEND = SRC_ALPHA, ONE_MINUS_SRC_ALPHA
GL_DEPTH_MASK = FALSE
for layer in layers (farthest first):
    bind RGBA texture
    draw flat quad at layer.z_m
```

### 8.5 Hai chế độ 2.5D

| Chế độ | Config | Mô tả |
|--------|--------|-------|
| **Displacement mesh** | `layer_clustering_enabled: false` | 1 quad subdivided, Z per-vertex từ depth map |
| **Layer clustering** | `layer_clustering_enabled: true` | N lớp RGBA, alpha blend back-to-front |

### 8.6 Depth Worker — `src/content/depth_worker.py`

| Hàm | Công dụng |
|-----|-----------|
| `submit(media_key, rgb)` | Queue job nếu chưa cache |
| `pop_ready(media_key)` | Lấy depth map hoàn thành |
| `media_key(path)` | Cache key = đường dẫn tuyệt đối file |

Thread daemon, không block render loop.

---

## 9. Scene graph & phân tầng Z

### 9.1 Scene Node — `src/scene/node.py`

| Thành phần | Công dụng |
|------------|-----------|
| `SceneNode` | Transform node + optional mesh + material |
| `Material` | `color` RGB, `shininess` |
| `world_matrix(parent)` | Tích lũy transform cha-con |
| `traverse(parent)` | DFS yield `(node, world_matrix)` |

### 9.2 Video Window Scene — `src/scene/video_scene.py`

`build_video_window_scene(content)` — void room với các lớp Z:

| Layer | Z mặc định (m) | Renderer |
|-------|----------------|----------|
| Backdrop | -2.8 | Phong |
| Mid panel trái/phải | -1.5 | Phong |
| Content plane | -0.75 (`plane_position_m`) | ContentPlaneRenderer |
| Window frame (bezel) | -0.58 | Phong |
| Depth layers (clustering) | 0 … -0.35 | RGBA blend |
| Displaced mesh (local Z) | trên content plane | video.frag |

### 9.3 Room Scene — `src/scene/room.py`

`build_room_scene()` — phòng demo đầy đủ (sàn, tường, đồ nội thất) cho parallax test.

---

## 10. Hệ thống cấu hình

### 10.1 Thứ tự merge

```
default.json (hoặc --config)
  → deep merge user.json
  → screen.json (override screen settings)
  → content.json (load riêng trong app._load_content)
  → ui.json (override ui settings)
```

**Loader:** `src/config/loader.py::merge_user_config`

### 10.2 Các file config

| File | Nội dung |
|------|----------|
| `assets/config/default.json` | Camera, filter, tracking, renderer, optimization |
| `assets/config/user.json` | Override người dùng (tùy chọn) |
| `assets/config/screen.json` | Kích thước màn hình vật lý, camera offset |
| `assets/config/content.json` | Media, depth, layer clustering |
| `assets/config/ui.json` | Webcam PiP settings |
| `assets/config/camera_calibration.json` | Intrinsics đã đo. **Đã wired** qua `resolve_intrinsics` (§4.6) — nhưng block đang ship **bị từ chối** vì không khai `width`/`height` |

### 10.3 ContentSettings — trường quan trọng

| Trường | Mặc định | Đơn vị | Công dụng |
|--------|----------|--------|-----------|
| `enabled` | true | — | Bật content plane |
| `mode` | "video" | — | "video" = void window scene |
| `plane_position_m` | (0, 0, -0.75) | m | Vị trí content plane |
| `plane_width_m` | 1.28 | m | Chiều rộng plane |
| `plane_height_m` | 0 | m | 0 = auto theo aspect media |
| `frame_z_m` | -0.58 | m | Z khung cửa sổ |
| `backdrop_z_m` | -2.8 | m | Z nền xa |
| `depth_enabled` | true | — | Bật depth estimation |
| `depth_scale_meters` | 0.15 | m | Biên độ displacement Z |
| `depth_mesh_resolution` | 96 | — | Lưới displacement (segments) |
| `depth_model` | "depth-anything-small" | — | Model depth |
| `layer_clustering_enabled` | false | — | true = phân lớp, false = displacement |
| `layer_count` | 3 | — | Số cụm k-means (3=fg/mid/bg, 5=multi-object) |
| `layer_z_near_m` | 0.0 | m | Z lớp gần nhất |
| `layer_z_far_m` | -0.35 | m | Z lớp xa nhất |
| `layer_feather_dilate_px` | 1 | px | Dilate mask trước blur |
| `layer_feather_blur_px` | 5 | px | Gaussian blur biên alpha |
| `layer_atmosphere_blend` | 0.4 | — | Hệ số trộn màu khí quyển |
| `layer_desaturation` | 0.5 | — | Giảm saturation theo độ xa |
| `layer_haze_lift` | 0.08 | — | Tăng brightness lớp xa |
| `layer_rim_px` | 2 | px | Độ rộng rim light |
| `layer_rim_intensity` | 0.6 | — | Cường độ rim light |
| `layer_alpha_threshold` | 0.05 | — | Ngưỡng discard alpha (shader) |

### 10.4 Screen calibration — `screen.json`

| Trường | Ví dụ | Đơn vị |
|--------|-------|--------|
| `screen_width` / `screen_height` | 0.344 / 0.194 | meters |
| `screen_center` | [0, 0, 0] | meters |
| `camera_offset` | [0, 0.103, 0.005] | meters → mm trong tracking |
| `eye_offset` | [0, 0, 0] | meters |

### 10.5 Filter & optimization defaults

| Nhóm | Trường quan trọng | Mặc định |
|------|-------------------|----------|
| filter | type, min_cutoff, beta | one_euro, 1.0, 0.007 |
| filter | prediction_enabled, prediction_frames | true, 1.5 |
| optimization | render_extrapolation, fast_solve_pnp | true, true |
| optimization | target_latency_ms, tracking_downscale | 20.0, true |
| adaptive_quality | target_fps, min_fps, quality_levels | 60, 45, [100,75,50,35] |

---

## 11. Phím tắt & điều khiển

| Phím | Handler | Công dụng |
|------|---------|-----------|
| **Esc** | `window.py` | Thoát |
| **F1** | `app.run` | Bật/tắt debug overlay (FPS, latency, pose) |
| **F2** | `app.run` | Chuyển loại filter (one_euro → kalman → ema → none) |
| **F3** | `app.run` | Bật/tắt motion prediction |
| **F4** | `app.run` | Bật/tắt webcam PiP preview |
| **F6** | `app.run` | Mở file dialog chọn ảnh/video |
| **+ / -** | `app.run` | Zoom offset ±5mm (Kooima `d`, không đổi pose) |
| **WASD / QE** | `app.run` | Nudge pose thủ công khi mất face |
| **Drag-drop** | `window.py` | Kéo thả file media vào cửa sổ |

---

## 12. Chỉ mục hàm theo file

### Entry & Engine

| File | Hàm / Class | Công dụng |
|------|-------------|-----------|
| `main.py` | `main()` | Entry point |
| `engine/app.py` | `HcpdApplication` | Điều phối toàn bộ |
| `engine/camera_worker.py` | `CameraWorker` | Thread capture webcam |
| `engine/tracking_worker.py` | `TrackingWorker` | Thread tracking + pose |
| `engine/pose_bus.py` | `PoseBus` | Slot pose thread-safe |
| `engine/profiler.py` | `Profiler` | FPS, latency breakdown |
| `engine/adaptive_quality.py` | `AdaptiveQualityController` | Giảm quality khi FPS thấp |
| `engine/latency_budget.py` | `LatencyBudget` | EMA latency compensation |

### Tracking & Pose

| File | Hàm / Class | Công dụng |
|------|-------------|-----------|
| `tracking/face_tracker.py` | `FaceTracker.process()` | MediaPipe landmarks |
| `tracking/landmarks.py` | `FACE_MODEL_MM`, `PNP_LANDMARKS` | Model 3D/2D cho solvePnP |
| `pose/solve_pnp.py` | `solve_head_pose()` | cv2.solvePnP wrapper |
| `pose/head_pose.py` | `HeadPoseEstimator.estimate()` | PnP → HCPD 6DoF |
| `pose/eye_position.py` | `EyePositionEstimator.estimate()` | Tinh chỉnh iris |
| `pose/calibration.py` | `CameraIntrinsics` | Ma trận K camera |

### Filters

| File | Hàm / Class | Công dụng |
|------|-------------|-----------|
| `filters/pipeline.py` | `PoseFilterPipeline` | Orchestrate filter + prediction |
| `filters/one_euro.py` | `OneEuroFilter` | Adaptive low-pass |
| `filters/kalman.py` | Kalman smoother | Kalman 1D |
| `filters/ema.py` | EMA smoother | Exponential moving average |
| `filters/prediction.py` | `MotionPredictor` | Velocity extrapolation |

### Math

| File | Hàm / Class | Công dụng |
|------|-------------|-----------|
| `math/projection.py` | `compute_off_axis_projection()` | Kooima projection + view |
| `math/projection.py` | `off_axis_projection()` | Ma trận P 4×4 |
| `math/homogeneous.py` | `translation_matrix`, `to_column_major_flat` | Ma trận 4×4 helpers (⚠ tên hàm sai, xem §3.3) |
| `math/quaternion.py` | `from_rotation_matrix`, `to_rotation_matrix` | Quaternion ops — **không có `slerp`** |
| `math/pose_utils.py` | `Pose6DoF`, `interpolate_pose` | Pose data structure + **SLERP** |
| ~~`math/frustum.py`~~ | — | **Đã xoá** 2026-09-22 (§6.3) |
| ~~`math/coordinates.py`~~ | — | **Đã xoá** 2026-09-22 (§3.2) |
| `pose/calibration.py` | `resolve_intrinsics`, `resolve_face_scale` | Nguồn duy nhất cho K và `face_scale` (§4.6) |

### Content & Depth

| File | Hàm / Class | Công dụng |
|------|-------------|-----------|
| `content/depth_estimator.py` | `DepthEstimator`, `normalize_depth_map` | Monocular depth |
| `content/depth_worker.py` | `DepthWorker` | Async depth thread |
| `content/layer_clustering.py` | `cluster_into_layers`, `DepthLayer` | K-means phân lớp |
| `content/layer_clustering.py` | `feather_mask_alpha`, `apply_atmospheric_perspective` | Hiệu ứng biên + haze |
| `content/layer_clustering.py` | `add_rim_light`, `estimate_atmosphere_color` | Rim + auto atmosphere |
| `content/formats.py` | `detect_content_type` | Phát hiện image/video |
| `content/image_source.py` | `ImageSource` | Static image provider |
| `content/cv_io.py` | `imread_bgr` | Unicode-safe image load |
| `content/persistence.py` | `save_content_settings` | Lưu path media vào JSON |

### Scene & Primitives

| File | Hàm / Class | Công dụng |
|------|-------------|-----------|
| `scene/node.py` | `SceneNode`, `Material` | Scene graph node |
| `scene/video_scene.py` | `build_video_window_scene()` | Void + frame + backdrop |
| `scene/room.py` | `build_room_scene()` | Phòng demo đầy đủ |
| `scene/primitives.py` | `create_displaced_vertical_plane_uv` | Depth displacement mesh |
| `scene/primitives.py` | `create_vertical_plane_uv`, `sample_depth_map` | Flat quad + depth sample |
| `scene/primitives.py` | `create_window_frame_bezel` | Khung cửa sổ 4 thanh |

### Renderer

| File | Hàm / Class | Công dụng |
|------|-------------|-----------|
| `renderer/renderer.py` | `SceneRenderer` | Main OpenGL renderer |
| `renderer/camera.py` | `VirtualCamera` | Eye position + Kooima projection |
| `renderer/content_plane.py` | `ContentPlaneRenderer` | Ảnh/video + depth/layers |
| `renderer/shader.py` | `ShaderProgram` | Compile/link GLSL |
| `renderer/mesh.py` | `Mesh` | Phong mesh GPU |
| `renderer/textured_mesh.py` | `TexturedMesh` | UV mesh GPU |
| `renderer/video_player.py` | `VideoPlayer` | OpenCV video decode |
| `renderer/window.py` | `GlfwWindow`, `InputState` | GLFW window + input |
| `renderer/debug_overlay.py` | `DebugOverlay` | F1 metrics HUD |
| `renderer/webcam_preview.py` | `WebcamPreviewOverlay` | PiP webcam + landmarks |
| `renderer/debug_geometry.py` | `DebugGeometryRenderer` | Frustum wireframe |
| `renderer/toast_overlay.py` | `ToastOverlay` | Thông báo load media |

### Config & GUI

| File | Hàm / Class | Công dụng |
|------|-------------|-----------|
| `config/schema.py` | `AppConfig`, `ContentSettings`, ... | Dataclass settings |
| `config/loader.py` | `load_config`, `merge_user_config` | JSON/YAML load + merge |
| `gui/file_dialog.py` | `open_media_dialog()` | F6 native file picker |
| `gui/calibration_wizard.py` | `run_calibration_wizard()` | --calibrate GUI |

### Utility

| File | Công dụng |
|------|-----------|
| `convert_nyu.py` | Chuyển NYU Depth V2 `.mat` → `rgb/` + `depth/` PNG (test data) |

---

## Phụ lục A: Chạy ứng dụng

```powershell
# Cài dependencies
pip install -r requirements.txt

# Chạy app
.\.venv\Scripts\python.exe main.py

# Calibration màn hình/webcam
.\.venv\Scripts\python.exe main.py --calibrate

# Verbose logging
.\.venv\Scripts\python.exe main.py -v
```

## Phụ lục B: Chuyển đổi NYU Depth V2

```powershell
# Download .mat (~2.77 GiB)
curl.exe -L -o nyu_depth_v2_labeled.mat "https://horatio.cs.nyu.edu/mit/silberman/nyu_depth_v2/nyu_depth_v2_labeled.mat"

# Convert test split (654 frames)
.\.venv\Scripts\python.exe convert_nyu.py --mat nyu_depth_v2_labeled.mat --out ./nyu_test
```

## Phụ lục C: Test suite

```powershell
.\.venv\Scripts\python.exe -m pytest tests/ -q
# 76 tests — pose, projection, depth mesh, layer clustering, config, unicode load, ...
```

---

*Tài liệu này phản ánh trạng thái codebase tại thời điểm viết. Cập nhật khi thêm tính năng mới.*
