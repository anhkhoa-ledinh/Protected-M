"""OpenCV solvePnP wrapper for head pose estimation."""

from __future__ import annotations

import cv2
import numpy as np
import numpy.typing as npt

from src.pose.calibration import CameraIntrinsics


def solve_head_pose(
    model_points_mm: npt.NDArray[np.float64],
    image_points_px: npt.NDArray[np.float64],
    intrinsics: CameraIntrinsics,
    fast: bool = True,
) -> tuple[bool, npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """
    Estimate head rotation and translation via solvePnP.

    Returns:
        success, rotation_vector (3,), translation_vector (3,) in mm.
    """
    if model_points_mm.shape[0] < 4 or image_points_px.shape[0] < 4:
        return False, np.zeros(3), np.zeros(3)

    # SQPNP thay EPNP (Phase B, đo ngày 2026-09-19).
    #
    # Lý do gốc: FACE_MODEL_MM gần đồng phẳng (coplanar). Phân rã SVD của 6
    # điểm sau khi dời về trọng tâm cho giá trị kỳ dị 89.21 / 76.49 / 23.35,
    # tỉ lệ phẳng s3/s1 = 0.2617 — bề dày 27.4 mm trên đường kính 126.3 mm, và
    # phần lớn bề dày đó chỉ do chóp mũi nhô ra 20.2 mm. Với dữ liệu gần phẳng,
    # bài toán PnP có nghiệm gương (mirror solution) và trở nên điều kiện xấu;
    # EPnP yếu đúng ở cấu hình này vì 4 điểm điều khiển ảo của nó suy biến theo.
    #
    # Đo trên 1200 khung landmark THẬT của máy này (artifacts/landmarks.npz,
    # dải 142-840 mm, yaw ±60°), phát lại qua scripts/compare_solvers.py:
    #
    #   bien the            tz<0      nghiem rac   reproj    jitter3D   ms
    #   EPNP (truoc)        0.0%      0.0%         14.08px   5.3mm      0.116
    #   ITERATIVE          18.6%      0.0%         10.07px   4.5mm      1.65
    #   ITERATIVE + seed    0.0%      0.0%         10.34px   4.9mm      1.18
    #   SQPNP (sau)         0.0%      0.0%         10.42px   4.8mm      0.134
    #
    # Cột nghiem rac dung DIST_SANE_MM = (120, 1500). Ban do dau tien chay voi
    # can duoi 200 mm va bao 4.8% cho EPNP — toan bo la BAO DONG GIA tren khung
    # nguoi dung ghe sat that (58 khung gom thanh 5 doan lien tuc, be rong mat
    # dat 625 px, fx*E/625 = 150 mm khop solvePnP trong 6%). Ha can duoi xuong
    # 120 thi ca bon bien the deu ve 0.0%.
    #
    # Bảng trên không đo được độ sâu, nên đo thêm bằng tham chiếu độc lập
    # (tam giác đồng dạng fx*90/bề_rộng_mắt, 220 khung gần chính diện):
    #
    #   solver   sai lech trung vi   rung Z luc ngoi yen (p95)
    #   EPNP     -28.5 mm            10.06 mm
    #   SQPNP    -18.1 mm             4.63 mm
    #
    # SQPNP thắng ở reproj (-26%), sai lệch độ sâu (-36%) và rung Z lúc ngồi
    # yên (-54%). Thua duy nhất ở tốc độ: +0.019 ms/khung = 0.11% ngân sách
    # 16.67 ms của một khung 60 fps.
    #
    # ITERATIVE là BẪY, đừng chọn dù reproj của nó thấp nhất: 18.6% khung trả
    # tz<=0, và độ lệch chuẩn độ sâu 229 mm so với 19.5 mm của EPNP. Reproj
    # thấp không chứng minh nghiệm đúng — nghiệm gương khớp ảnh rất khớp.
    #
    # fast=False vẫn giữ ITERATIVE để còn A/B so sánh được.
    flags = cv2.SOLVEPNP_SQPNP if fast else cv2.SOLVEPNP_ITERATIVE
    success, rvec, tvec = cv2.solvePnP(
        model_points_mm.astype(np.float64),
        image_points_px.astype(np.float64),
        intrinsics.matrix,
        intrinsics.dist_coeffs,
        flags=flags,
    )
    if not success:
        return False, np.zeros(3), np.zeros(3)

    return True, rvec.ravel(), tvec.ravel()
