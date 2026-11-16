"""
Kiểm phần TÍNH TOÁN của quét chiều sâu lúc chạy (Phase 2).

Không kiểm phần bấm phím — cái đó cần GLFW và một con người. Kiểm cái mà
người quét sẽ ĐỌC để ra quyết định: tỉ số thị sai, tỉ lệ nội dung lọt khung,
và cảnh báo nội dung vượt ra trước khung bezel.

Dựng instance bằng object.__new__ vì __init__ thật cần context OpenGL và nạp
media. _log_depth_state chỉ đụng vài thuộc tính, gán đủ chúng là chạy được.
"""

from __future__ import annotations

import logging
import unittest

import numpy as np

from src.config.schema import ContentSettings
from src.renderer.content_plane import ContentPlaneRenderer

EYE_D_MM = 600.0
SCREEN_W_MM = 344.0
SCREEN_H_MM = 194.0


def _plane(
    plane_z: float, z_far: float = -0.35, e_max: float = 0.0
) -> ContentPlaneRenderer:
    """
    Content plane tối thiểu, chỉ đủ cho _log_depth_state.

    ``e_max`` mặc định 0.0 (TẮT overscan) để các test hình học nền đo đúng
    hình học gốc. Test overscan tự truyền giá trị của nó.
    """
    plane = object.__new__(ContentPlaneRenderer)
    content = ContentSettings()
    content.layer_z_near_m = 0.0
    content.layer_z_far_m = z_far
    content.overscan_e_max_mm = e_max
    plane._content = content
    plane._position = np.array([0.0, 0.0, plane_z], dtype=np.float64)
    plane._layers = [object(), object(), object()]   # K = 3
    # Kich thuoc quad lop THAT: be rong co dinh 1.28 m, chieu cao suy tu ti le
    # anh (t21.png 736x552 -> 4:3). _resize_plane dat hai gia tri nay luc chay.
    plane._plane_width = 1.28
    plane._plane_height = 1.28 / (736.0 / 552.0)
    return plane


def _capture(plane: ContentPlaneRenderer, case: unittest.TestCase) -> str:
    """Chạy _log_depth_state và trả về dòng log nó ghi ra."""
    with case.assertLogs("src.renderer.content_plane", level=logging.INFO) as cm:
        plane._log_depth_state("test", EYE_D_MM, SCREEN_W_MM, SCREEN_H_MM)
    return cm.output[0]


class ParallaxRatioTest(unittest.TestCase):
    """Tỉ số thị sai phải khớp số của scripts/measure_parallax.py."""

    def test_baseline_ratio_matches_the_measured_table(self) -> None:
        # plane_z = -0.75, z_far = -0.35 -> bang Phase 1 cho 1.165x
        self.assertIn("thi sai xa/gan=1.165x", _capture(_plane(-0.75), self))

    def test_pulling_the_plane_closer_widens_the_ratio(self) -> None:
        # Day la ly do plane_z la can gat CHINH: 1.165x -> 1.913x
        self.assertIn("thi sai xa/gan=1.913x", _capture(_plane(-0.20), self))

    def test_pushing_the_far_layer_back_is_the_weaker_lever(self) -> None:
        # z_far -2.50 la mot buoc RAT lon, van thua plane_z -0.20.
        line = _capture(_plane(-0.75, z_far=-2.50), self)
        self.assertIn("thi sai xa/gan=1.519x", line)


class InFrameRatioTest(unittest.TestCase):
    """
    Tỉ lệ phủ khung — chỉ số PHỤ, và cao KHÔNG phải tốt.

    Cửa sổ màn hình mở một hình nón từ mắt; ở độ sâu t nó cắt ra dải rộng
    screen_w * (D + t) / D. 100% nghĩa là cả ảnh lọt trong khung nhìn, tức CÓ
    VÙNG TRỐNG quanh nó — hỏng. Chỉ số để quyết định là VoidOnsetTest.
    """

    def test_closer_plane_shows_less_content(self) -> None:
        far = _capture(_plane(-0.75), self)
        near = _capture(_plane(-0.20), self)

        def pct(line: str) -> float:
            token = line.split("anh phu ")[1]
            return float(token.split("%")[0])

        self.assertGreater(pct(far), pct(near))

    def test_in_frame_ratio_matches_the_cone_formula(self) -> None:
        # t = 750 mm, D = 600 -> dai nhin thay = 344 * 1350/600 = 774 mm
        # noi dung rong 1280 mm -> 774/1280 = 60.5% -> lam tron 60%
        self.assertIn("anh phu 60%", _capture(_plane(-0.75), self))

    def test_content_narrower_than_the_cone_reads_100(self) -> None:
        plane = _plane(-3.0)
        plane._plane_width = 0.20                # hep hon dai nhin thay
        self.assertIn("anh phu 100%", _capture(plane, self))


class BezelWarningTest(unittest.TestCase):
    """Nội dung ra trước khung bezel thì phá ẩn dụ cửa sổ — phải báo."""

    def test_no_warning_behind_the_bezel(self) -> None:
        # bezel mac dinh -0.58; noi dung o -0.75 nam SAU no.
        self.assertNotIn("CANH BAO", _capture(_plane(-0.75), self))

    def test_warning_when_content_crosses_the_bezel(self) -> None:
        # -0.20 nam TRUOC bezel -0.58.
        self.assertIn("RA TRUOC khung bezel", _capture(_plane(-0.20), self))

    def test_warning_boundary_is_the_configured_bezel_z(self) -> None:
        plane = _plane(-0.75)
        plane._content.frame_z_m = -0.90         # day bezel ra sau noi dung
        self.assertIn("RA TRUOC khung bezel", _capture(plane, self))


class VoidOnsetTest(unittest.TestCase):
    """
    Ngưỡng lộ vùng trống — CHỈ SỐ ĐỂ QUYẾT ĐỊNH.

    Ở độ sâu t, vùng nhìn thấy trên mặt phẳng nội dung trải
    ±(w/2)·(D+t)/D − e·t/D, nên ngưỡng là
    e_void = (W − (w/2)·(D+t)/D)·D/t với W là nửa bề rộng nội dung.
    """

    def test_reports_the_layer_that_runs_out_first(self) -> None:
        # Lop xa nhat (t = 1100 mm) cham truoc: 83 mm, khop
        # scripts/measure_parallax.py.
        self.assertIn("VOID TU +-83 mm", _capture(_plane(-0.75), self))

    def test_farther_layers_run_out_sooner(self) -> None:
        # Lop cang xa truot cang nhieu -> nguong cang nho. Day chinh la ly do
        # overscan phai tinh THEO TUNG LOP chu khong mot he so chung.
        def onset(t_mm: float, half_content: float, half_screen: float) -> float:
            return (half_content - half_screen * (EYE_D_MM + t_mm) / EYE_D_MM)                 * EYE_D_MM / t_mm

        near = onset(750.0, 640.0, SCREEN_W_MM / 2)
        far = onset(1100.0, 640.0, SCREEN_W_MM / 2)
        self.assertGreater(near, far)

    def test_pulling_the_plane_closer_buys_head_room(self) -> None:
        # Keo gan lam t nho lai -> nguong void LON hon. Cung chieu voi can gat
        # chinh cua Phase 2, nen hai muc tieu khong xung dot.
        def onset_mm(line: str) -> float:
            return float(line.split("VOID TU +-")[1].split(" mm")[0])

        self.assertGreater(
            onset_mm(_capture(_plane(-0.40), self)),
            onset_mm(_capture(_plane(-0.75), self)),
        )

    def test_sixteen_by_nine_binds_on_the_vertical_axis(self) -> None:
        # Anh 16:9 thap hon -> chieu doc cham truoc, nguong tut con 46 mm.
        plane = _plane(-0.75)
        plane._plane_height = 1.28 / 1.778
        self.assertIn("VOID TU +-46 mm", _capture(plane, self))


class OverscanTest(unittest.TestCase):
    """Overscan theo từng lớp phải đẩy ngưỡng void lên đúng e_max."""

    def test_overscan_reaches_the_requested_e_max(self) -> None:
        # 83 mm (tat) -> 150 mm (bat). Chinh xac bang e_max vi he so duoc
        # tinh tu cung mot cong thuc.
        self.assertIn("VOID TU +-150 mm", _capture(_plane(-0.75, e_max=150.0), self))

    def test_disabled_keeps_the_original_geometry(self) -> None:
        # 0.0 = tat han, quad dung bang plane_width_m.
        self.assertIn("VOID TU +-83 mm", _capture(_plane(-0.75, e_max=0.0), self))

    def test_farther_layers_get_a_larger_factor(self) -> None:
        # Day la ly do he so phai THEO TUNG LOP: lop xa truot nhieu hon.
        plane = _plane(-0.75, e_max=150.0)
        near = plane._overscan_scale(750.0, 1.28, 0.96, SCREEN_W_MM, SCREEN_H_MM)
        far = plane._overscan_scale(1100.0, 1.28, 0.96, SCREEN_W_MM, SCREEN_H_MM)
        self.assertGreater(far, near)
        self.assertAlmostEqual(far, 1.191, places=3)
        # Lop GAN khong can phong gi: nua rong can 574 mm < 640 mm co san.
        self.assertEqual(near, 1.0)
        mid = plane._overscan_scale(925.0, 1.28, 0.96, SCREEN_W_MM, SCREEN_H_MM)
        self.assertAlmostEqual(mid, 1.044, places=3)

    def test_scale_is_uniform_so_aspect_is_preserved(self) -> None:
        # Mot he so chung cho ca X va Y. Co gian khac nhau se keo dan anh.
        plane = _plane(-0.75, e_max=150.0)
        k = plane._overscan_scale(1100.0, 1.28, 0.96, SCREEN_W_MM, SCREEN_H_MM)
        w0, h0 = plane._plane_width, plane._plane_height
        self.assertAlmostEqual((w0 * k) / (h0 * k), w0 / h0, places=9)

    def test_never_shrinks_below_one(self) -> None:
        # Noi dung da du rong thi khong duoc thu nho lai.
        plane = _plane(-0.75, e_max=150.0)
        self.assertEqual(
            plane._overscan_scale(1100.0, 10.0, 7.5, SCREEN_W_MM, SCREEN_H_MM), 1.0
        )


if __name__ == "__main__":
    unittest.main()
