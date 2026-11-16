"""Tests for C1 disocclusion fill (background synthesis behind silhouettes)."""

from __future__ import annotations

import unittest

import numpy as np

from src.content.disocclusion import (
    DisocclusionParams,
    disocclusion_band,
    fill_layer_disocclusion,
    max_reveal_px,
)


def _scene(size: int = 200, fg_left: int = 100):
    """Blue background, solid red foreground block on the right half."""
    rgb = np.zeros((size, size, 3), dtype=np.uint8)
    rgb[..., 2] = 200                      # blue background
    rgb[:, fg_left:, :] = (220, 30, 30)    # red foreground
    own = np.zeros((size, size), dtype=bool)
    own[:, :fg_left] = True                # far layer owns the left half
    nearer = np.zeros((size, size), dtype=bool)
    nearer[:, fg_left:] = True             # near layer owns the right half
    return rgb, own, nearer


class BandTest(unittest.TestCase):
    """The band is the only strip that can ever be revealed."""

    def test_band_is_inside_nearer_and_outside_own(self) -> None:
        _, own, nearer = _scene()
        band = disocclusion_band(own, nearer, fill_px=20)
        self.assertTrue(bool(band.any()))
        self.assertFalse(bool((band & own).any()))
        self.assertTrue(bool((band & ~nearer).sum() == 0))

    def test_band_width_respects_fill_px(self) -> None:
        _, own, nearer = _scene(size=200, fg_left=100)
        for fill_px in (10, 30, 60):
            band = disocclusion_band(own, nearer, fill_px=fill_px)
            columns = np.nonzero(band.any(axis=0))[0]
            width = int(columns.max() - columns.min() + 1)
            # Euclidean disc of radius fill_px, measured from the silhouette.
            self.assertLessEqual(width, fill_px + 1)
            self.assertGreaterEqual(width, fill_px - 2)

    def test_no_band_without_anything_in_front(self) -> None:
        _, own, _ = _scene()
        empty = np.zeros_like(own)
        self.assertFalse(bool(disocclusion_band(own, empty, fill_px=20).any()))


class ForegroundBleedTest(unittest.TestCase):
    """
    The C3 property, and the reason the whole occluder is masked.

    extract_layer_rgba keeps full RGB at every pixel, so the far layer's array
    still holds the foreground's red pixels. A fill that samples them would
    smear the foreground into the gap — the exact artefact depth-aided
    inpainting exists to prevent.
    """

    def test_fill_takes_background_colour_not_foreground(self) -> None:
        rgb, own, nearer = _scene()
        params = DisocclusionParams(fill_px=25, inpaint_radius=5, feather_px=0)
        filled, alpha, report = fill_layer_disocclusion(rgb, own, nearer, params)

        band = alpha > 0
        self.assertGreater(report.band_px, 0)
        blue = filled[band][:, 2].astype(float).mean()
        red = filled[band][:, 0].astype(float).mean()
        self.assertGreater(blue, red, "band should read as background blue")
        self.assertLess(red, 100.0, "foreground red bled into the gap")

    def test_pixels_outside_the_band_are_untouched(self) -> None:
        rgb, own, nearer = _scene()
        params = DisocclusionParams(fill_px=25, inpaint_radius=5, feather_px=0)
        filled, alpha, _ = fill_layer_disocclusion(rgb, own, nearer, params)
        outside = alpha == 0
        np.testing.assert_array_equal(filled[outside], rgb[outside])


class AlphaRampTest(unittest.TestCase):
    def test_feather_ramps_at_the_outer_rim(self) -> None:
        rgb, own, nearer = _scene()
        params = DisocclusionParams(fill_px=40, inpaint_radius=5, feather_px=10)
        _, alpha, _ = fill_layer_disocclusion(rgb, own, nearer, params)
        band_cols = np.nonzero((alpha > 0).any(axis=0))[0]
        inner, outer = int(band_cols.min()), int(band_cols.max())
        row = alpha.shape[0] // 2
        # Deep inside the band alpha saturates; at the far rim it has decayed.
        self.assertEqual(int(alpha[row, inner + 20]), 255)
        self.assertLess(int(alpha[row, outer]), 255)

    def test_zero_feather_gives_a_hard_band(self) -> None:
        rgb, own, nearer = _scene()
        params = DisocclusionParams(fill_px=30, inpaint_radius=5, feather_px=0)
        _, alpha, _ = fill_layer_disocclusion(rgb, own, nearer, params)
        values = set(np.unique(alpha).tolist())
        self.assertTrue(values <= {0, 255}, f"unexpected alpha values {values}")


class GeometryTest(unittest.TestCase):
    """max_reveal_px sizes fill_px from the optics rather than by guesswork."""

    def test_matches_hand_computation(self) -> None:
        px = max_reveal_px(
            eye_distance_m=0.6,
            plane_z_m=-0.75,
            layer_z_near_m=0.0,
            layer_z_far_m=-0.35,
            plane_width_m=1.28,
            texture_width_px=1866,
            head_shift_mm=100.0,
        )
        self.assertAlmostEqual(px, 37.8, delta=1.0)

    def test_reveal_grows_with_layer_separation(self) -> None:
        def reveal(z_far: float) -> float:
            return max_reveal_px(
                eye_distance_m=0.6, plane_z_m=-0.75, layer_z_near_m=0.0,
                layer_z_far_m=z_far, plane_width_m=1.28,
                texture_width_px=1866, head_shift_mm=100.0,
            )

        self.assertLess(reveal(-0.20), reveal(-0.35))
        self.assertLess(reveal(-0.35), reveal(-0.80))

    def test_reveal_is_linear_in_head_shift(self) -> None:
        def reveal(shift: float) -> float:
            return max_reveal_px(
                eye_distance_m=0.6, plane_z_m=-0.75, layer_z_near_m=0.0,
                layer_z_far_m=-0.35, plane_width_m=1.28,
                texture_width_px=1866, head_shift_mm=shift,
            )

        self.assertAlmostEqual(reveal(200.0), 2.0 * reveal(100.0), places=6)


def _plate_scene(size: int = 320, sq: int = 60):
    """Nền xa phẳng + một ô vuông gần ở giữa, rộng hơn 2 x fill_px."""
    rgb = np.zeros((size, size, 3), dtype=np.uint8)
    rgb[..., 2] = 180
    depth = np.full((size, size), 0.1, dtype=np.float32)
    c = size // 2
    rgb[c - sq:c + sq, c - sq:c + sq] = (220, 40, 40)
    depth[c - sq:c + sq, c - sq:c + sq] = 0.9
    return rgb, depth


class PlateTest(unittest.TestCase):
    """
    Lớp xa nhất là **tấm nền**: sau nó không còn gì, nên nó phải đục toàn phần.

    Trước bản sửa, ``fill_layer_disocclusion`` inpaint cả lỗ nhưng chỉ chép lại
    dải ``fill_px``; ngoài dải alpha = 0 và vùng trống lộ qua. Đo trên 6 ảnh
    thật 2026-10-13: 23,6-71,3% diện tích lớp 0 có alpha < 250.
    """

    def test_plate_layer_is_fully_opaque(self) -> None:
        from src.content.layer_clustering import cluster_into_layers

        rgb, depth = _plate_scene()
        res = cluster_into_layers(
            rgb, depth, n_clusters=2, seed=0,
            disocclusion=DisocclusionParams(enabled=True, fill_px=16),
        )
        self.assertEqual(int(res.layers[0].rgba[..., 3].min()), 255)


class SeamTest(unittest.TestCase):
    """
    Sát đường bao của chính lớp, ``own ∪ band`` phải đục.

    ``_band_alpha`` gieo distance transform trên RIÊNG dải nên mọi mép dải đều
    dốc xuống — kể cả mép áp vào đường bao riêng, nơi alpha riêng cũng đang
    feather. ``np.maximum`` ở khối C1 để lại vệt bán trong suốt chạy dọc đường
    bao, mắt đọc thành viền tối. Đo thật: band_seam_min 23-62 trên 255.
    """

    def test_band_is_opaque_next_to_its_own_silhouette(self) -> None:
        import cv2

        from src.content.layer_clustering import cluster_into_layers

        size = 320
        rgb = np.zeros((size, size, 3), dtype=np.uint8)
        rgb[..., 2] = 180
        depth = np.full((size, size), 0.1, dtype=np.float32)
        rgb[80:240, 80:240] = (40, 200, 40)
        depth[80:240, 80:240] = 0.5
        rgb[130:190, 130:190] = (220, 40, 40)
        depth[130:190, 130:190] = 0.9

        res = cluster_into_layers(
            rgb, depth, n_clusters=3, seed=0,
            disocclusion=DisocclusionParams(enabled=True, fill_px=16, feather_px=8),
        )
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        for rank, layer in enumerate(res.layers[1:], start=1):
            own = layer.mask.astype(bool)
            alpha = layer.rgba[..., 3]
            band = (alpha > 0) & ~own
            near = cv2.dilate(own.astype(np.uint8), kernel) > 0
            seed = np.where(band | own, np.uint8(255), np.uint8(0))
            dist = cv2.distanceTransform(seed, cv2.DIST_L2, 3)
            sel = near & band & (dist > 8.0)
            if not bool(sel.any()):
                continue
            self.assertGreaterEqual(
                int(alpha[sel].min()), 250,
                f"lop {rank}: dai sat duong bao rieng con ban trong suot",
            )


class InpaintSourceTest(unittest.TestCase):
    """
    Màu vật thể phía trước không được làm nguồn cho Telea.

    ``fill_layer_disocclusion`` co ``rgb`` bằng ``INTER_AREA`` (lấy trung bình)
    nhưng co mặt nạ lỗ bằng ``INTER_NEAREST``. Điểm sát biên trong ảnh đã co vì
    thế là hỗn hợp nền + vật mà vẫn nằm NGOÀI lỗ, nên Telea coi là nguồn hợp lệ
    và kéo màu vật vào trong. Đo trên ảnh thật 2026-10-13: chỗ lấp trong lỗ hải
    đăng (t23) ngả trắng kèm chóp tối, dù trời quanh đó mượt.

    ``max_work_dim = 64`` ép tỉ lệ co xuống 0.213x để biên lai đủ rộng mà bắt;
    trên ảnh thật tỉ lệ là 0.751x nên biên mỏng hơn — xem cột ``fill_leak_dL``.
    """

    def test_fill_does_not_copy_the_foreground_colour(self) -> None:
        import cv2

        size, half = 300, 60
        rgb = np.full((size, size, 3), 128, dtype=np.uint8)
        own = np.ones((size, size), dtype=bool)
        nearer = np.zeros((size, size), dtype=bool)
        c = size // 2
        rgb[c - half:c + half, c - half:c + half] = 255
        nearer[c - half:c + half, c - half:c + half] = True
        own &= ~nearer

        filled, _, _ = fill_layer_disocclusion(
            rgb, own, nearer,
            DisocclusionParams(enabled=True, fill_px=16, max_work_dim=64),
            plate=True,
        )
        hole = nearer & ~own
        h8 = np.where(hole, np.uint8(255), np.uint8(0))
        rim = hole & (cv2.distanceTransform(h8, cv2.DIST_L2, 3) <= 6.0)
        mean = float(filled[rim].astype(np.float64).mean())
        self.assertAlmostEqual(
            mean, 128.0, delta=6.0,
            msg=f"mep lo lap ra {mean:.1f}, lech khoi nen 128 -> mau vat da ro ri",
        )


class PlateHaloTest(unittest.TestCase):
    """
    Tấm nền không được giữ lại viền của vật đã trượt đi.

    Điểm ngay NGOÀI lỗ là của chính tấm nền, nhưng chúng là hỗn hợp vật + nền
    (mép anti-alias, và mép depth lệch vài px). Chúng không được inpaint vì nằm
    ngoài lỗ, cũng không bị thay vì chỉ ``zone`` được chép về — nên khi vật
    trượt khỏi lỗ, viền 1-3 px của nó vẫn nằm đó. Ảnh chụp 2026-10-13 10:07 ở
    mắt (−14, 61, 333) và (−131, 54, 319): vạch tối dọc đèn hải đăng, vạch sáng
    dọc thân tháp.
    """

    def test_no_object_outline_is_left_just_outside_the_hole(self) -> None:
        import cv2

        size, half = 300, 60
        rgb = np.full((size, size, 3), 128, dtype=np.uint8)
        c = size // 2
        rgb[c - half:c + half, c - half:c + half] = 255
        rgb = cv2.GaussianBlur(rgb, (5, 5), 0)      # mép nhoè như anti-alias
        nearer = rgb[..., 0] > 191
        own = ~nearer

        filled, _, _ = fill_layer_disocclusion(
            rgb, own, nearer,
            DisocclusionParams(enabled=True, fill_px=16, max_work_dim=64),
            plate=True,
        )
        n8 = np.where(nearer, np.uint8(255), np.uint8(0))
        ring = own & (cv2.distanceTransform(255 - n8, cv2.DIST_L2, 3) <= 2.0)
        mean = float(filled[ring].astype(np.float64).mean())
        self.assertAlmostEqual(
            mean, 128.0, delta=6.0,
            msg=f"vanh ngoai lo ra {mean:.1f}, lech khoi nen 128 -> con vien vat",
        )


class DisabledPathTest(unittest.TestCase):
    def test_cluster_into_layers_unchanged_when_disabled(self) -> None:
        from src.content.layer_clustering import cluster_into_layers

        rng = np.random.default_rng(0)
        rgb = rng.integers(0, 255, (64, 64, 3), dtype=np.uint8)
        depth = np.tile(np.linspace(0.0, 1.0, 64, dtype=np.float32), (64, 1))

        without = cluster_into_layers(rgb, depth, n_clusters=3, seed=0)
        disabled = cluster_into_layers(
            rgb, depth, n_clusters=3, seed=0,
            disocclusion=DisocclusionParams(enabled=False),
        )
        for a, b in zip(without.layers, disabled.layers):
            np.testing.assert_array_equal(a.rgba, b.rgba)


if __name__ == "__main__":
    unittest.main()
