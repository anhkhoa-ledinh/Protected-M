r"""
TASK_game_1110 F1 — hộp trúng là **cả con bọ**, không phải vòng tròn 40 px.

Sprite tổng hợp trong test cố ý có **chân mảnh** chìa ra bốn phía: đó chính
là chỗ mà cách cũ luôn báo hụt (sprite vẽ ở ``radius × 3`` nên đầu mút chân
nằm ngoài bán kính 40 px quanh tâm chiếu).

Ba cổng của kịch bản:

- 100 % trúng trên ≥ 200 điểm lấy trên hình bóng, kể cả đầu mút chân;
- 0 % trúng ở các điểm cách hình bóng ≥ 4 px;
- 0 % trúng ở điểm trên hình bóng mà một lớp gần hơn đang che (alpha > 0,5).
"""

from __future__ import annotations

import unittest

import numpy as np

from src.content.hidden_props import LayerPlane
from src.game import hit as H

#: Mắt nghỉ của kịch bản L0/F1 và tấm kính toàn màn.
EYE = (0.0, 0.0, 600.0)
GLASS = (344.0, 194.0)
#: Con bọ nằm sau kính 180 mm, bán kính 12 mm, sprite phóng 3 lần (bugs.py).
PROP = (0.0, 0.0, -180.0)
RADIUS_MM = 12.0
SIZE_FACTOR = 3.0
SPRITE = 128


def _bug_sprite(size: int = SPRITE) -> np.ndarray:
    """
    Con bọ tổng hợp: thân bầu dục + 4 chân mảnh 3 px chìa ra sát mép.

    Chân mới là phần đáng kể: chúng mỏng và xa tâm, nên mọi hộp trúng "tròn
    quanh tâm" đều bỏ sót chúng.
    """
    rgba = np.zeros((size, size, 4), np.uint8)
    yy, xx = np.mgrid[0:size, 0:size]
    cy = cx = (size - 1) / 2.0
    body = ((xx - cx) / (size * 0.20)) ** 2 + ((yy - cy) / (size * 0.30)) ** 2 <= 1.0
    legs = np.zeros_like(body)
    half = max(1, size // 42)                      # chân dày ~3 px ở 128
    legs[int(cy) - half:int(cy) + half + 1, 2:size - 2] = True
    legs[2:size - 2, int(cx) - half:int(cx) + half + 1] = True
    mask = body | legs
    rgba[..., :3] = 200
    rgba[mask, 3] = 255
    return rgba


def _silhouette_points(mask: H.SpriteMask, quad: H.SpriteQuad, count: int):
    """``count`` điểm thiết kế rải đều trên hình bóng, ưu tiên cả đầu mút chân."""
    rows, cols = np.nonzero(mask.alpha > 64)
    step = max(1, rows.size // count)
    sprite_h, sprite_w = mask.alpha.shape
    out = []
    for row, col in zip(rows[::step], cols[::step]):
        u = col / (sprite_w - 1)
        v = row / (sprite_h - 1)
        out.append((quad.centre_x - quad.half_w + u * 2.0 * quad.half_w,
                    quad.centre_y - quad.half_h + v * 2.0 * quad.half_h))
    return out


class HitBoxTest(unittest.TestCase):
    def setUp(self) -> None:
        self.mask = H.SpriteMask.from_rgba(_bug_sprite())
        self.quad = H.sprite_quad(PROP, RADIUS_MM, SIZE_FACTOR, EYE, GLASS)
        self.assertIsNotNone(self.quad)

    def test_the_quad_is_much_wider_than_the_old_forty_pixel_circle(self) -> None:
        # Nếu không thì cả bài test này không chứng minh được gì: vòng cũ đã
        # đủ to và lỗi người dùng báo hẳn ở chỗ khác.
        self.assertGreater(self.quad.half_w * 2.0, 40.0)

    def test_every_silhouette_point_is_a_hit(self) -> None:
        points = _silhouette_points(self.mask, self.quad, 240)
        self.assertGreaterEqual(len(points), 200)
        misses = [p for p in points if not H.hits_sprite(self.mask, self.quad, p)]
        self.assertEqual(misses, [], f"{len(misses)}/{len(points)} diem hut")

    def test_leg_tips_are_hits(self) -> None:
        # Bốn đầu mút chân, sát mép quad — chỗ mà vòng tròn cũ chắc chắn hụt.
        edge_u = 3.5 / SPRITE
        tips = [(0.5, edge_u), (0.5, 1.0 - edge_u),
                (edge_u, 0.5), (1.0 - edge_u, 0.5)]
        for u, v in tips:
            point = (self.quad.centre_x - self.quad.half_w + u * 2 * self.quad.half_w,
                     self.quad.centre_y - self.quad.half_h + v * 2 * self.quad.half_h)
            with self.subTest(uv=(u, v)):
                self.assertTrue(H.hits_sprite(self.mask, self.quad, point))

    def test_points_four_pixels_outside_are_misses(self) -> None:
        """
        Quét **cả** ô quad rồi giữ đúng những điểm cách hình bóng ≥ 4 px.

        Lọc theo khoảng cách thật chứ không "dịch 4 px từ một điểm trên thân":
        thân con bọ dày, dịch 4 px từ giữa thân vẫn rơi vào thân, nên cách kia
        chỉ chấm được vài điểm và không chứng minh gì.
        """
        sprite_w, sprite_h = self.mask.size
        per_design_x = sprite_w / (2.0 * self.quad.half_w)
        per_design_y = sprite_h / (2.0 * self.quad.half_h)
        tested = hits = 0
        for row in range(0, sprite_h, 2):
            for col in range(0, sprite_w, 2):
                if self.mask.dist[row, col] < 4.0 * min(per_design_x,
                                                        per_design_y):
                    continue
                point = (self.quad.centre_x - self.quad.half_w
                         + col / (sprite_w - 1) * 2.0 * self.quad.half_w,
                         self.quad.centre_y - self.quad.half_h
                         + row / (sprite_h - 1) * 2.0 * self.quad.half_h)
                tested += 1
                hits += bool(H.hits_sprite(self.mask, self.quad, point))
        self.assertGreater(tested, 100, "khong du diem ngoai de cham")
        self.assertEqual(hits, 0, f"{hits}/{tested} diem ngoai van bao trung")

    def test_the_top_third_is_a_crit(self) -> None:
        top = (self.quad.centre_x, self.quad.centre_y - self.quad.half_h * 0.8)
        low = (self.quad.centre_x, self.quad.centre_y + self.quad.half_h * 0.8)
        self.assertTrue(H.is_crit(self.quad, top))
        self.assertFalse(H.is_crit(self.quad, low))


class BlockedByNearerLayerFunctionTest(unittest.TestCase):
    """
    Hàm thuần ``blocked_by_nearer_layer`` vẫn đúng hình học của nó — chỉ không
    còn được ``pick_target`` gọi lại (xem ``NoDoubleOcclusionTest`` dưới).
    """

    def _plane(self, alpha_value: int, z_m: float = -0.09) -> LayerPlane:
        alpha = np.full((64, 64), np.uint8(alpha_value))
        return LayerPlane(alpha=alpha, z_m=z_m, quad_w_m=1.0, quad_h_m=1.0,
                          centre_x_m=0.0, centre_y_m=0.0,
                          z_min_m=z_m, z_max_m=z_m)

    def test_an_opaque_nearer_plane_blocks_the_centre(self) -> None:
        centre = (640.0, 360.0)
        self.assertTrue(H.blocked_by_nearer_layer(
            (self._plane(255),), EYE, centre, PROP[2], GLASS))

    def test_a_farther_plane_never_blocks(self) -> None:
        centre = (640.0, 360.0)
        self.assertFalse(H.blocked_by_nearer_layer(
            (self._plane(255, z_m=-0.50),), EYE, centre, PROP[2], GLASS))


class NoDoubleOcclusionTest(unittest.TestCase):
    """
    TASK_fix_0130 Phase 1 — hồi quy F1: con bọ ``is_exposed`` cho vẽ ra thì
    phải bắn trúng được trên MỌI điểm đã vẽ, không bị kiểm che một lần nữa.

    ``BugPainter.draw`` vẽ bọ sau cảnh và KHÔNG test depth với các lớp (xem
    ``bugs.py``), nên một con đã qua cổng ``is_exposed`` luôn hiện trên hình
    bất kể lớp nào đứng "gần" nó về mặt toạ độ. Trước bản sửa,
    ``pick_target`` gọi lại ``blocked_by_nearer_layer`` tại ĐÚNG điểm ngắm —
    một tia hẹp hơn hẳn phép 5 điểm của ``is_exposed`` — nên bấm đúng thân,
    kể cả chân, vẫn báo hụt. Test này THẤT BẠI trên code trước bản sửa.
    """

    def setUp(self) -> None:
        self.mask = H.SpriteMask.from_rgba(_bug_sprite())
        self.quad = H.sprite_quad(PROP, RADIUS_MM, SIZE_FACTOR, EYE, GLASS)
        self.bugs = {0: (PROP, RADIUS_MM, SIZE_FACTOR, self.mask)}
        # Lớp đục hoàn toàn, GẦN hơn con bọ, phủ kín cả màn — đúng kịch bản
        # "quad của con bọ đang lộ trùng lên một lớp đục gần hơn trong khung
        # hình" mà Phase 1 bước 3 đòi.
        self.nearer_opaque = (LayerPlane(
            alpha=np.full((64, 64), np.uint8(255)), z_m=-0.09,
            quad_w_m=1.0, quad_h_m=1.0, centre_x_m=0.0, centre_y_m=0.0,
            z_min_m=-0.09, z_max_m=-0.09),)

    def test_hits_the_body_despite_the_overlapping_nearer_layer(self) -> None:
        centre = (self.quad.centre_x, self.quad.centre_y)
        hit, _crit = H.pick_target(self.bugs, centre, EYE,
                                   self.nearer_opaque, GLASS)
        self.assertEqual(hit, 0)

    def test_hits_a_leg_tip_despite_the_overlapping_nearer_layer(self) -> None:
        edge_u = 3.5 / SPRITE
        point = (self.quad.centre_x - self.quad.half_w + edge_u * 2 * self.quad.half_w,
                 self.quad.centre_y)
        hit, _crit = H.pick_target(self.bugs, point, EYE,
                                   self.nearer_opaque, GLASS)
        self.assertEqual(hit, 0)

    def test_every_silhouette_point_still_hits_with_a_nearer_opaque_layer(self) -> None:
        points = _silhouette_points(self.mask, self.quad, 240)
        self.assertGreaterEqual(len(points), 200)
        misses = [p for p in points
                 if H.pick_target(self.bugs, p, EYE, self.nearer_opaque,
                                  GLASS)[0] < 0]
        self.assertEqual(misses, [], f"{len(misses)}/{len(points)} diem hut "
                         f"du lop gan duc da duoc 'is_exposed' cho qua")


class ShootableSetMatchesDrawnSetTest(unittest.TestCase):
    """
    TASK_hit_0200 — tập bắn được phải bằng tập vẽ ra.

    ``BugPainter.draw`` (``src/ui_album/bugs.py``, gọi qua
    ``app.py:341``/``341`` với nguyên ``self._game.props``) vẽ MỌI con chưa
    diệt, không hỏi ``is_exposed``. Trước bản sửa, ``app._aim_context`` chỉ
    đưa vào ``bugs`` những con ``is_exposed`` — một con đang vẽ (``found``
    còn ``False``) mà ``visible_since_s`` vẫn ``None`` (mới hiện khung này,
    hoặc do kẹt khung) thì không nằm trong tập bắn được, bắn trúng thân vẫn
    báo hụt. Test này THẤT BẠI trên code trước bản sửa.
    """

    def _app(self):
        from types import SimpleNamespace

        from src.engine.app import HcpdApplication

        app = HcpdApplication.__new__(HcpdApplication)
        mask = H.SpriteMask.from_rgba(_bug_sprite())
        app._bugs = SimpleNamespace(sprite_mask=lambda level: mask)
        app._virtual_camera = SimpleNamespace(eye_position_mm=np.array(EYE), effective_eye_position_mm=np.array(EYE))
        app._glass_mm = lambda: (GLASS, (0.0, 0.0))
        app._content = SimpleNamespace(game_prop_radius_mm=RADIUS_MM,
                                       layer_alpha_threshold=0.05)
        app._album = SimpleNamespace(state=SimpleNamespace(level=1))
        app._game = SimpleNamespace(
            props=[{"pos_mm": PROP, "radius_mm": RADIUS_MM}],
            found=[False], visible_since_s=[None])     # chua "is_exposed"
        app._game_planes = ()
        return app

    def test_hits_the_body_of_an_unexposed_but_drawn_bug(self) -> None:
        ctx = self._app()._aim_context()
        self.assertIn(0, ctx.bugs, "con bo chua is_exposed phai van o trong "
                                  "tap ban duoc — no van dang duoc VE ra")
        quad = H.sprite_quad(PROP, RADIUS_MM, SIZE_FACTOR, EYE, GLASS)
        hit, _crit = ctx.resolve((quad.centre_x, quad.centre_y))
        self.assertEqual(hit, 0)

    def test_hits_a_leg_tip_of_an_unexposed_but_drawn_bug(self) -> None:
        ctx = self._app()._aim_context()
        quad = H.sprite_quad(PROP, RADIUS_MM, SIZE_FACTOR, EYE, GLASS)
        edge_u = 3.5 / SPRITE
        point = (quad.centre_x - quad.half_w + edge_u * 2 * quad.half_w,
                 quad.centre_y)
        hit, _crit = ctx.resolve(point)
        self.assertEqual(hit, 0)


class NearerLayerAtTheRenderThresholdBlocksTest(unittest.TestCase):
    """
    TASK_hit_0200 bước 1 = "có" (xem file:line trong commit): một lớp gần
    hơn CÓ THỂ vẽ đè lên con bọ trên màn (depth test không bị tắt, chỉ depth
    WRITE của lớp tắt — ``content_plane.py:944`` — và bọ vẽ TRƯỚC lớp,
    ``renderer.py:391-392``). Phải loại phát bắn khi đó, dùng ĐÚNG ngưỡng vẽ
    của ``video_rgba.frag`` (0,05), không phải ``ALPHA_BLOCK`` 0,5 của
    ``is_visible``.
    """

    def test_a_point_covered_by_an_opaque_nearer_layer_is_a_miss(self) -> None:
        mask = H.SpriteMask.from_rgba(_bug_sprite())
        bugs = {0: (PROP, RADIUS_MM, SIZE_FACTOR, mask)}
        quad = H.sprite_quad(PROP, RADIUS_MM, SIZE_FACTOR, EYE, GLASS)
        centre = (quad.centre_x, quad.centre_y)
        opaque_nearer = (LayerPlane(
            alpha=np.full((64, 64), np.uint8(255)), z_m=-0.09,
            quad_w_m=1.0, quad_h_m=1.0, centre_x_m=0.0, centre_y_m=0.0,
            z_min_m=-0.09, z_max_m=-0.09),)
        hit, _ = H.pick_target(bugs, centre, EYE, opaque_nearer, GLASS,
                               alpha_threshold=0.05)
        self.assertEqual(hit, -1)
        # Doi chung: khong co lop gan hon thi van trung.
        hit_clear, _ = H.pick_target(bugs, centre, EYE, (), GLASS,
                                     alpha_threshold=0.05)
        self.assertEqual(hit_clear, 0)


class AimRayTest(unittest.TestCase):
    """Nghịch đảo của ``project_to_design`` phải khớp từng phần nghìn mm."""

    def test_aim_point_round_trips_through_the_projection(self) -> None:
        from src.game.viewport_filter import project_to_design

        for point in ((640.0, 360.0), (200.0, 120.0), (1100.0, 640.0)):
            for z_mm in (-50.0, -180.0, -600.0):
                with self.subTest(point=point, z=z_mm):
                    world = H.aim_point_mm(point, z_mm, EYE, GLASS)
                    back = project_to_design(world, EYE, GLASS)
                    self.assertAlmostEqual(back[0], point[0], places=6)
                    self.assertAlmostEqual(back[1], point[1], places=6)


if __name__ == "__main__":
    unittest.main()
