r"""
TASK_game_1110 F1 — bắn trúng **cả con bọ**, không chỉ cái tâm của nó.

Trước F1, ``game_bridge.fire`` chỉ so tâm ngắm với **tâm chiếu** của con bọ
trong bán kính 40 px. Sprite lại được vẽ ở ``radius_mm × SIZE_FACTOR 3.0``,
nên phần lớn hình bóng của nó — chân, rìa thân — nằm NGOÀI vòng tròn ấy:
nhắm đúng vào cái chân mà vẫn báo hụt.

Ở đây hộp trúng là **đúng cái người chơi nhìn thấy**:

1. hình chiếu của quad sprite xuống không gian thiết kế 1280×720;
2. **mặt nạ alpha** của chính file sprite, ngưỡng 0,25, nới ra 4 px;
3. tia ngắm từ mắt đang được theo dõi: gặp một lớp GẦN HƠN có alpha > 0,5
   thì phát đó găm vào lớp — hụt.

Toàn bộ module thuần số: không GL, không trạng thái toàn cục, nên
``tests/test_hit_box.py`` chấm được nó trên một sprite tự dựng.
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from src.content.hidden_props import ALPHA_BLOCK, _blocked, _nearer
from src.ui_album.viewport_filter import DESIGN_SIZE, project_to_design

#: Alpha tối thiểu của sprite để coi là "có thân ở đây" (kịch bản F1).
SPRITE_ALPHA_MIN = 0.25
#: Nới hộp trúng ra ngần này pixel thiết kế. Điểm xa hơn thế là hụt — chốt
#: bằng ``<``, nên đúng 4 px vẫn là hụt (kịch bản: "0 % ở ≥ 4 px bên ngoài").
HIT_TOLERANCE_PX = 4.0
#: Phần TRÊN của sprite tính là điểm yếu (đầu con bọ) → chí mạng.
CRIT_TOP_FRACTION = 0.35


@dataclass(frozen=True)
class SpriteMask:
    """
    Mặt nạ alpha của một sprite, kèm khoảng cách tới hình bóng.

    ``dist`` là distance transform của phần NGOÀI hình bóng, tính bằng pixel
    của chính file sprite. Nhờ nó, phép "nới 4 px" là một phép so sánh chứ
    không phải một chùm điểm lấy mẫu quanh tâm ngắm — nên biên của hộp trúng
    đúng bằng 4 px ở mọi hướng, kể cả ở đầu mút cái chân.
    """

    alpha: np.ndarray            # H x W uint8
    dist: np.ndarray             # H x W float32, 0 ở trong hình bóng

    @classmethod
    def from_rgba(cls, rgba: np.ndarray) -> "SpriteMask":
        alpha = np.ascontiguousarray(np.asarray(rgba)[..., 3], dtype=np.uint8)
        inside = alpha > int(round(SPRITE_ALPHA_MIN * 255))
        outside = np.where(inside, np.uint8(0), np.uint8(255))
        if not bool(inside.any()):
            dist = np.full(alpha.shape, np.inf, dtype=np.float32)
        else:
            dist = cv2.distanceTransform(outside, cv2.DIST_L2, 5)
        return cls(alpha=alpha, dist=np.ascontiguousarray(dist,
                                                          dtype=np.float32))

    @property
    def size(self) -> tuple[int, int]:
        """``(rộng, cao)`` tính bằng pixel của file sprite."""
        return (int(self.alpha.shape[1]), int(self.alpha.shape[0]))


@dataclass(frozen=True)
class SpriteQuad:
    """Hình chiếu của quad sprite xuống không gian thiết kế 1280×720."""

    centre_x: float
    centre_y: float
    half_w: float
    half_h: float

    @property
    def rect(self) -> tuple[float, float, float, float]:
        """``(x, y, rộng, cao)`` — cùng quy ước với ``VIEWPORT_RECT``."""
        return (self.centre_x - self.half_w, self.centre_y - self.half_h,
                self.half_w * 2.0, self.half_h * 2.0)


def sprite_quad(prop_mm, radius_mm, size_factor, rest_eye_mm,
                screen_mm=(344.0, 194.0), centre_mm=(0.0, 0.0)) -> SpriteQuad | None:
    """
    Quad của con bọ, chiếu xuống pixel thiết kế từ ``rest_eye_mm``.

    Sprite là một quad **quay mặt về camera** cạnh ``radius_mm × size_factor``
    (xem ``bugs.BugPainter``). Camera của HCPD không xoay — trục nhìn luôn
    vuông góc với màn — nên trong không gian thế giới quad ấy song song với
    hai trục màn, và bốn góc của nó chiếu xuống cùng một hệ số ``t`` như tâm.
    Vì vậy nửa cạnh trên màn chỉ là nửa cạnh thật nhân ``t``.
    """
    centre = project_to_design(prop_mm, rest_eye_mm, screen_mm, centre_mm)
    if centre is None:
        return None
    eye_z = float(rest_eye_mm[2])
    denom = eye_z - float(prop_mm[2])
    if abs(denom) < 1e-6:
        return None
    t = eye_z / denom
    half_mm = 0.5 * float(radius_mm) * float(size_factor) * t
    width_mm, height_mm = screen_mm
    return SpriteQuad(centre_x=centre[0], centre_y=centre[1],
                      half_w=half_mm / width_mm * DESIGN_SIZE[0],
                      half_h=half_mm / height_mm * DESIGN_SIZE[1])


def quad_uv(quad: SpriteQuad, point) -> tuple[float, float]:
    """``(u, v)`` trong quad; ``v = 0`` ở MÉP TRÊN, như hàng 0 của mảng."""
    u = (point[0] - (quad.centre_x - quad.half_w)) / max(2.0 * quad.half_w, 1e-9)
    v = (point[1] - (quad.centre_y - quad.half_h)) / max(2.0 * quad.half_h, 1e-9)
    return (u, v)


def hits_sprite(mask: SpriteMask, quad: SpriteQuad, point,
                tolerance_px: float = HIT_TOLERANCE_PX) -> bool:
    """
    Tâm ngắm có nằm trên thân con bọ không (alpha > 0,25, nới ``tolerance_px``).

    Dung sai đổi từ pixel thiết kế sang pixel sprite theo đúng hệ số phóng của
    quad; dùng cạnh NGẮN hơn nên dung sai không bao giờ rộng hơn mức đã hứa.
    """
    sprite_w, sprite_h = mask.size
    if quad.half_w <= 0.0 or quad.half_h <= 0.0:
        return False
    u, v = quad_uv(quad, point)
    scale = min(sprite_w / max(2.0 * quad.half_w, 1e-9),
                sprite_h / max(2.0 * quad.half_h, 1e-9))
    tol_sprite = float(tolerance_px) * scale
    # Điểm có thể rơi ngoài quad và vẫn trúng nhờ dung sai, nên kẹp vào mảng
    # thay vì loại ngay: khoảng cách tới hình bóng đã nói đủ.
    col = int(round(np.clip(u * (sprite_w - 1), 0, sprite_w - 1)))
    row = int(round(np.clip(v * (sprite_h - 1), 0, sprite_h - 1)))
    outside_u = max(0.0, -u, u - 1.0) * 2.0 * quad.half_w * scale
    outside_v = max(0.0, -v, v - 1.0) * 2.0 * quad.half_h * scale
    beyond = float(np.hypot(outside_u, outside_v))
    return float(mask.dist[row, col]) + beyond < tol_sprite


def is_crit(quad: SpriteQuad, point,
            top_fraction: float = CRIT_TOP_FRACTION) -> bool:
    """Điểm yếu = ``top_fraction`` phía TRÊN của sprite (đầu con bọ)."""
    _u, v = quad_uv(quad, point)
    return v <= float(top_fraction)


def aim_point_mm(point, z_mm, rest_eye_mm, screen_mm=(344.0, 194.0),
                 centre_mm=(0.0, 0.0)):
    """
    Điểm 3D nơi tia ngắm cắt mặt phẳng ``z_mm``.

    Nghịch đảo đúng của ``project_to_design``: tâm ngắm → điểm trên mặt kính
    (z = 0) → kéo dài tia mắt tới độ sâu của con bọ.
    """
    width_mm, height_mm = screen_mm
    sx = (point[0] / DESIGN_SIZE[0] - 0.5) * width_mm + centre_mm[0]
    sy = (0.5 - point[1] / DESIGN_SIZE[1]) * height_mm + centre_mm[1]
    ex, ey, ez = (float(v) for v in rest_eye_mm)
    if abs(ez) < 1e-6:
        return None
    # P(s) = E + s·(S − E); P_z = ez·(1 − s) = z_mm.
    s = 1.0 - float(z_mm) / ez
    return (ex + s * (sx - ex), ey + s * (sy - ey), float(z_mm))


def blocked_by_nearer_layer(planes, rest_eye_mm, point, z_mm,
                            screen_mm=(344.0, 194.0),
                            centre_mm=(0.0, 0.0),
                            alpha_threshold: float = ALPHA_BLOCK) -> bool:
    """
    Tia ngắm có găm vào một lớp **gần hơn** con bọ không.

    TASK_hit_0200: ``alpha_threshold`` mặc định vẫn ``ALPHA_BLOCK`` (0,5, hành
    vi cũ của ``_blocked``/``is_visible``). ``pick_target`` gọi với
    ``content.layer_alpha_threshold`` (0,05) — ĐÚNG ngưỡng mà
    ``video_rgba.frag`` dùng để vẽ hay bỏ một điểm lớp — nên "bắn trúng được"
    khớp đúng "có vẽ đè lên màn hay không", không phải một ngưỡng tự chọn.
    Dùng lại đúng ``hidden_props._blocked`` (cùng hình học ``layer_planes``
    của renderer, không dựng lại công thức).
    """
    target = aim_point_mm(point, z_mm, rest_eye_mm, screen_mm, centre_mm)
    if target is None:
        return False
    return _blocked(tuple(float(v) for v in rest_eye_mm), target,
                    _nearer(planes, float(z_mm) / 1000.0), alpha_threshold)


def pick_target(bugs, point, rest_eye_mm, planes=(),
                screen_mm=(344.0, 194.0), centre_mm=(0.0, 0.0),
                tolerance_px: float = HIT_TOLERANCE_PX,
                alpha_threshold: float | None = None):
    """
    Chọn con bọ bị bắn trúng. Trả ``(chỉ số, chí mạng)``; ``(-1, False)`` = hụt.

    TASK_hit_0200: ``bugs`` giờ là MỌI con **chưa diệt** (không lọc theo
    ``is_exposed`` nữa — ``BugPainter.draw`` cũng vẽ y hệt tập này, xem
    ``app.py:341``, nên "bắn được" phải khớp "có vẽ ra"). Chỉ loại khi
    ``alpha_threshold`` được truyền (khác ``None``) VÀ một lớp gần hơn có
    alpha ở điểm ngắm ≥ ngưỡng đó — đúng ngưỡng ``video_rgba.frag`` dùng để
    vẽ/bỏ một điểm lớp (``content.layer_alpha_threshold``, không phải
    ``ALPHA_BLOCK`` 0,5 của ``is_visible``), vì câu hỏi ở đây là "lớp đó có
    đang VẼ ĐÈ lên màn hay không", không phải "có nên tính là ẩn hay không".
    ``alpha_threshold=None`` (mặc định) = không kiểm lớp (dùng khi lớp gần
    không thể che bọ trên màn — xem file:line trong commit log).

    Trúng nhiều con thì lấy con GẦN NGƯỜI XEM nhất.
    """
    best, best_z, best_crit = -1, None, False
    for index, (prop_mm, radius_mm, size_factor, mask) in bugs.items():
        quad = sprite_quad(prop_mm, radius_mm, size_factor, rest_eye_mm,
                           screen_mm, centre_mm)
        if quad is None or not hits_sprite(mask, quad, point, tolerance_px):
            continue
        z_mm = float(prop_mm[2])
        if alpha_threshold is not None and blocked_by_nearer_layer(
                planes, rest_eye_mm, point, z_mm, screen_mm, centre_mm,
                alpha_threshold):
            continue
        if best_z is None or z_mm > best_z:
            best, best_z, best_crit = index, z_mm, is_crit(quad, point)
    return best, best_crit


def shot_diagnostic(bugs, point, rest_eye_mm, planes=(),
                    screen_mm=(344.0, 194.0), centre_mm=(0.0, 0.0),
                    tolerance_px: float = HIT_TOLERANCE_PX,
                    alpha_threshold: float | None = None,
                    is_exposed: dict | None = None) -> str:
    """
    TASK_hit_0200 bước 2 — một dòng chẩn đoán mỗi phát bắn, MỖI con chưa diệt:
    quad chiếu, ``is_exposed``, alpha sprite tại điểm ngắm, và lý do bị loại
    (nếu có). Dòng này ở lại trong bản build — không phải công cụ gỡ tạm.
    """
    is_exposed = is_exposed or {}
    parts = [f"aim=({point[0]:.0f},{point[1]:.0f})"]
    for index, (prop_mm, radius_mm, size_factor, mask) in sorted(bugs.items()):
        quad = sprite_quad(prop_mm, radius_mm, size_factor, rest_eye_mm,
                           screen_mm, centre_mm)
        if quad is None:
            parts.append(f"bo#{index} ly_do=khong_chieu_duoc")
            continue
        u, v = quad_uv(quad, point)
        sprite_w, sprite_h = mask.size
        col = int(np.clip(u, 0.0, 1.0) * (sprite_w - 1))
        row = int(np.clip(v, 0.0, 1.0) * (sprite_h - 1))
        alpha = float(mask.alpha[row, col]) / 255.0
        hit_sprite = hits_sprite(mask, quad, point, tolerance_px)
        reason = "trung"
        if not hit_sprite:
            reason = "ngoai_quad_hoac_alpha"
        elif alpha_threshold is not None and blocked_by_nearer_layer(
                planes, rest_eye_mm, point, float(prop_mm[2]), screen_mm,
                centre_mm, alpha_threshold):
            reason = "lop_gan_hon_che"
        parts.append(
            f"bo#{index} quad=({quad.centre_x:.0f},{quad.centre_y:.0f},"
            f"{quad.half_w:.0f},{quad.half_h:.0f}) "
            f"is_exposed={is_exposed.get(index, '?')} alpha={alpha:.2f} "
            f"ly_do={reason}")
    return " | ".join(parts)


__all__ = ["ALPHA_BLOCK", "CRIT_TOP_FRACTION", "HIT_TOLERANCE_PX",
           "SPRITE_ALPHA_MIN", "SpriteMask", "SpriteQuad", "aim_point_mm",
           "blocked_by_nearer_layer", "hits_sprite", "is_crit", "pick_target",
           "quad_uv", "shot_diagnostic", "sprite_quad"]
