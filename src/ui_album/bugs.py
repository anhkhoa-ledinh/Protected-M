"""
F3 — vẽ bọ: quad quay mặt về camera, có sprite theo cấp, tan dần khi bị diệt.

Vì sao vẽ **trong cảnh 3D** chứ không vẽ đè lên giao diện: cơ chế ẩn/hiện của
trò chơi là **depth** — lớp gần hơn che con bọ nằm sau nó. Vẽ đè lên UI thì
con bọ nào cũng thấy, ván mất sạch ý nghĩa. Nên quad đi đúng chỗ của quả cầu
cũ: sau pass cảnh, trước các lớp, có depth test và depth write.

Tan: 0,4 s mờ dần + co lại; rồi 2–4 chấm bụi ``text_muted`` nằm lại 2 s.
"""

from __future__ import annotations

import logging
import random
import time

import numpy as np
from OpenGL import GL

from src.renderer.shader import ShaderProgram
from src.game.view import LEGACY_CAMO
from src.ui_album.textures import load_rgba, upload_rgba

logger = logging.getLogger(__name__)

FADE_S = 0.4
DUST_S = 2.0
#: Sprite to gấp ngần này lần bán kính quả cầu đánh dấu cũ.
#:
#: Lúc chưa sửa lỗi nạp ma trận, sprite KHÔNG hề hiện; tôi tưởng nó quá nhỏ
#: nên đẩy lên 6,0. Sửa xong thì nó hiện và 6,0 **to quá** (người dùng
#: 07/11 01:17). 3,0 cho con bọ cỡ 30-35 px: thấy rõ mà không che mất ảnh,
#: vẫn nằm gọn trong bán kính trúng 40 px. Chỉnh một số này là đổi cỡ bọ.
SIZE_FACTOR = 3.0
DUST_MIN, DUST_MAX = 2, 4

#: TASK_bugs_look_0130: mỗi con bọ vẽ nhỏ lại một tỉ lệ ngẫu nhiên của
#: ``SIZE_FACTOR`` — KHÔNG bao giờ lớn hơn, nên vẫn nằm gọn trong
#: ``radius_mm`` mà đặt chỗ / ``is_exposed`` đã giả định (không đụng tới).
#: TASK_bug_art_0530: nguoi dung choi 11/11 thay bo con qua to — ha xuong hon
#: han (0.55-0.85 -> 0.28-0.45), cung ty le co dinh voi SIZE_FACTOR.
BUG_SIZE_MIN = 0.28
BUG_SIZE_MAX = 0.45
#: Độ mạnh pha màu nguỵ trang (0 = giữ y nguyên pixel sprite hôm nay).
#: TASK_bug_art_0530: 0.55 -> 0.75 — công thức mới nhân theo độ sáng riêng
#: của pixel (xem sprite.frag) nên tăng độ mạnh không còn biến bọ thành một
#: khối màu phẳng như công thức ``mix`` thẳng cũ.
BUG_CAMO_STRENGTH = LEGACY_CAMO
#: Cạnh ô vuông lấy mẫu màu nền quanh vị trí chiếu của con bọ, px thiết kế.
CAMO_PATCH_PX = 48


def sprite_for(manifest, level: int) -> str | None:
    """File sprite của cấp ``level`` theo ``manifest.sprites``."""
    for key, record in manifest.sprites.items():
        if int(level) in [int(v) for v in record.get("levels", [])]:
            return record.get("file")
    return None


def assign_bug_sizes(props, seed_key, scale: float = 1.0) -> None:
    """
    Gán ``size_k`` (= ``SIZE_FACTOR × u``, ``u`` đều trong
    ``[BUG_SIZE_MIN, BUG_SIZE_MAX]``) cho mỗi con, MỘT LẦN khi ván mở.

    Seed = ``(seed_key, chỉ số con)``. ``seed_key`` is the per-match bug seed
    (``app._bug_seed``), so one match keeps its sizes and a new match gets new
    ones; it used to be the photo name.
    Ghi thẳng vào ``prop["size_k"]`` — ``BugPainter.draw`` và
    ``app._aim_context`` đọc chung đúng một giá trị này, không có nguồn thứ
    hai. Chỉ đổi kích thước VẼ; ``radius_mm`` (đặt chỗ, ``is_exposed``) không
    động tới.
    """
    for index, prop in enumerate(props):
        rng = random.Random(f"{seed_key}:{index}")
        prop["size_k"] = (rng.uniform(BUG_SIZE_MIN, BUG_SIZE_MAX)
                          * SIZE_FACTOR * float(scale))


def assign_bug_camo(props, rgb_image, rest_eye_mm, screen_mm=(344.0, 194.0),
                    centre_mm=(0.0, 0.0), patch_px: int = CAMO_PATCH_PX,
                    strength: float = BUG_CAMO_STRENGTH) -> None:
    """
    Gán ``camo_rgb`` (0–1) = màu trung bình một ô ``patch_px`` quanh vị trí
    chiếu của con bọ ở MẮT NGHỈ, đọc từ ảnh gốc (CPU, không đọc lại GL).

    Chạy một lần khi ván mở, cùng lúc với ``assign_bug_sizes``. Ảnh gốc
    không có (tải lỗi) thì để ``(0, 0, 0)`` — shader khi đó vẫn tô vì
    ``BUG_CAMO_STRENGTH`` > 0, nhưng ``draw()`` chỉ dùng giá trị này nếu nó
    tồn tại trên prop.
    """
    from src.game.view import DESIGN_SIZE, project_to_design

    height, width = rgb_image.shape[:2]
    half = max(1, int(patch_px) // 2)
    for prop in props:
        prop["camo_strength"] = float(strength)
        point = project_to_design(tuple(prop["pos_mm"]), rest_eye_mm,
                                  screen_mm, centre_mm)
        if point is None:
            prop["camo_rgb"] = (0.0, 0.0, 0.0)
            continue
        px = int(round(point[0] / DESIGN_SIZE[0] * width))
        py = int(round(point[1] / DESIGN_SIZE[1] * height))
        x0, x1 = max(0, px - half), min(width, px + half)
        y0, y1 = max(0, py - half), min(height, py + half)
        patch = rgb_image[y0:y1, x0:x1]
        if patch.size == 0:
            prop["camo_rgb"] = (0.0, 0.0, 0.0)
        else:
            mean = patch.reshape(-1, 3).astype(np.float64).mean(axis=0) / 255.0
            prop["camo_rgb"] = tuple(float(v) for v in mean)


class BugPainter:
    """Vẽ bọ bằng một shader riêng; giữ texture và mốc thời gian đã diệt."""

    def __init__(self, manifest, shader_dir) -> None:
        self.manifest = manifest
        self._shader = ShaderProgram(shader_dir / "sprite.vert",
                                     shader_dir / "sprite.frag")
        self._vao = self._vbo = 0
        self._textures: dict[str, int] = {}
        self._masks: dict[str, object] = {}
        self.found_at: dict[int, float] = {}
        self.dust: dict[int, list] = {}

    def initialize(self) -> None:
        corners = np.array([-0.5, -0.5, 0.0, 1.0, 0.5, -0.5, 1.0, 1.0,
                            0.5, 0.5, 1.0, 0.0, -0.5, -0.5, 0.0, 1.0,
                            0.5, 0.5, 1.0, 0.0, -0.5, 0.5, 0.0, 0.0],
                           dtype=np.float32)
        self._vao = GL.glGenVertexArrays(1)
        self._vbo = GL.glGenBuffers(1)
        GL.glBindVertexArray(self._vao)
        GL.glBindBuffer(GL.GL_ARRAY_BUFFER, self._vbo)
        GL.glBufferData(GL.GL_ARRAY_BUFFER, corners.nbytes, corners,
                        GL.GL_STATIC_DRAW)
        GL.glEnableVertexAttribArray(0)
        GL.glVertexAttribPointer(0, 2, GL.GL_FLOAT, GL.GL_FALSE, 16, None)
        GL.glEnableVertexAttribArray(1)
        GL.glVertexAttribPointer(1, 2, GL.GL_FLOAT, GL.GL_FALSE, 16,
                                 GL.ctypes.c_void_p(8))
        GL.glBindVertexArray(0)

    def texture(self, relative: str) -> int | None:
        """Texture của một file sprite; nạp một lần rồi giữ."""
        if relative in self._textures:
            return self._textures[relative]
        array = load_rgba(self.manifest.image(relative))
        if array is None:
            return None
        self._textures[relative] = upload_rgba(array)
        return self._textures[relative]

    def sprite_mask(self, level: int):
        """
        ``SpriteMask`` của sprite cấp ``level`` — hộp trúng của F1 đọc từ đây.

        Cùng một file PNG mà ``texture`` nạp lên GPU, nên thứ người chơi NHÌN
        và thứ phép bắn ĐO không bao giờ là hai hình khác nhau. ``None`` khi
        thiếu bộ asset: lúc đó không có bọ để vẽ, cũng không có gì để bắn.
        """
        from src.game.view import SpriteMask

        relative = sprite_for(self.manifest, level)
        if not relative:
            return None
        if relative in self._masks:
            return self._masks[relative]
        array = load_rgba(self.manifest.image(relative))
        if array is None:
            return None
        self._masks[relative] = SpriteMask.from_rgba(array)
        return self._masks[relative]

    def mark_found(self, index: int, position) -> None:
        """Ghi mốc bị diệt để chạy hiệu ứng tan, rồi rải bụi quanh chỗ đó."""
        if index in self.found_at:
            return
        self.found_at[index] = time.perf_counter()
        rng = random.Random(index)
        self.dust[index] = [
            (position[0] + rng.uniform(-0.02, 0.02),
             position[1] + rng.uniform(-0.02, 0.02), position[2])
            for _ in range(rng.randint(DUST_MIN, DUST_MAX))]

    def reset(self) -> None:
        self.found_at.clear()
        self.dust.clear()

    def _draw_quad(self, view, projection, centre, size, tint, texture,
                   camo=(0.0, 0.0, 0.0, 0.0)) -> None:
        program = self._shader.program_id
        # Dùng CHÍNH set_mat4 của dự án: nó nạp với GL_TRUE. Tự gọi
        # glUniformMatrix4fv với GL_FALSE là nạp ma trận CHUYỂN VỊ — con bọ
        # bay ra ngoài khung, vẽ xong mà không thấy gì (07/11).
        self._shader.set_mat4("uView", view)
        self._shader.set_mat4("uProjection", projection)
        self._shader.set_vec3("uCenter", *[float(v) for v in centre])
        self._shader.set_float("uSize", float(size))
        GL.glUniform4f(GL.glGetUniformLocation(program, "uTint"), *tint)
        GL.glUniform4f(GL.glGetUniformLocation(program, "uCamo"), *camo)
        GL.glUniform1i(GL.glGetUniformLocation(program, "uTexture"), 0)
        GL.glActiveTexture(GL.GL_TEXTURE0)
        GL.glBindTexture(GL.GL_TEXTURE_2D, texture)
        GL.glBindVertexArray(self._vao)
        GL.glDrawArrays(GL.GL_TRIANGLES, 0, 6)
        GL.glBindVertexArray(0)

    def draw(self, view, projection, props, found, level, theme) -> None:
        """Vẽ mọi con bọ của ván; ``props`` là danh sách của ``prop_game``."""
        relative = sprite_for(self.manifest, level)
        texture = self.texture(relative) if relative else None
        if texture is None:
            return
        white = self.manifest.colour("text_muted", theme)
        if not getattr(self, "_logged", False):
            self._logged = True
            logger.info("BugPainter: ve %d bo, sprite=%s, texture=%s, size=%.3f m",
                        len(props), relative, texture,
                        float(props[0]["radius_mm"]) / 1000.0
                        * float(props[0].get("size_k", SIZE_FACTOR))
                        if props else 0.0)
        now = time.perf_counter()
        self._shader.use()
        GL.glEnable(GL.GL_BLEND)
        GL.glBlendFunc(GL.GL_SRC_ALPHA, GL.GL_ONE_MINUS_SRC_ALPHA)
        # Pass cảnh bật GL_CULL_FACE. Quad quay mặt về camera dựng trong không
        # gian view nên chiều quấn của nó NGƯỢC với mặt mà driver giữ lại —
        # bật cull thì con bọ bị loại sạch, vẽ xong mà không thấy gì.
        culling = GL.glIsEnabled(GL.GL_CULL_FACE)
        GL.glDisable(GL.GL_CULL_FACE)
        for index, prop in enumerate(props):
            centre = [v / 1000.0 for v in prop["pos_mm"]]
            # TASK_bugs_look_0130: ĐÚNG MỘT nguồn kích thước — ``size_k`` ghi
            # một lần lúc mở ván (``assign_bug_sizes``); ``app._aim_context``
            # đọc lại chính giá trị này cho phép bắn, không có bản sao công
            # thức thứ hai. Thiếu khoá (đường cũ/đồ test) thì lùi về hằng cũ.
            size_k = float(prop.get("size_k", SIZE_FACTOR))
            size = float(prop["radius_mm"]) / 1000.0 * size_k
            camo_rgb = prop.get("camo_rgb")
            strength = float(prop.get("camo_strength", BUG_CAMO_STRENGTH))
            camo = (*camo_rgb, strength) if camo_rgb else (0.0, 0.0, 0.0, 0.0)
            if index not in self.found_at:
                self._draw_quad(view, projection, centre, size,
                                (1.0, 1.0, 1.0, 1.0), texture, camo)
                continue
            age = now - self.found_at[index]
            if age < FADE_S:
                # Tan: mờ dần và co lại trong 0,4 s.
                k = 1.0 - age / FADE_S
                self._draw_quad(view, projection, centre, size * k,
                                (1.0, 1.0, 1.0, k), texture, camo)
            elif age < FADE_S + DUST_S:
                for dot in self.dust.get(index, []):
                    # Bụi không nguỵ trang (strength 0): nó không phải sprite
                    # thật nữa, tô theo màu riêng của nó là đủ.
                    self._draw_quad(view, projection, dot, size * 0.12,
                                    (white[0], white[1], white[2], 0.9),
                                    texture)
        _ = found
        GL.glDisable(GL.GL_BLEND)
        if culling:
            GL.glEnable(GL.GL_CULL_FACE)

    def release(self) -> None:
        for texture in self._textures.values():
            GL.glDeleteTextures(1, [texture])
        self._textures.clear()
        if self._vao:
            GL.glDeleteVertexArrays(1, [self._vao])
        if self._vbo:
            GL.glDeleteBuffers(1, [self._vbo])
        self._shader.release()
