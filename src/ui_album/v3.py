"""
Giao diện v3 ("v2 · Redesign"): nạp ``assets/ui_v3/manifest.json`` và vẽ.

Thứ tự vẽ là **hợp đồng** của bộ asset (README mục "Draw order"):
``base`` → ``photos`` → ``top`` → ``bars`` → ``pips`` → ``pills`` →
``sprites`` → ``texts`` → hover/focus. ``assets/ui_v3/tools/preview.py`` là
bản tham chiếu bằng PIL của đúng thứ tự ấy, và ``expected/*.jpg`` là đầu ra
của nó; mọi phép tính chữ/thanh/viên ở đây **chép theo** preview.py, không tự
nghĩ lại — lệch một pixel là trượt ngưỡng 24 của ``tools/compare.py``.

Khác với lớp vẽ cũ (``draw.py``), module này không làm việc PIL nào trong
đường chạy mỗi khung: chữ, viên thuốc, đầu thanh bo góc và pip đều dựng một
lần rồi nằm trong bộ đệm texture (luật F4). ``base``, ``top`` và sprite của
một màn nạp **một lần cho mỗi màn**.
"""

from __future__ import annotations

import json
import logging
import math
from collections import OrderedDict
from pathlib import Path

import numpy as np
from OpenGL import GL
from PIL import Image, ImageDraw

from src.renderer.shader import ShaderProgram
from src.ui_album.fonts import FontBook
from src.ui_album.manifest import letterbox
from src.ui_album.textures import TextureCache, load_rgba, upload_rgba

logger = logging.getLogger(__name__)

#: Thư mục bộ asset v3, tính từ gốc project.
ASSET_DIR = Path("assets") / "ui_v3"

#: Không gian thiết kế của bộ v3. Mọi ``rect`` trong manifest nằm trong hệ này.
DESIGN_SIZE = (1280.0, 720.0)

#: Màu chữ mặc định khi ``runs[].col`` rỗng (preview.py dùng đúng hằng này).
INK = "#24415F"

#: Lề quanh ảnh chữ. Glyph có thể tràn ra ngoài bút (bearing âm, dấu cao hơn
#: ascent); không chừa lề thì chữ bị cắt mép và điểm compare.py nhảy vọt.
TEXT_PAD = 8

#: Vùng vẽ 3D và webcam ở **chế độ chụp** — README mục "Shot mode".
SHOT_VIEWPORT_RGB = (60, 70, 80)
SHOT_CAMERA_RGB = (90, 90, 100)

#: Vòng hover: 2 px #3B72B4 ở 45 %. Vòng focus bàn phím: 2,5 px, lùi ra 3 px.
HOVER_PX, HOVER_ALPHA = 2.0, 0.45
FOCUS_PX, FOCUS_INSET = 2.5, 3.0
#: TASK_bug_art_0530: bán kính bo góc của field_user/field_pass, đo trực tiếp
#: trên ``v3_01a_sign_in/base.png`` (quét cột đầu mép bo ở mỗi hàng, bán kính
#: = khoảng cách từ mép tới điểm offset về 0 — ra ~22 px ở ảnh ×1,5, tức 15 px
#: không gian thiết kế). Không có trường bán kính nào trong manifest.
FIELD_CORNER_RADIUS_PX = 15.0

#: TASK_shared_album_1111 Phase 1: hotspot nào cũng có thể nhận vòng hover/
#: focus (``_draw_rings`` lặp hết ``hotspots[]``), mà sửa 03:xx ("man chọn
#: cấp — thẻ bo góc hiện viền vuông") cho thấy lỗi không chỉ ở ô nhập. Không
#: có trường bán kính trong manifest nên đo trực tiếp trên ``base.png`` của
#: từng loại — cùng cách đo FIELD_CORNER_RADIUS_PX ở trên (quét hàng, tìm
#: hàng đầu mà mép bo chạm hẳn vào cạnh rect, đổi ảnh ×1,5 -> thiết kế):
#: - ``btn_*`` (``v3_01a_sign_in``: "Sign in" ~9-12 px, "Create album" ~13 px;
#:   ``v3_03_level_select``: "Back" ~10 px, "Next: choose a photo" ~12 px).
#: - ``level_row_*`` / ``weapon_*`` (hàng/thẻ danh sách — cùng kiểu bo, đo
#:   ``v3_03_level_select``: level_row_1 ~18-19 px; ``v3_05_loadout``:
#:   weapon_Precision gun ~19 px).
#: ``switch_*`` không cần đo: nó là một viên nhộng, bán kính luôn bằng nửa
#: chiều cao của chính hotspot đó (tính trong ``_ring_radius``).
#: Các loại còn lại (tab_, link_, icon_, sticker_, slot_, dropzone, dropdown_,
#: slider, display_, mode_, seg_, theme_, footer_, continue/leave/sign_out/
#: recent_memory...) không có art tĩnh trong ``base.png`` để đo (vẽ bằng
#: sprite động hoặc không có nền) — giữ bán kính 0 (khung vuông như hôm nay,
#: đúng với art của chúng, không phải một lựa chọn bỏ qua).
BTN_CORNER_RADIUS_PX = 12.0
ROW_CORNER_RADIUS_PX = 18.0


def _ring_radius(name: str, rect) -> float:
    """Bán kính bo góc của vòng hover/focus quanh hotspot ``name``."""
    if name.startswith("field_"):
        return FIELD_CORNER_RADIUS_PX
    if name.startswith("btn_"):
        return BTN_CORNER_RADIUS_PX
    if name.startswith("level_row_") or name.startswith("weapon_"):
        return ROW_CORNER_RADIUS_PX
    if name.startswith("switch_"):
        return float(rect[3]) * 0.5
    return 0.0

#: Tâm ngắm: một dấu ``+`` nhỏ, tính bằng pixel thiết kế.
#:
#: Bộ asset có sẵn sprite ``crosshair`` nhưng đó là vòng tròn 92 px — to gần
#: bằng một con bọ, che mất đúng chỗ người chơi đang soi. Dấu cộng vẽ bằng
#: bốn ô tô màu nên không tốn texture nào, và luôn nét ở mọi cỡ cửa sổ.
#: Viền sẫm vẽ dưới, nét sáng vẽ trên: ảnh nền có cả vùng tối lẫn vùng sáng,
#: một màu đơn sẽ biến mất ở một trong hai.
AIM_ARM = 7.0
AIM_THICK = 2.0
AIM_GAP = 2.0
AIM_EDGE = 1.0
AIM_COL, AIM_EDGE_COL = "#FFFFFF", "#24415F"

#: Bắn xong thì tâm ngắm **bung ra** và mờ đi, rồi khép lại theo nhịp súng.
#:
#: Đo trên một phiên chơi thật: 54 phát ra, 54 cú bấm bị chặn vì chưa tới
#: nhịp. Người chơi bấm nhanh gấp đôi súng, mà mỗi cú bị chặn thì màn hình
#: im lặng hoàn toàn — không tiếng (bản demo không có âm), không hình. Nhịp
#: bắn vô hình thì cảm giác như app đớp mất cú bấm. Cho tâm ngắm khép dần
#: lại là nói được "súng đang hồi" mà không phải đổi một con số cân bằng nào.
AIM_RECOIL = 6.0
AIM_DIM = 0.45

#: Lỗ đạn trên ảnh. Người chơi đang bắn vào một tấm ảnh, nên dấu để lại là
#: một cái **lỗ**, không phải chấm đỏ báo lỗi.
#:
#: Ba lớp tròn lồng nhau: quầng ngoài nhạt (mảnh vụn), vành sẫm, lõi gần
#: đen. Kích thước CỐ ĐỊNH nên cả vệt mờ dần chỉ tốn ba texture — xem
#: ``V3Painter.disc``.
HOLE_LIFE_S = 3.0
HOLE_FADE_S = 0.8               # chỉ mờ đi ở quãng cuối
HOLE_CORE, HOLE_RING, HOLE_DUST = 5.0, 9.0, 15.0
HOLE_CORE_COL, HOLE_RING_COL, HOLE_DUST_COL = "#141C28", "#2E3A4C", "#8DA3BA"

#: Trúng bọ thì thêm một chớp xanh ngắn đè lên lỗ — để biết là đã diệt được.
HIT_FLASH_S = 0.45
HIT_FLASH_SIZE = 26.0
HIT_FLASH_COL = "#3AA17A"

#: Dòng báo nổi ở giữa dưới (toast của giao diện).
TOAST_RECT = (340.0, 626.0, 600.0, 44.0)
TOAST_RADIUS = 22.0
TOAST_BG, TOAST_FG = "#24415F", "#FFFFFF"
TOAST_FONT = {"ff": "Nunito", "fw": 800, "fs": 15.0, "col": TOAST_FG}

#: Code-drawn album buttons, with fixed rectangles matching the manifest
#: hotspots. Share sits on the memory caption row; Random in the drop card.
CHOOSE_PHOTO_SCREEN = "v3_04_choose_photo"
SHARE_BTN_RECT = (944.0, 384.0, 228.0, 30.0)
RANDOM_BTN_RECT = (264.0, 478.0, 240.0, 52.0)
SHARE_BTN_FONT = {"ff": "Nunito", "fw": 800, "fs": 14.0, "ls": 0.0,
                  "col": "#3B72B4"}
#: Màu ``primary`` của manifest — hợp đồng dùng hex trực tiếp (xem ``CARET_COL``).
BUTTON_PRIMARY_COL = "#3B72B4"

#: Hộp xác nhận trước lượt chia sẻ đầu tiên của mỗi ảnh (P2.3) — chữ ĐÚNG
#: nguyên văn đề bài, không một ký tự khác.
CONFIRM_TEXT = ("This photo will be public: anyone with the app can "
               "download it. Do not share photos of people without their "
               "consent. Upload?")
CONFIRM_CARD_RECT = (380.0, 230.0, 520.0, 260.0)
CONFIRM_YES_RECT = (766.0, 422.0, 110.0, 44.0)
CONFIRM_NO_RECT = (640.0, 422.0, 110.0, 44.0)
CONFIRM_TEXT_FONT = {"ff": "Nunito", "fw": 700, "fs": 15.0, "ls": 0.0,
                     "col": INK}
CONFIRM_BTN_FONT = {"ff": "Nunito", "fw": 800, "fs": 15.0, "ls": 0.0}

#: Con trỏ nhập của ô đang chọn (ghi trong ``screens[].notes``).
CARET_SIZE = (2.0, 20.0)
CARET_GAP = 2.0
CARET_COL = "#3B72B4"


def rgba(colour: str | None, alpha: float = 1.0) -> tuple[float, float, float, float]:
    """``"#RRGGBB"`` → ``(r, g, b, a)`` 0–1. ``None`` trả màu chữ mặc định."""
    text = str(colour or INK).lstrip("#")
    if len(text) == 6:
        text += "FF"
    try:
        value = [int(text[i:i + 2], 16) / 255.0 for i in (0, 2, 4, 6)]
    except ValueError:
        return (1.0, 0.0, 1.0, 1.0)             # màu báo lỗi, thấy ngay trên màn
    return (value[0], value[1], value[2], value[3] * float(alpha))


def _byte(colour: str | None, alpha: float = 1.0) -> tuple[int, int, int, int]:
    """Màu cho PIL: ``(r, g, b, a)`` 0–255."""
    return tuple(int(round(c * 255)) for c in rgba(colour, alpha))


class V3Manifest:
    """Bản kê v3 đã nạp. Chỉ đọc; không giữ trạng thái vẽ nào."""

    def __init__(self, root: Path) -> None:
        self.root = Path(root) / ASSET_DIR
        path = self.root / "manifest.json"
        self.data = json.loads(path.read_text(encoding="utf-8"))
        version = int(self.data.get("version", 0))
        if version != 3:
            raise ValueError(f"{path}: can manifest version 3, doc duoc {version}")
        self.image_scale = float(self.data.get("image_scale", 1.5))
        self.design_size = tuple(self.data.get("design_size", DESIGN_SIZE))
        self.screens = self.data["screens"]
        self.colors = self.data.get("colors", {})
        # ``FontBook`` đọc đúng hai thuộc tính này của bản kê.
        self.fonts = self.data.get("fonts", {})
        self.sprites = {}
        missing = self.validate()
        logger.info("UI v3: %d man, %d sprite, thieu %d file",
                    len(self.screens), self.sprite_count(), len(missing))
        if missing:
            logger.warning("UI v3: thieu file %s", missing[:5])

    def screen_names(self) -> list[str]:
        """Tên 23 màn, theo thứ tự trong manifest."""
        return list(self.screens)

    def sprite_count(self) -> int:
        return sum(len(s.get("sprites", ())) for s in self.screens.values())

    def path(self, relative: str | None) -> Path | None:
        return None if not relative else self.root / relative

    def sample_photo(self) -> Path | None:
        return self.path(self.data.get("sample_photo"))

    def sprite(self, screen: str, key: str) -> dict | None:
        """Bản ghi sprite ``key`` của màn ``screen``; ``None`` nếu không có."""
        for record in self.screens.get(screen, {}).get("sprites", ()):
            if record["key"] == key:
                return record
        return None

    def validate(self) -> list[str]:
        """File khai trong manifest mà không có trên đĩa."""
        missing = []
        for name, screen in self.screens.items():
            for layer in ("base", "top"):
                relative = screen.get(layer)
                path = self.path(relative)
                if path is not None and not path.exists():
                    missing.append(relative)
            for record in screen.get("sprites", ()):
                path = self.path(record.get("file"))
                if path is not None and not path.exists():
                    missing.append(record["file"])
            _ = name
        return missing


class _PilCache:
    """
    LRU "dựng một lần bằng PIL, dùng lại mãi": khoá → (texture, số đo).

    Vì sao không dùng thẳng ``TextureCache.small``: chữ và viên thuốc cần
    **số đo** (bề rộng chuỗi, ascent) để đặt quad đúng chỗ, mà số đo ấy chỉ
    tính được lúc dựng. Trả kèm số đo nghĩa là đường chạy mỗi khung không gọi
    ``font.getlength`` lần nào nữa.
    """

    def __init__(self, limit: int = 768) -> None:
        self._store: OrderedDict[tuple, tuple] = OrderedDict()
        self._limit = int(limit)
        self.builds = 0

    def get(self, key: tuple, build):
        """``(texture, w_px, h_px, meta)``; gọi ``build()`` khi chưa có."""
        if key in self._store:
            self._store.move_to_end(key)
            return self._store[key]
        array, meta = build()
        if array is None:
            return None
        entry = (upload_rgba(array), array.shape[1], array.shape[0], meta)
        self._store[key] = entry
        self.builds += 1
        while len(self._store) > self._limit:
            _key, old = self._store.popitem(last=False)
            GL.glDeleteTextures(1, [old[0]])
        return entry

    def release(self) -> None:
        for entry in self._store.values():
            GL.glDeleteTextures(1, [entry[0]])
        self._store.clear()


class V3Painter:
    """
    Vẽ quad trong **không gian thiết kế** 1280×720; tự lo tỉ lệ và blend.

    ``set_frame`` nhận cỡ framebuffer THẬT mỗi khung, nên đổi cỡ cửa sổ (D1)
    không cần dựng lại gì ngoài những texture phụ thuộc tỉ lệ.
    """

    def __init__(self, manifest: V3Manifest, shader_dir: Path) -> None:
        self.manifest = manifest
        self.cache = TextureCache()
        self.pil = _PilCache()
        self.fonts = FontBook(manifest)
        self._shader = ShaderProgram(shader_dir / "album.vert",
                                     shader_dir / "album.frag")
        self._vao = self._vbo = 0
        self._fb = (1280, 720)
        self._box = letterbox(DESIGN_SIZE, self._fb)
        #: Texture của màn đang mở: ``(path, max_width)`` → ``(tex, w, h)``.
        self._screen_tex: dict[tuple, tuple[int, int, int]] = {}
        self._screen_key: tuple | None = None

    # ---- hạ tầng ---------------------------------------------------------
    def initialize(self) -> None:
        """Tạo VAO/VBO một lần, sau khi đã có context GL."""
        self._vao = GL.glGenVertexArrays(1)
        self._vbo = GL.glGenBuffers(1)
        GL.glBindVertexArray(self._vao)
        GL.glBindBuffer(GL.GL_ARRAY_BUFFER, self._vbo)
        GL.glBufferData(GL.GL_ARRAY_BUFFER, 96, None, GL.GL_DYNAMIC_DRAW)
        GL.glEnableVertexAttribArray(0)
        GL.glVertexAttribPointer(0, 2, GL.GL_FLOAT, GL.GL_FALSE, 16, None)
        GL.glEnableVertexAttribArray(1)
        GL.glVertexAttribPointer(1, 2, GL.GL_FLOAT, GL.GL_FALSE, 16,
                                 GL.ctypes.c_void_p(8))
        GL.glBindVertexArray(0)

    def set_frame(self, framebuffer) -> None:
        self._fb = (max(1, int(framebuffer[0])), max(1, int(framebuffer[1])))
        self._box = letterbox(DESIGN_SIZE, self._fb)

    @property
    def scale(self) -> float:
        """Pixel framebuffer trên mỗi pixel thiết kế."""
        return self._box.scale

    @property
    def k(self) -> float:
        """
        Tỉ lệ **của bộ asset** (``image_scale`` = 1,5), dùng cho chữ và viên.

        Vì sao không dựng chữ ở tỉ lệ framebuffer, là thứ trông có vẻ đúng
        hơn: FreeType làm tròn bước tiến của **từng** glyph về pixel nguyên,
        nên bề rộng một chuỗi KHÔNG tỉ lệ thuận với cỡ chữ. Dựng ở 1366/1280
        = 1,0667 thay vì 1,5 làm "anhkhoa" rộng thêm 5 %, dãy tám dấu chấm
        hẹp đi 11 %, và điểm ``compare.py`` nhảy từ ~3 lên 42. Dựng đúng tỉ lệ
        mà ``tools/preview.py`` đã dùng thì số đo khớp từng pixel, còn việc
        co giãn để GPU làm — y như nó đang làm với ``base`` và sprite, vốn
        cũng xuất ở 1,5×.
        """
        return float(self.manifest.image_scale)

    @property
    def box(self):
        return self._box

    def unmap(self, x: float, y: float) -> tuple[float, float]:
        """Pixel cửa sổ → toạ độ thiết kế (bắt chuột, U1.2)."""
        return self._box.unmap(x, y)

    def begin(self) -> None:
        GL.glDisable(GL.GL_DEPTH_TEST)
        GL.glDisable(GL.GL_CULL_FACE)
        GL.glEnable(GL.GL_BLEND)
        GL.glBlendFunc(GL.GL_SRC_ALPHA, GL.GL_ONE_MINUS_SRC_ALPHA)

    def end(self) -> None:
        GL.glDisable(GL.GL_BLEND)
        GL.glEnable(GL.GL_DEPTH_TEST)

    def viewport_px(self, rect) -> tuple[int, int, int, int]:
        """``rect`` thiết kế → ``(x, y, w, h)`` pixel GL (gốc dưới-trái)."""
        x, y, w, h = self._box.rect(rect)
        return (int(round(x)), int(round(self._fb[1] - y - h)),
                max(1, int(round(w))), max(1, int(round(h))))

    def quad(self, rect, texture, tint=(1.0, 1.0, 1.0, 1.0),
             uv=(0.0, 0.0, 1.0, 1.0)) -> None:
        """Một quad ở ``rect`` thiết kế, lấy vùng ``uv`` của texture."""
        if texture is None or rect[2] <= 0.0 or rect[3] <= 0.0:
            return
        x, y, w, h = self._box.rect(rect)
        fb_w, fb_h = self._fb
        x0, x1 = 2.0 * x / fb_w - 1.0, 2.0 * (x + w) / fb_w - 1.0
        y0, y1 = 1.0 - 2.0 * y / fb_h, 1.0 - 2.0 * (y + h) / fb_h
        u0, v0, u1, v1 = uv
        data = np.array([x0, y1, u0, v1, x1, y1, u1, v1,
                         x1, y0, u1, v0, x0, y1, u0, v1,
                         x1, y0, u1, v0, x0, y0, u0, v0], dtype=np.float32)
        GL.glBindBuffer(GL.GL_ARRAY_BUFFER, self._vbo)
        GL.glBufferSubData(GL.GL_ARRAY_BUFFER, 0, data.nbytes, data)
        self._shader.use()
        GL.glUniform1i(GL.glGetUniformLocation(self._shader.program_id,
                                               "uTexture"), 0)
        GL.glUniform4f(GL.glGetUniformLocation(self._shader.program_id,
                                               "uTint"), *tint)
        GL.glActiveTexture(GL.GL_TEXTURE0)
        GL.glBindTexture(GL.GL_TEXTURE_2D, texture)
        GL.glBindVertexArray(self._vao)
        GL.glDrawArrays(GL.GL_TRIANGLES, 0, 6)
        GL.glBindVertexArray(0)

    def fill(self, rect, colour) -> None:
        """Hình chữ nhật tô một màu (texture 1×1 trắng × tint)."""
        entry = self.cache.white()
        if entry:
            self.quad(rect, entry[0], colour)

    def frame(self, rect, colour, width: float) -> None:
        """Khung rỗng bốn thanh — vòng hover và vòng focus."""
        x, y, w, h = rect
        self.fill((x, y, w, width), colour)
        self.fill((x, y + h - width, w, width), colour)
        self.fill((x, y, width, h), colour)
        self.fill((x + w - width, y, width, h), colour)

    def rounded_frame(self, rect, colour, width: float, radius: float) -> None:
        """
        Khung rỗng bo góc — vòng hover/focus/selection quanh một ô BO GÓC.

        ``radius`` 0 (phần tử không bo, xem ``_ring_radius``) đi thẳng về
        ``frame()`` vuông góc: khỏi tốn một texture đệm cho trường hợp phổ
        biến nhất. ``radius`` > 0 dựng bằng PIL rồi đưa lên GPU, đệm theo
        khoá — cùng khuôn với ``cap``/``disc``/``pip`` ở trên.
        """
        if radius < 0.5:
            self.frame(rect, colour, width)
            return
        scale = self.scale
        w_px = max(8, int(round(rect[2] * scale)))
        h_px = max(8, int(round(rect[3] * scale)))
        key = ("rframe", w_px, h_px, round(width * scale, 2),
               round(radius * scale, 2))

        def build():
            image = Image.new("RGBA", (w_px, h_px), (0, 0, 0, 0))
            pen = ImageDraw.Draw(image)
            sw = max(1, int(round(width * scale)))
            half = sw / 2.0
            pen.rounded_rectangle(
                [half, half, w_px - 1 - half, h_px - 1 - half],
                radius=radius * scale, outline=(255, 255, 255, 255), width=sw)
            return np.asarray(image, dtype=np.uint8), None

        entry = self.pil.get(key, build)
        if entry:
            self.quad(rect, entry[0], colour)

    # ---- texture của một màn --------------------------------------------
    def bind_screen(self, screen: str, extra_files=()) -> None:
        """
        Nạp ``base``, ``top`` và sprite của màn — **một lần cho mỗi màn**.

        ``base``/``top`` thu về đúng bề rộng framebuffer: ảnh xuất ở 1920 px
        mà cửa sổ 1280 thì giữ nguyên cỡ chỉ tốn RAM GPU và fillrate. Sprite
        giữ cỡ gốc (``item_tile``/``weapon_tile`` xuất ở 4× để còn nét).
        """
        manifest = self.manifest
        record = manifest.screens[screen]
        big = max(1, int(self._fb[0]))
        wanted: dict[tuple, int | None] = {}
        for layer in ("base", "top"):
            path = manifest.path(record.get(layer))
            if path is not None:
                wanted[(str(path), big)] = big
        for sprite in record.get("sprites", ()):
            path = manifest.path(sprite.get("file"))
            if path is not None:
                wanted[(str(path), 0)] = None
        for relative in extra_files:
            path = manifest.path(relative)
            if path is not None:
                wanted[(str(path), 0)] = None
        key = (screen, big, tuple(sorted(extra_files)))
        if key == self._screen_key:
            return
        for old_key, entry in list(self._screen_tex.items()):
            if old_key not in wanted:
                GL.glDeleteTextures(1, [entry[0]])
                del self._screen_tex[old_key]
        for tex_key, max_width in wanted.items():
            if tex_key in self._screen_tex:
                continue
            array = load_rgba(tex_key[0], max_width=max_width)
            if array is None:
                continue
            self._screen_tex[tex_key] = (upload_rgba(array), array.shape[1],
                                         array.shape[0])
        self._screen_key = key
        logger.debug("UI v3: man %s — %d texture", screen, len(self._screen_tex))

    def screen_image(self, relative: str | None, *, big: bool = False):
        """``(tex, w, h)`` của một file đã nạp cho màn đang mở."""
        path = self.manifest.path(relative)
        if path is None:
            return None
        return self._screen_tex.get((str(path), int(self._fb[0]) if big else 0))

    def layer(self, rect, relative: str | None) -> None:
        """Vẽ ``base``/``top`` phủ kín ``rect``."""
        entry = self.screen_image(relative, big=True)
        if entry:
            self.quad(rect, entry[0])

    def sprite(self, rect, relative: str | None, alpha: float = 1.0,
               mirror: bool = False) -> None:
        """``mirror`` lật ngang bằng UV — không cần file sprite thứ hai."""
        entry = self.screen_image(relative)
        if entry:
            uv = (1.0, 0.0, 0.0, 1.0) if mirror else (0.0, 0.0, 1.0, 1.0)
            self.quad(rect, entry[0], (1.0, 1.0, 1.0, float(alpha)), uv)

    # ---- chữ -------------------------------------------------------------
    def _text_entry(self, runs, texts):
        """
        Texture + số đo của một dòng chữ (một baseline, nhiều run).

        Hai tỉ lệ, và phải tách bạch:

        - **Số đo** lấy ở ``k`` = 1,5, đúng tỉ lệ ``tools/preview.py`` dùng.
          Bước tiến từng glyph bị FreeType làm tròn về pixel nguyên, nên bề
          rộng chuỗi không tỉ lệ thuận với cỡ chữ; đo ở tỉ lệ khác là lệch
          tới 11 % (xem ``V3Painter.k``).
        - **Nét chữ** rasterise ở tỉ lệ framebuffer thật. Dựng ở 1,5× rồi để
          GL thu nhỏ bằng bilinear (không mipmap) làm chữ rỗ và mảnh đi:
          hình học đúng tới ±1 px mà ``compare.py`` vẫn 30–50.

        Nên: vị trí bút tính ở ``k`` rồi quy về pixel thiết kế, còn glyph vẽ
        bằng font cỡ ``k``. Không có kerning (PIL layout cơ bản — đã kiểm:
        tổng bước tiến từng glyph khớp đúng bề rộng cả chuỗi), nên vẽ rời
        từng glyph cho ra đúng chữ mà ``preview.py`` vẽ liền một hơi.

        Đã thử rasterise ở tỉ lệ framebuffer cho chữ nét hơn: nó **xa**
        ``expected/`` hơn chứ không gần hơn (ô tệ nhất 49,6 → 59,1). Lý do:
        ``expected/`` cũng là bản 1,5× bị thu nhỏ, nên muốn giống nó thì
        phải đi qua đúng cái phễu thu nhỏ ấy. Mà hệ số thu ở đây là 0,711 —
        vẫn nằm trong mip 0, nên GL chỉ lọc bilinear một nhịp, không nhoè.
        """
        k = self.k
        scale = k
        style = tuple((r.get("ff"), r.get("fw"), round(float(r.get("fs", 16)), 3),
                       r.get("col"), round(float(r.get("ls") or 0.0), 3),
                       bool(r.get("upper")))
                      for r in runs)
        key = ("text", style, tuple(texts), round(scale, 5))

        def build():
            items, total = [], 0.0
            for record, raw in zip(runs, texts):
                text = raw.upper() if record.get("upper") else raw
                size = float(record.get("fs", 16))
                metric = self.fonts.get(record.get("ff"), record.get("fw"),
                                        round(size * k))
                glyph = self.fonts.get(record.get("ff"), record.get("fw"),
                                       round(size * scale))
                spacing = float(record.get("ls") or 0.0)
                items.append((metric, glyph, text, _byte(record.get("col")),
                              spacing))
                total += _run_width(metric, text, spacing * k) / k
            biggest = max(runs, key=lambda r: float(r.get("fs", 16)))
            ascent_k, descent_k = self.fonts.get(
                biggest.get("ff"), biggest.get("fw"),
                round(float(biggest.get("fs", 16)) * k)).getmetrics()
            ascent, descent = ascent_k / k, descent_k / k
            width = max(1, int(math.ceil(total * scale)) + 2 * TEXT_PAD)
            height = max(1, int(math.ceil((ascent + descent) * scale))
                         + 2 * TEXT_PAD)
            image = Image.new("RGBA", (width, height), (0, 0, 0, 0))
            pen = ImageDraw.Draw(image)
            cursor = 0.0
            baseline = float(TEXT_PAD) + ascent * scale
            for metric, glyph, text, colour, spacing in items:
                for char in text:
                    pen.text((TEXT_PAD + cursor * scale, baseline), char,
                             font=glyph, fill=colour, anchor="ls")
                    cursor += metric.getlength(char) / k + spacing
            return (np.asarray(image, dtype=np.uint8),
                    (float(total), float(ascent), float(descent)))

        return self.pil.get(key, build)

    def line(self, rect, runs, texts, align: str = "left",
             alpha: float = 1.0) -> float:
        """
        Vẽ một dòng chữ theo đúng phép canh của preview.py.

        Trả mép phải của chuỗi, tính bằng **pixel thiết kế** (con trỏ nhập
        bám vào đó). Canh dọc: hộp dòng của run **lớn nhất** nằm giữa
        ``rect``. Canh ngang theo ``align`` trên tổng bề rộng mọi run.
        """
        entry = self._text_entry(runs, texts)
        if not entry:
            return 0.0
        texture, width_px, height_px, (total, ascent, descent) = entry
        scale = self.k
        x, y, w, h = (float(v) for v in rect)
        if align == "right":
            start = x + w - total
        elif align == "center":
            start = x + (w - total) * 0.5
        else:
            start = x
        baseline = y + (h - (ascent + descent)) * 0.5 + ascent
        self.quad((start - TEXT_PAD / scale,
                   baseline - ascent - TEXT_PAD / scale,
                   width_px / scale, height_px / scale),
                  texture, (1.0, 1.0, 1.0, float(alpha)))
        return start + total

    def text_width(self, runs, texts) -> float:
        """Bề rộng một dòng chữ, tính bằng **pixel thiết kế**."""
        entry = self._text_entry(runs, texts)
        return 0.0 if not entry else entry[3][0]

    # ---- hình khối bo góc ------------------------------------------------
    def rounded(self, rect, radius: float, colour) -> None:
        """
        Hình chữ nhật bo góc, **không** gọi PIL mỗi khung.

        Chia ba: hai đầu lấy từ một texture đệm cao đúng ``h`` và rộng
        ``2r+2``, khúc giữa là một ô tô màu phẳng. Thanh tiến độ đổi bề rộng
        mỗi khung nên dựng lại ảnh PIL theo từng bề rộng là đường chắc chắn
        vỡ ngưỡng FPS (luật F4).
        """
        x, y, w, h = (float(v) for v in rect)
        if w <= 0.0 or h <= 0.0:
            return
        radius = max(0.0, min(float(radius), h * 0.5, w * 0.5))
        if radius < 0.5:
            self.fill(rect, colour)
            return
        scale = self.scale
        cap_w = 2.0 * radius + 2.0
        key = ("cap", round(cap_w * scale), round(h * scale),
               round(radius * scale, 2))

        def build():
            width = max(2, int(round(cap_w * scale)))
            height = max(2, int(round(h * scale)))
            image = Image.new("RGBA", (width, height), (0, 0, 0, 0))
            ImageDraw.Draw(image).rounded_rectangle(
                [0, 0, width - 1, height - 1], radius=radius * scale,
                fill=(255, 255, 255, 255))
            return np.asarray(image, dtype=np.uint8), None

        entry = self.pil.get(key, build)
        if not entry:
            return
        texture = entry[0]
        half = radius + 1.0
        if w <= cap_w:
            # Hẹp hơn cả hai đầu: kéo nguyên ảnh đệm. README bảo thanh không
            # bao giờ hẹp hơn 2×radius, nên đây chỉ là lưới an toàn.
            self.quad((x, y, w, h), texture, colour)
            return
        middle = w - 2.0 * half
        self.quad((x, y, half, h), texture, colour, (0.0, 0.0, 0.5, 1.0))
        self.quad((x + half, y, middle, h), texture, colour,
                  (0.5, 0.0, 0.5, 1.0))
        self.quad((x + w - half, y, half, h), texture, colour,
                  (0.5, 0.0, 1.0, 1.0))

    def disc(self, rect, colour) -> None:
        """
        Hình tròn đặc, mềm biên. Texture TRẮNG đệm theo cỡ, màu đi qua tint.

        Vì sao không dùng ``pip``: khoá đệm của nó gồm cả màu, mà lỗ đạn thì
        mờ dần từng khung — mỗi khung một màu là mỗi khung một texture mới.
        Tách màu ra khỏi khoá thì cả vệt mờ dần chỉ tốn đúng một texture.
        """
        scale = self.scale
        size = max(1, int(round(float(rect[2]) * scale)))
        key = ("disc", size, max(1, int(round(float(rect[3]) * scale))))

        def build():
            width = size + 1
            height = max(1, int(round(float(rect[3]) * scale))) + 1
            image = Image.new("RGBA", (width, height), (0, 0, 0, 0))
            ImageDraw.Draw(image).ellipse([0, 0, width - 1, height - 1],
                                          fill=(255, 255, 255, 255))
            return np.asarray(image, dtype=np.uint8), None

        entry = self.pil.get(key, build)
        if entry:
            self.quad((rect[0], rect[1], entry[1] / scale, entry[2] / scale),
                      entry[0], colour)

    def pip(self, rect, shape: str, radius: float, style: dict) -> None:
        """
        Một pip: bầu dục hoặc chữ nhật bo góc, tô và/hoặc viền.

        ``fill_op`` được tôn trọng thật. ``preview.py`` vẽ pip mờ bằng PIL
        lên canvas RGBA rồi ``convert('RGB')`` ở cuối — thao tác đó **vứt**
        kênh alpha chứ không trộn, nên trong ``expected/`` pip "off" hiện ra
        đặc kịt. ``design/*.jpg`` (README: "the visual truth") cho thấy nó
        phải mờ, và đó là thứ dãy pip dùng để nói "lớp này chưa tới".
        """
        scale = self.k
        key = ("pip", shape, round(rect[2] * scale), round(rect[3] * scale),
               round(radius * scale, 2), style.get("fill"),
               round(float(style.get("fill_op", 1.0) or 1.0), 3),
               style.get("stroke"), round(float(style.get("sw", 1) or 1) * scale, 2))

        def build():
            width = max(1, int(round(rect[2] * scale))) + 1
            height = max(1, int(round(rect[3] * scale))) + 1
            image = Image.new("RGBA", (width, height), (0, 0, 0, 0))
            pen = ImageDraw.Draw(image)
            fill = (_byte(style["fill"], float(style.get("fill_op", 1.0) or 1.0))
                    if style.get("fill") else None)
            outline = _byte(style["stroke"]) if style.get("stroke") else None
            stroke_px = max(1, int(round(float(style.get("sw", 1) or 1) * scale)))
            box = [0, 0, width - 1, height - 1]
            if shape == "ellipse":
                pen.ellipse(box, fill=fill, outline=outline, width=stroke_px)
            else:
                pen.rounded_rectangle(box, radius=radius * scale, fill=fill,
                                      outline=outline, width=stroke_px)
            return np.asarray(image, dtype=np.uint8), None

        entry = self.pil.get(key, build)
        if entry:
            self.quad((rect[0], rect[1], entry[1] / scale, entry[2] / scale),
                      entry[0])

    def pill(self, record: dict, text: str, style: dict) -> None:
        """
        Viên thuốc: nền bo góc + chữ canh giữa, dựng thành **một** texture.

        Bề rộng = bề rộng chữ + 2×``pad``, đúng công thức của preview.py (kể
        cả chỗ nó nhân ``ls`` với ``len`` chứ không phải ``len - 1``).
        """
        k = self.k
        scale = k
        font_spec = record["font"]
        x, y, _w, h = (float(v) for v in record["rect"])
        radius = float(record.get("radius", 0.0))
        pad = float(record.get("pad", 0.0))
        key = ("pill", text, font_spec["ff"], font_spec["fw"],
               round(float(font_spec["fs"]), 3),
               round(float(font_spec.get("ls") or 0.0), 3),
               style.get("bg"), style.get("fg"), round(h, 2), round(radius, 2),
               round(pad, 2), round(scale, 5))

        def build():
            size = float(font_spec["fs"])
            metric = self.fonts.get(font_spec["ff"], font_spec["fw"],
                                    round(size * k))
            glyph = self.fonts.get(font_spec["ff"], font_spec["fw"],
                                   round(size * scale))
            spacing = float(font_spec.get("ls") or 0.0)
            # Công thức bề rộng của preview.py, kể cả chỗ nó nhân ``ls`` với
            # ``len`` chứ không phải ``len - 1``.
            design_w = metric.getlength(text) / k + spacing * len(text) + 2.0 * pad
            width = max(2, int(round(design_w * scale))) + 1
            height = max(2, int(round(h * scale))) + 1
            image = Image.new("RGBA", (width, height), (0, 0, 0, 0))
            pen = ImageDraw.Draw(image)
            pen.rounded_rectangle([0, 0, width - 1, height - 1],
                                  radius=radius * scale, fill=_byte(style.get("bg")))
            total = _run_width(metric, text, spacing * k) / k
            ascent_k, descent_k = metric.getmetrics()
            ascent, descent = ascent_k / k, descent_k / k
            cursor = (width - 1 - total * scale) * 0.5
            baseline = (height - 1 - (ascent + descent) * scale) * 0.5 + ascent * scale
            colour = _byte(style.get("fg"))
            for char in text:
                pen.text((cursor, baseline), char, font=glyph, fill=colour,
                         anchor="ls")
                cursor += (metric.getlength(char) / k + spacing) * scale
            return np.asarray(image, dtype=np.uint8), design_w

        entry = self.pil.get(key, build)
        if entry:
            self.quad((x, y, entry[1] / scale, entry[2] / scale), entry[0])

    # ---- ảnh người dùng --------------------------------------------------
    def cover(self, rect, source, clip=None) -> bool:
        """Ảnh cover-fit (phủ kín, cắt giữa) vào ``rect``; ``clip`` thu hẹp."""
        entry = self.cache.image(source)
        if not entry:
            return False
        _tex, tex_w, tex_h = entry
        factor = max(rect[2] / tex_w, rect[3] / tex_h)
        width, height = tex_w * factor, tex_h * factor
        placed = (rect[0] + (rect[2] - width) * 0.5,
                  rect[1] + (rect[3] - height) * 0.5, width, height)
        GL.glEnable(GL.GL_SCISSOR_TEST)
        for area in (clip or [rect]):
            GL.glScissor(*self.viewport_px(area))
            self.quad(placed, entry[0])
        GL.glDisable(GL.GL_SCISSOR_TEST)
        return True

    def release(self) -> None:
        self.cache.release()
        self.pil.release()
        for entry in self._screen_tex.values():
            GL.glDeleteTextures(1, [entry[0]])
        self._screen_tex.clear()
        if self._vao:
            GL.glDeleteVertexArrays(1, [self._vao])
        if self._vbo:
            GL.glDeleteBuffers(1, [self._vbo])
        self._shader.release()


def _run_width(font, text: str, spacing: float) -> float:
    """Bề rộng một run, kể cả letter-spacing (y như ``preview.run_width``)."""
    if spacing:
        return font.getlength(text) + spacing * max(0, len(text) - 1)
    return font.getlength(text)


class V3State:
    """
    Mọi thứ đổi theo thời gian của một màn v3.

    Lớp vẽ cố tình **không biết** luật nào: ``v3_values.py`` quyết định sprite
    nào bật, chữ nào thay, thanh đầy bao nhiêu, rồi đổ vào đây. Nhờ vậy lượt
    chụp (``--ui-shots``) chỉ là một bộ giá trị khác, không phải một nhánh mã
    khác — mà README nói rõ ``expected/`` chính là chế độ chụp.
    """

    __slots__ = ("texts", "colours", "hidden", "sprites", "borrowed", "bars",
                 "pips", "pills", "photo", "camera_tex", "viewport_draw",
                 "hover", "focus", "caret", "alpha", "shot", "message", "wrap",
                 "extra", "mirror", "aim", "shots", "aim_ready", "confirm")

    def __init__(self) -> None:
        #: ``key`` → chuỗi (thay run đầu) hoặc danh sách chuỗi (thay từng run).
        self.texts: dict = {}
        #: ``key`` → mã màu, hoặc danh sách mã màu theo từng run.
        self.colours: dict = {}
        self.hidden: set = set()
        #: ``key`` → ``True`` (vẽ ở ``rect`` của nó) hoặc ``rect`` đặt chỗ khác.
        self.sprites: dict = {}
        #: Sprite app tự đặt chỗ: ``nhãn`` → ``(màn, key, rect)``. Nhãn là duy
        #: nhất nên **một** sprite vẽ được nhiều lần (ba ô vật phẩm, ba hàng
        #: dấu kiểm), và ``màn`` khác màn đang mở thì là sprite vay — HUD lấy
        #: ``item_tile:*``/``weapon_tile:*`` của màn trang bị (README).
        self.borrowed: dict = {}
        #: ``key`` → tỉ lệ đầy 0–1.
        self.bars: dict = {}
        #: ``key`` → số pip đang bật, hoặc danh sách cờ, hoặc ``("n", trần)``.
        self.pips: dict = {}
        #: ``key`` → (chuỗi, tên kiểu).
        self.pills: dict = {}
        self.photo = None
        self.camera_tex = None
        self.viewport_draw = None
        self.hover: str | None = None
        self.focus: str | None = None
        #: ``key`` của ô chữ đang nhận chữ gõ — vẽ con trỏ sau nó.
        self.caret: str | None = None
        #: ``key`` → độ mờ 0–1 (ô vật phẩm cấp 1 vẽ 45 %).
        self.alpha: dict = {}
        self.shot = False
        self.message: str = ""
        self.wrap = True
        #: TASK_shared_album_1111: đường dẫn ảnh đang chờ xác nhận chia sẻ;
        #: "" (mặc định của ``str``) = không có hộp thoại nào đang mở.
        self.confirm: str = ""
        #: Nhãn sprite cần lật ngang (mũi tên nghiêng đầu khi góc > 0).
        self.mirror: set = set()
        #: Tâm ngắm ``(x, y)`` trong toạ độ thiết kế; ``None`` = không vẽ.
        self.aim = None
        #: Vết đạn ``(x, y, tuổi giây, trúng?)``, mới nhất ở cuối.
        self.shots: list = []
        #: Súng đã hồi tới đâu, 0–1. Dưới 1 thì tâm ngắm bung ra và mờ đi.
        self.aim_ready: float = 1.0
        #: Dòng chữ app tự đặt chỗ: ``(rect, runs, texts, align)``. Bản vẽ chỉ
        #: xuất MỘT thực thể cho mỗi kiểu ô vật phẩm / dòng điểm, còn ván chơi
        #: cần ba ô và số dòng thay đổi — chỗ lặp lại nằm ở đây, không phải
        #: trong manifest.
        self.extra: list = []


def shot_state(manifest: V3Manifest, screen: str) -> V3State:
    """
    Trạng thái của **chế độ chụp** = đúng luật ``expected/`` trong README.

    Mọi chữ lấy ``sample``, mọi sprite ``default: true`` vẽ ở ``rect`` của nó,
    pip theo ``sample_on``, thanh theo ``sample_frac``, viên thuốc lấy kiểu
    đầu tiên, ô ảnh lấy ``sample/sample_room.png``, không hover/focus/caret.
    Riêng ``wrap`` phải TẮT: ``preview.py`` bỏ qua cờ đó, nên ``expected/``
    vẽ ``tip_text`` thành một dòng.
    """
    record = manifest.screens[screen]
    state = V3State()
    state.shot = True
    state.wrap = False
    state.photo = manifest.sample_photo()
    state.sprites = {s["key"]: True for s in record.get("sprites", ())
                     if s.get("default")}
    state.bars = {b["key"]: float(b.get("sample_frac", 0.0))
                  for b in record.get("bars", ())}
    state.pips = {p["key"]: list(p.get("sample_on", ()))
                  for p in record.get("pips", ())}
    state.pills = {p["key"]: (p.get("sample") or "",
                              next(iter(p["styles"]), None))
                   for p in record.get("pills", ())}
    return state


def _texts_for(record: dict, state: V3State) -> list[str]:
    """Chuỗi của từng run sau khi thay giá trị sống."""
    runs = record["runs"]
    value = state.texts.get(record["key"])
    if value is None:
        return [str(r.get("c") or "") for r in runs]
    if isinstance(value, (list, tuple)):
        parts = list(value)
        # Nhiều giá trị hơn số run thì DỒN phần thừa vào run cuối. Cắt bỏ
        # chúng là mất chữ mà không báo gì: ô "LAYER 2/4" chỉ có một run, đưa
        # vào ``["2", "/4"]`` thì màn chơi hiện mỗi "2".
        if len(parts) > len(runs) and runs:
            parts = parts[:len(runs) - 1] + ["".join(
                "" if p is None else str(p) for p in parts[len(runs) - 1:])]
        return [str(r.get("c") or "") if v is None else str(v)
                for r, v in zip(runs, parts + [None] * len(runs))]
    # Một giá trị cho nhiều run: run đầu là con số sống, run sau giữ nguyên
    # ("7" + "/12" — README mục Draw order 8).
    return [str(value)] + [str(r.get("c") or "") for r in runs[1:]]


def _runs_for(record: dict, state: V3State) -> list[dict]:
    """Run sau khi thay màu sống (``key`` hoặc từng run)."""
    runs = record["runs"]
    colour = state.colours.get(record["key"])
    if colour is None:
        return runs
    values = colour if isinstance(colour, (list, tuple)) else [colour]
    out = []
    for index, run in enumerate(runs):
        new = values[index] if index < len(values) else None
        out.append({**run, "col": new} if new else run)
    return out


def _wrap_lines(painter: V3Painter, runs, texts, width: float) -> list[list[str]]:
    """Cắt dòng theo bề rộng ``rect.w`` (chỉ dùng cho text có ``wrap``)."""
    words = (texts[0] or "").split()
    if not words:
        return [texts]
    lines, current = [], ""
    for word in words:
        probe = f"{current} {word}".strip()
        if current and painter.text_width(runs[:1], [probe]) > width:
            lines.append(current)
            current = word
        else:
            current = probe
    lines.append(current)
    return [[line] + texts[1:] for line in lines]


def draw_screen(painter: V3Painter, manifest: V3Manifest, screen: str,
                state: V3State) -> None:
    """Vẽ trọn một màn theo thứ tự lớp của README."""
    record = manifest.screens[screen]
    full = (0.0, 0.0, DESIGN_SIZE[0], DESIGN_SIZE[1])
    painter.layer(full, record.get("base"))
    _draw_photos(painter, manifest, record, state)
    painter.layer(full, record.get("top"))
    _draw_bars(painter, record, state)
    _draw_pips(painter, record, state)
    _draw_pills(painter, record, state)
    _draw_sprites(painter, manifest, screen, record, state)
    _draw_texts(painter, record, state)
    _draw_choose_photo_extra(painter, screen, state)
    _draw_shots(painter, state)
    _draw_aim(painter, state)
    _draw_rings(painter, record, state)
    _draw_toast(painter, state, screen)
    _draw_confirm_dialog(painter, state)


def _draw_photos(painter, manifest, record, state) -> None:
    for photo in record.get("photos", ()):
        kind, rect = photo.get("kind"), photo["rect"]
        if kind == "photo":
            source = state.photo
            if source is not None and Path(source).exists():
                painter.cover(rect, source, photo.get("clip"))
            elif state.shot:
                painter.cover(rect, manifest.sample_photo(), photo.get("clip"))
        elif kind == "camera":
            if state.shot:
                painter.fill(rect, rgba("#%02X%02X%02X" % SHOT_CAMERA_RGB))
            elif state.camera_tex is not None:
                _cover_texture(painter, rect, state.camera_tex, mirror=True)
        elif kind == "viewport":
            if state.shot:
                painter.fill(rect, rgba("#%02X%02X%02X" % SHOT_VIEWPORT_RGB))
            elif state.viewport_draw is not None:
                state.viewport_draw(rect)


def _cover_texture(painter, rect, entry, mirror: bool = False) -> None:
    """Texture cover-fit vào ``rect`` (khung webcam, lật ngang nếu cần)."""
    texture, tex_w, tex_h = entry
    factor = max(rect[2] / tex_w, rect[3] / tex_h)
    width, height = tex_w * factor, tex_h * factor
    uv = (1.0, 0.0, 0.0, 1.0) if mirror else (0.0, 0.0, 1.0, 1.0)
    GL.glEnable(GL.GL_SCISSOR_TEST)
    GL.glScissor(*painter.viewport_px(rect))
    painter.quad((rect[0] + (rect[2] - width) * 0.5,
                  rect[1] + (rect[3] - height) * 0.5, width, height),
                 texture, uv=uv)
    GL.glDisable(GL.GL_SCISSOR_TEST)


def _draw_bars(painter, record, state) -> None:
    for bar in record.get("bars", ()):
        key = bar["key"]
        if key in state.hidden:
            continue
        x, y, w, h = (float(v) for v in bar["rect"])
        radius = float(bar.get("radius", 0.0))
        fraction = state.bars.get(key)
        if fraction is None:
            fraction = float(bar.get("sample_frac", 0.0))
        fraction = max(0.0, min(1.0, float(fraction)))
        if bar.get("rest_col"):
            painter.rounded((x, y, w, h), radius, rgba(bar["rest_col"]))
        # Không bao giờ hẹp hơn 2×radius (README): một viên bo góc bị bóp
        # dưới mức đó thành vệt nhòn nhọn, không còn là "thanh đầy 0 %".
        painter.rounded((x, y, max(w * fraction, 2.0 * radius), h), radius,
                        rgba(bar["col"]))


def _pip_flags(spec: dict, state: V3State) -> list[bool]:
    """Cờ bật/tắt từng pip: giá trị sống, nếu không thì ``sample_on``."""
    total = len(spec["rects"])
    value = state.pips.get(spec["key"])
    if value is None:
        flags = list(spec.get("sample_on", ()))
        return [bool(flags[i]) if i < len(flags) else False for i in range(total)]
    if isinstance(value, (list, tuple)):
        return [bool(value[i]) if i < len(value) else False for i in range(total)]
    return [i < int(value) for i in range(total)]


def _draw_pips(painter, record, state) -> None:
    for spec in record.get("pips", ()):
        key = spec["key"]
        if key in state.hidden:
            continue
        flags = _pip_flags(spec, state)
        limit = state.pips.get(f"{key}:limit")
        for index, rect in enumerate(spec["rects"]):
            if limit is not None and index >= int(limit):
                continue          # pip vượt quá N thì ẩn hẳn (HUD lớp/đạn)
            style = spec["on"] if flags[index] else spec["off"]
            painter.pip(rect, spec.get("shape", "rect"),
                        float(spec.get("radius", 0.0)), style)


def _draw_pills(painter, record, state) -> None:
    for spec in record.get("pills", ()):
        key = spec["key"]
        if key in state.hidden:
            continue
        text, style_name = state.pills.get(key, (None, None))
        if text is None:
            text = spec.get("sample") or ""
        styles = spec["styles"]
        style = styles.get(style_name) if style_name else None
        if style is None:
            style = next(iter(styles.values()))
        painter.pill(spec, str(text), style)


def _draw_sprites(painter, manifest, screen, record, state) -> None:
    for sprite in record.get("sprites", ()):
        key = sprite["key"]
        if key in state.hidden or key not in state.sprites:
            continue
        where = state.sprites[key]
        rect = sprite["rect"] if where is True else where
        painter.sprite(rect, sprite.get("file"),
                       float(state.alpha.get(key, 1.0)), key in state.mirror)
    for label, (other, key, rect) in state.borrowed.items():
        borrowed = manifest.sprite(other, key)
        if borrowed is not None:
            painter.sprite(rect, borrowed.get("file"),
                           float(state.alpha.get(label, 1.0)),
                           label in state.mirror)
    _ = screen


def _draw_texts(painter, record, state) -> None:
    for spec in record.get("texts", ()):
        key = spec["key"]
        if key in state.hidden:
            continue
        runs = _runs_for(spec, state)
        texts = _texts_for(spec, state)
        alpha = float(state.alpha.get(key, 1.0))
        rect = [float(v) for v in spec["rect"]]
        if spec.get("wrap") and state.wrap:
            line_h = float(spec.get("line_h") or rect[3])
            lines = _wrap_lines(painter, runs, texts, rect[2])
            for index, parts in enumerate(lines):
                painter.line((rect[0], rect[1] + index * line_h, rect[2], line_h),
                             runs, parts, spec.get("al", "left"), alpha)
            continue
        right = painter.line(rect, runs, texts, spec.get("al", "left"), alpha)
        if state.caret == key:
            painter.fill((right + CARET_GAP,
                          rect[1] + (rect[3] - CARET_SIZE[1]) * 0.5,
                          CARET_SIZE[0], CARET_SIZE[1]), rgba(CARET_COL))
    for rect, runs, texts, align in state.extra:
        painter.line(rect, runs, texts, align)


def _hole(painter, x: float, y: float, size: float, colour) -> None:
    """Một lớp tròn của lỗ đạn, canh giữa ``(x, y)``."""
    painter.disc((x - size * 0.5, y - size * 0.5, size, size), colour)


def _draw_shots(painter, state) -> None:
    """
    Lỗ đạn ở chỗ vừa bắn, cộng một chớp xanh khi trúng bọ.

    Giữ nguyên lỗ gần như suốt tuổi thọ rồi mới mờ hẳn ở quãng cuối: lỗ đạn
    trên ảnh thì không nhạt dần ngay, và người chơi cần nhìn lại được mình
    vừa bắn chệch về phía nào. Giữ vài phát gần nhất chứ không chỉ phát cuối.
    """
    if state.shot:
        return
    for x, y, age, hit in state.shots:
        age = float(age)
        if age >= HOLE_LIFE_S:
            continue
        # Mờ dần CHỈ ở quãng cuối, trước đó giữ nguyên độ đậm.
        left = min(1.0, max(0.0, (HOLE_LIFE_S - age) / HOLE_FADE_S))
        _hole(painter, x, y, HOLE_DUST, rgba(HOLE_DUST_COL, 0.22 * left))
        _hole(painter, x, y, HOLE_RING, rgba(HOLE_RING_COL, 0.72 * left))
        _hole(painter, x, y, HOLE_CORE, rgba(HOLE_CORE_COL, 0.92 * left))
        if hit and age < HIT_FLASH_S:
            flash = 1.0 - age / HIT_FLASH_S
            _hole(painter, x, y, HIT_FLASH_SIZE * (1.0 - flash * 0.5),
                  rgba(HIT_FLASH_COL, 0.55 * flash))


def _draw_toast(painter, state, screen=None) -> None:
    """
    Dòng báo nổi ở giữa dưới.

    Thiếu nó thì mọi lời ``v3_flow.say`` — "Code copied", "Night theme is not
    in v2 yet", "Reloading" — được đặt rồi **không ai vẽ**, nên người dùng bấm
    nút mà tưởng như không có gì xảy ra. Toast của renderer không thay được:
    nó vẽ trong pass cảnh 3D, mà màn không có ô viewport thì pass ấy bị cắt
    còn 1 pixel.
    """
    if not state.message or state.shot:
        return
    x, y, w, h = TOAST_RECT
    if screen == CHOOSE_PHOTO_SCREEN:
        y = 574.0
    text = str(state.message)
    width = min(w, painter.text_width([TOAST_FONT], [text]) + 2 * TOAST_RADIUS)
    left = x + (w - width) * 0.5
    painter.rounded((left, y, width, h), TOAST_RADIUS, rgba(TOAST_BG, 0.92))
    painter.line((left, y, width, h), [TOAST_FONT], [text], "center")


def _draw_choose_photo_extra(painter, screen, state) -> None:
    """Nút "Share to album" / "Random from album" — xem ``SHARE_BTN_RECT``."""
    if state.shot or screen != CHOOSE_PHOTO_SCREEN:
        return
    for rect, label_key, alpha_key, default in (
            (SHARE_BTN_RECT, "share_to_album_label", "share_to_album_btn",
             "Share to album"),
            (RANDOM_BTN_RECT, "random_from_album_label", "random_from_album_btn",
             "Random from album")):
        label = state.texts.get(label_key, default)
        alpha = float(state.alpha.get(alpha_key, 1.0))
        painter.rounded(rect, BTN_CORNER_RADIUS_PX, rgba("#FFFFFF"))
        painter.rounded_frame(rect, rgba(BUTTON_PRIMARY_COL, alpha), 1.5,
                              BTN_CORNER_RADIUS_PX)
        font = dict(SHARE_BTN_FONT)
        if rect == SHARE_BTN_RECT:
            font["fs"] = 12.0
        # The disabled label must fit even in the small caption-row button.
        width = painter.text_width([font], [str(label)])
        if width > rect[2] - 16:
            font["fs"] *= (rect[2] - 16) / width
        painter.line(rect, [font], [str(label)], "center", alpha)


def _draw_confirm_dialog(painter, state) -> None:
    """Hộp xác nhận chia sẻ — nổi trên mọi thứ, kể cả toast (P2.3)."""
    if not state.confirm or state.shot:
        return
    painter.fill((0.0, 0.0, DESIGN_SIZE[0], DESIGN_SIZE[1]), rgba("#0B1520", 0.45))
    x, y, w, h = CONFIRM_CARD_RECT
    painter.rounded((x, y, w, h), 18.0, rgba("#FFFFFF"))
    pad, line_h = 28.0, 22.0
    lines = _wrap_lines(painter, [CONFIRM_TEXT_FONT], [CONFIRM_TEXT], w - 2 * pad)
    for index, parts in enumerate(lines):
        painter.line((x + pad, y + pad + index * line_h, w - 2 * pad, line_h),
                     [CONFIRM_TEXT_FONT], parts, "left")
    for name, rect, label, bg in (
            ("btn_confirm_no", CONFIRM_NO_RECT, "No", "#DCE9F7"),
            ("btn_confirm_yes", CONFIRM_YES_RECT, "Yes", BUTTON_PRIMARY_COL)):
        fg = INK if bg != BUTTON_PRIMARY_COL else "#FFFFFF"
        painter.rounded(rect, 10.0, rgba(bg))
        painter.line(rect, [{**CONFIRM_BTN_FONT, "col": fg}], [label], "center")
        if state.hover == name:
            painter.rounded_frame(rect, rgba(CARET_COL), HOVER_PX, 10.0)


def _draw_aim(painter, state) -> None:
    """
    Dấu ``+`` ở tâm ngắm: bốn nhánh, chừa một khoảng hở ở giữa.

    Chừa hở để chính cái tâm — chỗ viên đạn sẽ tới — không bị nét vẽ che.
    """
    if state.aim is None or state.shot:
        return
    x, y = float(state.aim[0]), float(state.aim[1])
    ready = max(0.0, min(1.0, float(state.aim_ready)))
    # Chưa hồi xong: bốn nhánh dạt ra và nhạt bớt, rồi khép lại dần.
    spread = AIM_RECOIL * (1.0 - ready)
    fade = AIM_DIM + (1.0 - AIM_DIM) * ready
    for base, grow in ((AIM_EDGE_COL, AIM_EDGE), (AIM_COL, 0.0)):
        alpha = (0.85 if grow else 1.0) * fade
        colour = rgba(base, alpha)
        half = AIM_THICK * 0.5 + grow
        inner = AIM_GAP + spread - grow
        outer = AIM_ARM + spread + grow
        painter.fill((x - outer, y - half, outer - inner, half * 2.0), colour)
        painter.fill((x + inner, y - half, outer - inner, half * 2.0), colour)
        painter.fill((x - half, y - outer, half * 2.0, outer - inner), colour)
        painter.fill((x - half, y + inner, half * 2.0, outer - inner), colour)


def _draw_rings(painter, record, state) -> None:
    """Vòng hover và vòng focus — lớp cuối, sau mọi thứ (README mục 9)."""
    if state.shot:
        return
    for hotspot in record.get("hotspots", ()):
        name, rect = hotspot["n"], hotspot["rect"]
        radius = _ring_radius(name, rect)
        if name == state.focus:
            ring_rect = (rect[0] - FOCUS_INSET, rect[1] - FOCUS_INSET,
                        rect[2] + 2 * FOCUS_INSET, rect[3] + 2 * FOCUS_INSET)
            # 01:18 + 03:xx: vòng focus/hover bốn thanh vuông góc quanh một ô
            # BO GÓC nhìn lệch hẳn (field_user/field_pass, level_row_*, ...).
            # Một helper duy nhất (``rounded_frame``) cho mọi hotspot; bán
            # kính 0 (phần tử không bo) tự quay lại khung vuông như hôm nay.
            painter.rounded_frame(ring_rect, rgba(CARET_COL), FOCUS_PX,
                                 radius + FOCUS_INSET if radius else 0.0)
        elif name == state.hover:
            painter.rounded_frame(rect, rgba(CARET_COL, HOVER_ALPHA),
                                 HOVER_PX, radius)


def hit_test(manifest: V3Manifest, screen: str, design_x: float,
             design_y: float) -> str | None:
    """Tên hotspot dưới con trỏ; ``None`` nếu không có."""
    for hotspot in manifest.screens[screen].get("hotspots", ()):
        x, y, w, h = hotspot["rect"]
        if x <= design_x <= x + w and y <= design_y <= y + h:
            return hotspot["n"]
    return None
