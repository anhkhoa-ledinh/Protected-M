"""Picture-in-picture webcam preview HUD (screen-space, F4 toggle)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import numpy.typing as npt
from OpenGL import GL

from src.camera.capture import FramePacket
from src.renderer.shader import ShaderProgram


@dataclass(frozen=True)
class WebcamPreviewSettings:
    """Layout and behaviour for the webcam PiP overlay."""

    enabled: bool = True
    corner: str = "bottom-right"
    scale: float = 0.18
    margin_px: int = 16
    mirror: bool = True
    draw_landmarks: bool = True


@dataclass(frozen=True)
class TrackingPreviewSnapshot:
    """Landmark overlay data from the tracking thread (read-only copy)."""

    detected: bool = False
    confidence: float = 0.0
    image_width: int = 0
    image_height: int = 0
    pnp_points: npt.NDArray[np.float64] | None = None
    left_iris: npt.NDArray[np.float64] | None = None
    right_iris: npt.NDArray[np.float64] | None = None


class WebcamPreviewOverlay:
    """
    Render a live webcam feed as a 2D HUD quad after the 3D scene.

    Uses the same capture stream as tracking (peek, never opens a second
    device).  Texture uploads use ``glTexSubImage2D`` when dimensions are
    unchanged.  Horizontal mirroring is cosmetic for the preview only.
    """

    def __init__(self, shader_dir: Path, settings: WebcamPreviewSettings) -> None:
        self._settings = settings
        self._visible = settings.enabled
        self._shader = ShaderProgram(shader_dir / "hud.vert", shader_dir / "hud.frag")
        self._texture = 0
        self._vao = 0
        self._vbo = 0
        self._ebo = 0
        self._tex_width = 0
        self._tex_height = 0
        self._last_frame_index = -1
        self._last_aspect = 0.0
        self._ready = False

    @property
    def visible(self) -> bool:
        return self._visible

    def toggle(self) -> None:
        self._visible = not self._visible

    def initialize(self) -> None:
        self._texture = GL.glGenTextures(1)
        self._vao = GL.glGenVertexArrays(1)
        self._vbo = GL.glGenBuffers(1)
        self._ebo = GL.glGenBuffers(1)
        indices = np.array([0, 1, 2, 0, 2, 3], dtype=np.uint32)
        GL.glBindVertexArray(self._vao)
        GL.glBindBuffer(GL.GL_ARRAY_BUFFER, self._vbo)
        GL.glBufferData(GL.GL_ARRAY_BUFFER, 64, None, GL.GL_DYNAMIC_DRAW)
        GL.glBindBuffer(GL.GL_ELEMENT_ARRAY_BUFFER, self._ebo)
        GL.glBufferData(
            GL.GL_ELEMENT_ARRAY_BUFFER, indices.nbytes, indices, GL.GL_STATIC_DRAW
        )
        GL.glEnableVertexAttribArray(0)
        GL.glVertexAttribPointer(0, 2, GL.GL_FLOAT, GL.GL_FALSE, 16, None)
        GL.glEnableVertexAttribArray(1)
        GL.glVertexAttribPointer(1, 2, GL.GL_FLOAT, GL.GL_FALSE, 16, GL.ctypes.c_void_p(8))
        GL.glBindVertexArray(0)
        self._ready = True

    def draw(
        self,
        frame: FramePacket | None,
        tracking: TrackingPreviewSnapshot,
        window_width: int,
        window_height: int,
    ) -> None:
        if not self._visible or not self._ready:
            return

        # The tracking thread consumes frames with pop_latest(), which CLEARS
        # the capture queue, so peek_frame() returns None most of the time.
        # Bailing out here would leave the PiP unpainted for that frame — and
        # since the framebuffer is cleared every frame, the overlay would
        # strobe.  Re-draw the cached texture instead of dropping out.
        if frame is None or frame.frame_index == self._last_frame_index:
            if self._tex_width > 0 and self._last_aspect > 0.0:
                self._draw_quad(window_width, window_height, self._last_aspect)
            return

        rgb = self._prepare_preview_image(frame, tracking)
        self._upload_texture(rgb)
        self._last_frame_index = frame.frame_index
        self._last_aspect = frame.width / frame.height
        self._draw_quad(window_width, window_height, self._last_aspect)

    def release(self) -> None:
        if self._texture:
            GL.glDeleteTextures(1, [self._texture])
            self._texture = 0
        if self._vao:
            GL.glDeleteVertexArrays(1, [self._vao])
            self._vao = 0
        if self._vbo:
            GL.glDeleteBuffers(1, [self._vbo])
            self._vbo = 0
        if self._ebo:
            GL.glDeleteBuffers(1, [self._ebo])
            self._ebo = 0
        self._shader.release()

    def _prepare_preview_image(
        self,
        frame: FramePacket,
        tracking: TrackingPreviewSnapshot,
    ) -> npt.NDArray[np.uint8]:
        image = frame.image
        src_h, src_w = image.shape[:2]
        preview_w = max(64, int(src_w * self._settings.scale))
        preview_h = max(36, int(preview_w * src_h / src_w))
        small = cv2.resize(image, (preview_w, preview_h), interpolation=cv2.INTER_LINEAR)

        # Mirror BEFORE drawing the overlay.  _draw_landmarks already mirrors
        # each point in pt(); flipping the canvas afterwards would undo that
        # and leave the skeleton reversed against the face — and it would also
        # render the confidence label as mirror text.
        if self._settings.mirror:
            small = cv2.flip(small, 1)

        if self._settings.draw_landmarks and tracking.detected:
            self._draw_landmarks(small, tracking, src_w, src_h)

        return cv2.cvtColor(small, cv2.COLOR_BGR2RGB)

    def _draw_landmarks(
        self,
        canvas: npt.NDArray[np.uint8],
        tracking: TrackingPreviewSnapshot,
        src_w: int,
        src_h: int,
    ) -> None:
        pw, ph = canvas.shape[1], canvas.shape[0]
        sx, sy = pw / float(src_w), ph / float(src_h)

        def pt(xy: npt.NDArray[np.float64]) -> tuple[int, int]:
            x = int(xy[0] * sx)
            y = int(xy[1] * sy)
            if self._settings.mirror:
                x = pw - 1 - x
            return x, y

        if tracking.pnp_points is not None:
            points = [pt(p) for p in tracking.pnp_points]
            for px, py in points:
                cv2.circle(canvas, (px, py), 3, (0, 220, 255), -1, cv2.LINE_AA)
            if len(points) >= 6:
                nose, chin, le, re, lm, rm = points
                cv2.line(canvas, nose, chin, (80, 200, 255), 1, cv2.LINE_AA)
                cv2.line(canvas, le, re, (80, 200, 255), 1, cv2.LINE_AA)
                cv2.line(canvas, lm, rm, (80, 200, 255), 1, cv2.LINE_AA)

        for iris in (tracking.left_iris, tracking.right_iris):
            if iris is not None and np.any(iris):
                cv2.circle(canvas, pt(iris), 4, (0, 255, 120), -1, cv2.LINE_AA)

        label = f"{tracking.confidence:.0%}" if tracking.confidence > 0 else "ok"
        cv2.putText(
            canvas, label, (6, ph - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.4,
            (0, 0, 0), 2, cv2.LINE_AA,
        )
        cv2.putText(
            canvas, label, (6, ph - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.4,
            (120, 255, 180), 1, cv2.LINE_AA,
        )

    def _upload_texture(self, rgb: npt.NDArray[np.uint8]) -> None:
        h, w = rgb.shape[:2]
        GL.glBindTexture(GL.GL_TEXTURE_2D, self._texture)
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MIN_FILTER, GL.GL_LINEAR)
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MAG_FILTER, GL.GL_LINEAR)
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_WRAP_S, GL.GL_CLAMP_TO_EDGE)
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_WRAP_T, GL.GL_CLAMP_TO_EDGE)

        if w != self._tex_width or h != self._tex_height:
            GL.glTexImage2D(
                GL.GL_TEXTURE_2D, 0, GL.GL_RGB, w, h, 0,
                GL.GL_RGB, GL.GL_UNSIGNED_BYTE, rgb,
            )
            self._tex_width = w
            self._tex_height = h
        else:
            GL.glTexSubImage2D(
                GL.GL_TEXTURE_2D, 0, 0, 0, w, h,
                GL.GL_RGB, GL.GL_UNSIGNED_BYTE, rgb,
            )

    def _draw_quad(
        self, window_w: int, window_h: int, frame_aspect: float
    ) -> None:
        x0, y0, x1, y1 = _preview_rect_ndc(
            window_w, window_h, frame_aspect,
            self._settings.corner, self._settings.scale, self._settings.margin_px,
        )
        # V is intentionally inverted: glTexImage2D treats array row 0 (the
        # image TOP) as v=0, while y0 here is the BOTTOM edge of the quad.
        # Mapping v=1 onto y0 puts the image top back at the top.  This is the
        # same trick DebugOverlay uses; the alternative is np.flipud on every
        # upload, which costs an array copy per frame.
        vertices = np.array(
            [
                x0, y0, 0.0, 1.0,
                x1, y0, 1.0, 1.0,
                x1, y1, 1.0, 0.0,
                x0, y1, 0.0, 0.0,
            ],
            dtype=np.float32,
        )
        GL.glBindBuffer(GL.GL_ARRAY_BUFFER, self._vbo)
        GL.glBufferSubData(GL.GL_ARRAY_BUFFER, 0, vertices.nbytes, vertices)

        GL.glDisable(GL.GL_DEPTH_TEST)
        GL.glEnable(GL.GL_BLEND)
        GL.glBlendFunc(GL.GL_SRC_ALPHA, GL.GL_ONE_MINUS_SRC_ALPHA)

        self._shader.use()
        tex_loc = GL.glGetUniformLocation(self._shader.program_id, "uTexture")
        GL.glUniform1i(tex_loc, 0)
        GL.glActiveTexture(GL.GL_TEXTURE0)
        GL.glBindTexture(GL.GL_TEXTURE_2D, self._texture)
        GL.glBindVertexArray(self._vao)
        GL.glDrawElements(GL.GL_TRIANGLES, 6, GL.GL_UNSIGNED_INT, None)
        GL.glBindVertexArray(0)

        GL.glDisable(GL.GL_BLEND)
        GL.glEnable(GL.GL_DEPTH_TEST)


def _preview_rect_ndc(
    window_w: int,
    window_h: int,
    frame_aspect: float,
    corner: str,
    scale: float,
    margin_px: int,
) -> tuple[float, float, float, float]:
    """Return NDC bounds (x0, y0, x1, y1) for the preview quad."""
    pw = window_w * float(np.clip(scale, 0.08, 0.45))
    ph = pw / max(frame_aspect, 1e-6)
    m = float(margin_px)

    if corner == "bottom-left":
        left, bottom = m, m
    elif corner == "top-right":
        left, bottom = window_w - m - pw, window_h - m - ph
    elif corner == "top-left":
        left, bottom = m, window_h - m - ph
    else:  # bottom-right (default)
        left, bottom = window_w - m - pw, m

    x0 = 2.0 * left / window_w - 1.0
    x1 = 2.0 * (left + pw) / window_w - 1.0
    y0 = 2.0 * bottom / window_h - 1.0
    y1 = 2.0 * (bottom + ph) / window_h - 1.0
    return x0, y0, x1, y1
