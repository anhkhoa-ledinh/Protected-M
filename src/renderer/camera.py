"""Physical-screen off-axis camera (Robert Kooima model)."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import numpy.typing as npt

from src.math.homogeneous import Mat4, to_column_major_flat
from src.math.projection import compute_off_axis_projection


@dataclass
class VirtualCamera:
    """
    Camera driven by tracked eye through a fixed physical monitor plane.

    The monitor is static in world space; only eye position changes.
    """

    screen_width_m: float = 0.531
    screen_height_m: float = 0.299
    screen_center_m: np.ndarray = field(
        default_factory=lambda: np.array([0.0, 0.0, 0.0], dtype=np.float64)
    )
    near_m: float = 0.01
    far_m: float = 8.0
    eye_position_mm: np.ndarray = field(
        default_factory=lambda: np.array([0.0, 0.0, 600.0], dtype=np.float64)
    )
    cache_projection: bool = True
    cache_epsilon_mm: float = 0.05

    max_lateral_mm: float = 2000.0
    min_depth_mm: float = 150.0
    max_depth_mm: float = 4000.0
    zoom_offset_mm: float = 0.0
    zoom_step_mm: float = 5.0

    def __post_init__(self) -> None:
        self.eye_position_mm = self._sanitize_eye(self.eye_position_mm)
        self._cached_eye = self.eye_position_mm.copy()
        self._cached_zoom_mm = float(self.zoom_offset_mm)
        self._view_flat: npt.NDArray[np.float32] | None = None
        self._proj_flat: npt.NDArray[np.float32] | None = None

    def set_eye_mm(self, position_mm: np.ndarray) -> None:
        """Update eye position in millimeters (rejecting degenerate poses)."""
        self.eye_position_mm = self._sanitize_eye(position_mm)

    def set_screen(self, width_m: float, height_m: float, center_m) -> None:
        """
        Đặt lại tấm kính thật (D2) và **bỏ** phép chiếu đang đệm.

        ``_needs_recompute`` chỉ nhìn mắt và zoom, nên dời cửa sổ mà không
        dọn bộ đệm thì phép chiếu cũ sống thêm tới lúc đầu người nhúc nhích
        quá ``cache_epsilon_mm`` — cảnh đứng sai một nhịp rồi mới giật về.
        """
        self.screen_width_m = float(width_m)
        self.screen_height_m = float(height_m)
        self.screen_center_m = np.asarray(center_m, dtype=np.float64).reshape(3).copy()
        self._view_flat = None
        self._proj_flat = None
        # Mắt được kẹp theo Z của tâm màn; tấm kính đổi thì kẹp lại cho chắc.
        self.eye_position_mm = self._sanitize_eye(self.eye_position_mm)

    def adjust_zoom(self, delta_mm: float | None = None) -> None:
        """Add manual depth offset to Kooima d (does not change tracked eye pose)."""
        step = self.zoom_step_mm if delta_mm is None else float(delta_mm)
        self.zoom_offset_mm += step

    def _sanitize_eye(self, position_mm: np.ndarray) -> np.ndarray:
        """Clamp eye to a physically plausible range and drop bad values.

        A transient garbage pose (non-finite or astronomically large eye
        translation from a bad solvePnP/filter spike) must never reach the
        off-axis projection, where it would collapse the frustum bounds.
        """
        candidate = np.asarray(position_mm, dtype=np.float64).reshape(3).copy()
        fallback = getattr(self, "eye_position_mm", None)
        if fallback is None:
            fallback = np.array([0.0, 0.0, 600.0], dtype=np.float64)
        if not np.all(np.isfinite(candidate)):
            candidate = np.where(np.isfinite(candidate), candidate, fallback)
        lateral = float(self.max_lateral_mm)
        candidate[0] = float(np.clip(candidate[0], -lateral, lateral))
        candidate[1] = float(np.clip(candidate[1], -lateral, lateral))
        center_z = float(self.screen_center_m[2]) * 1000.0
        candidate[2] = float(
            np.clip(candidate[2], center_z + self.min_depth_mm, self.max_depth_mm)
        )
        return candidate

    @property
    def eye_position_m(self) -> np.ndarray:
        """Eye position in meters for OpenGL world space."""
        return self.eye_position_mm * 0.001

    @property
    def pa(self) -> np.ndarray:
        half_w = self.screen_width_m * 0.5
        half_h = self.screen_height_m * 0.5
        cx, cy, cz = self.screen_center_m
        return np.array([cx - half_w, cy - half_h, cz], dtype=np.float64)

    @property
    def pb(self) -> np.ndarray:
        half_w = self.screen_width_m * 0.5
        half_h = self.screen_height_m * 0.5
        cx, cy, cz = self.screen_center_m
        return np.array([cx + half_w, cy - half_h, cz], dtype=np.float64)

    @property
    def pc(self) -> np.ndarray:
        half_w = self.screen_width_m * 0.5
        half_h = self.screen_height_m * 0.5
        cx, cy, cz = self.screen_center_m
        return np.array([cx - half_w, cy + half_h, cz], dtype=np.float64)

    @property
    def pd(self) -> np.ndarray:
        half_w = self.screen_width_m * 0.5
        half_h = self.screen_height_m * 0.5
        cx, cy, cz = self.screen_center_m
        return np.array([cx + half_w, cy + half_h, cz], dtype=np.float64)

    def projection_matrix(self) -> Mat4:
        projection, _ = compute_off_axis_projection(
            self.pa,
            self.pb,
            self.pc,
            self.eye_position_m,
            self.near_m,
            self.far_m,
            depth_offset_m=self.zoom_offset_mm * 0.001,
        )
        return projection

    def view_matrix(self) -> Mat4:
        _, view = compute_off_axis_projection(
            self.pa,
            self.pb,
            self.pc,
            self.eye_position_m,
            self.near_m,
            self.far_m,
            depth_offset_m=self.zoom_offset_mm * 0.001,
        )
        return view

    def view_projection_flat(self) -> tuple[npt.NDArray[np.float32], npt.NDArray[np.float32]]:
        """Return cached column-major view and projection for GPU upload."""
        if self._needs_recompute():
            self._view_flat = to_column_major_flat(self.view_matrix())
            self._proj_flat = to_column_major_flat(self.projection_matrix())
            self._cached_eye = self.eye_position_mm.copy()
            self._cached_zoom_mm = float(self.zoom_offset_mm)
        assert self._view_flat is not None and self._proj_flat is not None
        return self._view_flat, self._proj_flat

    def debug_frustum_lines(self) -> list[tuple[np.ndarray, np.ndarray]]:
        """Return world-space line segments for screen and frustum."""
        eye = self.eye_position_m
        vr = (self.pb - self.pa)
        vr = vr / np.linalg.norm(vr)
        vu = (self.pc - self.pa)
        vu = vu / np.linalg.norm(vu)
        vn = np.cross(vr, vu)
        vn = vn / np.linalg.norm(vn)

        va = self.pa - eye
        vb = self.pb - eye
        vc = self.pc - eye
        d_tracked = -float(np.dot(va, vn))
        d = max(d_tracked + self.zoom_offset_mm * 0.001, 1e-3)
        near = self.near_m
        far = self.far_m
        l = near * float(np.dot(vr, va)) / d
        r = near * float(np.dot(vr, vb)) / d
        b = near * float(np.dot(vu, va)) / d
        t = near * float(np.dot(vu, vc)) / d
        scale = far / near

        near_center = eye - vn * near
        far_center = eye - vn * far
        nll = near_center + vr * l + vu * b
        nlr = near_center + vr * r + vu * b
        nul = near_center + vr * l + vu * t
        nur = near_center + vr * r + vu * t
        fll = far_center + vr * (l * scale) + vu * (b * scale)
        flr = far_center + vr * (r * scale) + vu * (b * scale)
        ful = far_center + vr * (l * scale) + vu * (t * scale)
        fur = far_center + vr * (r * scale) + vu * (t * scale)

        lines: list[tuple[np.ndarray, np.ndarray]] = []
        screen = [self.pa, self.pb, self.pd, self.pc]
        for i in range(4):
            lines.append((screen[i], screen[(i + 1) % 4]))
        near_rect = [nll, nlr, nur, nul]
        far_rect = [fll, flr, fur, ful]
        for rect in (near_rect, far_rect):
            for i in range(4):
                lines.append((rect[i], rect[(i + 1) % 4]))
        for near_corner, far_corner in zip(near_rect, far_rect):
            lines.append((near_corner, far_corner))
            lines.append((eye, near_corner))
        return lines

    def _needs_recompute(self) -> bool:
        if not self.cache_projection:
            return True
        if self._view_flat is None or self._proj_flat is None:
            return True
        delta = np.linalg.norm(self.eye_position_mm - self._cached_eye)
        if abs(self.zoom_offset_mm - self._cached_zoom_mm) > 1e-6:
            return True
        return delta > self.cache_epsilon_mm
