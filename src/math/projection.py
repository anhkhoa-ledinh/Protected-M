"""Generalized off-axis projection (Robert Kooima)."""

from __future__ import annotations

import logging

import numpy as np

from src.math.homogeneous import Mat4

_log = logging.getLogger(__name__)


def off_axis_projection(
    left: float,
    right: float,
    bottom: float,
    top: float,
    near: float,
    far: float,
) -> Mat4:
    """
    Build a generalized off-axis perspective projection matrix.

    Frustum planes are defined at the near plane in camera space.
    This is the core matrix for head-coupled perspective rendering.
    """
    if near <= 0.0 or far <= near:
        raise ValueError("Invalid near/far clip distances")
    if right <= left or top <= bottom:
        raise ValueError("Invalid frustum bounds")

    matrix = np.zeros((4, 4), dtype=np.float64)
    matrix[0, 0] = (2.0 * near) / (right - left)
    matrix[1, 1] = (2.0 * near) / (top - bottom)
    matrix[0, 2] = (right + left) / (right - left)
    matrix[1, 2] = (top + bottom) / (top - bottom)
    matrix[2, 2] = -(far + near) / (far - near)
    matrix[2, 3] = -(2.0 * far * near) / (far - near)
    matrix[3, 2] = -1.0
    return matrix


def compute_off_axis_projection(
    pa: np.ndarray,
    pb: np.ndarray,
    pc: np.ndarray,
    pe: np.ndarray,
    near: float,
    far: float,
    *,
    depth_offset_m: float = 0.0,
) -> tuple[Mat4, Mat4]:
    """
    Compute Kooima generalized perspective from a physical screen plane.

    Inputs are world-space points in meters:
    - pa: lower-left screen corner
    - pb: lower-right screen corner
    - pc: upper-left screen corner
    - pe: tracked eye position
    """
    pe = np.asarray(pe, dtype=np.float64).reshape(3)
    if not np.all(np.isfinite(pe)):
        raise ValueError("Eye position must be finite")

    vr = _normalize(pb - pa)
    vu = _normalize(pc - pa)
    vn = _normalize(np.cross(vr, vu))

    va = pa - pe
    vb = pb - pe
    vc = pc - pe

    d_tracked = -float(np.dot(va, vn))
    d = max(d_tracked + float(depth_offset_m), 1e-3)

    left = near * float(np.dot(vr, va)) / d
    right = near * float(np.dot(vr, vb)) / d
    bottom = near * float(np.dot(vu, va)) / d
    top = near * float(np.dot(vu, vc)) / d

    if _log.isEnabledFor(logging.DEBUG):
        _log.debug(
            "Kooima d_tracked=%.1fmm d_offset=%.1fmm d_eff=%.1fmm  "
            "eye=[%.1f,%.1f,%.1f]mm  "
            "l=%.4f r=%.4f b=%.4f t=%.4f (m)",
            d_tracked * 1000,
            depth_offset_m * 1000,
            d * 1000,
            pe[0] * 1000, pe[1] * 1000, pe[2] * 1000,
            left, right, bottom, top,
        )

    projection = off_axis_projection(left, right, bottom, top, near, far)
    view = _screen_view_matrix(vr, vu, vn, pe)
    return projection, view


def ComputeOffAxisProjection(
    pa: np.ndarray,
    pb: np.ndarray,
    pc: np.ndarray,
    pe: np.ndarray,
    near: float,
    far: float,
    *,
    depth_offset_m: float = 0.0,
) -> tuple[Mat4, Mat4]:
    """C++-style alias kept for parity with requested naming."""
    return compute_off_axis_projection(
        pa, pb, pc, pe, near, far, depth_offset_m=depth_offset_m
    )


def _screen_view_matrix(
    vr: np.ndarray,
    vu: np.ndarray,
    vn: np.ndarray,
    eye: np.ndarray,
) -> Mat4:
    rotation = np.eye(4, dtype=np.float64)
    rotation[0, :3] = vr
    rotation[1, :3] = vu
    rotation[2, :3] = vn

    translation = np.eye(4, dtype=np.float64)
    translation[0, 3] = -eye[0]
    translation[1, 3] = -eye[1]
    translation[2, 3] = -eye[2]
    return rotation @ translation


def _normalize(vector: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if norm < 1e-12:
        raise ValueError("Cannot normalize zero-length vector")
    return vector / norm
