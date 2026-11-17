"""
TASK_game_feel_1118 Phase 6: ``src/renderer/camera.py``'s scope fields —
``effective_eye_position_mm``, ``_effective_corners``, and the recompute
cache key now including scope state. Pure numpy, no GL.
"""

from __future__ import annotations

import numpy as np

from src.renderer.camera import VirtualCamera


def _camera() -> VirtualCamera:
    camera = VirtualCamera(screen_width_m=0.344, screen_height_m=0.194,
                           screen_center_m=np.array([0.0, 0.0, 0.0]))
    camera.set_eye_mm(np.array([0.0, 0.0, 600.0]))
    return camera


def test_effective_eye_equals_raw_eye_when_not_scoped():
    camera = _camera()
    assert np.array_equal(camera.effective_eye_position_mm, camera.eye_position_mm)


def test_effective_eye_equals_raw_eye_when_scoped_but_no_anchor_set():
    camera = _camera()
    camera.scope_active = True
    camera.scope_zoom = 2.5
    assert np.array_equal(camera.effective_eye_position_mm, camera.eye_position_mm)


def test_effective_eye_steadies_toward_the_anchor_while_scoped():
    camera = _camera()
    anchor = camera.eye_position_mm.copy()
    camera.scope_eye_anchor_mm = anchor
    camera.scope_active = True
    camera.scope_zoom = 2.5
    camera.set_eye_mm(anchor + np.array([4.0, -2.0, 0.0]))
    steady = camera.effective_eye_position_mm
    np.testing.assert_allclose(steady, anchor + np.array([4.0, -2.0, 0.0]) / 2.5)


def test_effective_corners_equal_plain_corners_when_not_scoped():
    camera = _camera()
    pa, pb, pc = camera._effective_corners()
    assert np.array_equal(pa, camera.pa)
    assert np.array_equal(pb, camera.pb)
    assert np.array_equal(pc, camera.pc)


def test_effective_corners_warp_toward_the_crosshair_when_scoped():
    camera = _camera()
    camera.scope_active = True
    camera.scope_zoom = 2.5
    camera.scope_crosshair_m = np.array([0.0, 0.0, 0.0])    # glass centre
    pa, pb, pc = camera._effective_corners()
    # Warped corners are closer to the (centred) crosshair than the real
    # ones — a smaller sub-window, magnified.
    assert np.linalg.norm(pa) < np.linalg.norm(camera.pa)
    assert np.linalg.norm(pb) < np.linalg.norm(camera.pb)
    assert np.linalg.norm(pc) < np.linalg.norm(camera.pc)


# ---- recompute cache key --------------------------------------------------
def test_entering_scope_invalidates_the_cache():
    camera = _camera()
    v1, p1 = camera.view_projection_flat()
    camera.scope_active = True
    camera.scope_zoom = 2.5
    camera.scope_crosshair_m = np.array([0.01, 0.0, 0.0])
    v2, p2 = camera.view_projection_flat()
    assert not np.array_equal(p1, p2)


def test_changing_scope_zoom_invalidates_the_cache():
    camera = _camera()
    camera.scope_active = True
    camera.scope_zoom = 1.5
    camera.scope_crosshair_m = np.array([0.01, 0.0, 0.0])
    _v1, p1 = camera.view_projection_flat()
    camera.scope_zoom = 2.5
    _v2, p2 = camera.view_projection_flat()
    assert not np.array_equal(p1, p2)


def test_changing_scope_crosshair_invalidates_the_cache():
    camera = _camera()
    camera.scope_active = True
    camera.scope_zoom = 2.5
    camera.scope_crosshair_m = np.array([0.01, 0.0, 0.0])
    _v1, p1 = camera.view_projection_flat()
    camera.scope_crosshair_m = np.array([-0.01, 0.0, 0.0])
    _v2, p2 = camera.view_projection_flat()
    assert not np.array_equal(p1, p2)


def test_cache_still_holds_when_nothing_scope_related_changes():
    camera = _camera()
    camera.scope_active = True
    camera.scope_zoom = 2.5
    camera.scope_crosshair_m = np.array([0.01, 0.0, 0.0])
    v1, p1 = camera.view_projection_flat()
    v2, p2 = camera.view_projection_flat()
    assert v1 is v2 and p1 is p2
