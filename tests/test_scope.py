"""
TASK_game_feel_1118 Phase 6: the Precision-gun scope's pure math — the
Kooima sub-window projection, the steadied eye (hit-test equality), and
the finer-aim sensitivity mapping. No GL: ``VirtualCamera``/
``compute_off_axis_projection`` are plain numpy, so these run anywhere.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.renderer.camera import VirtualCamera
from src.game.hit import SpriteMask, pick_target, sprite_quad
from src.ui_album.scope import (DESIGN_SIZE, ScopeState, design_to_glass_m,
                                steadied_eye)

GLASS_MM = (344.0, 194.0)
CENTRE_MM = (0.0, 0.0)


def _ndc_to_design(ndc_x: float, ndc_y: float) -> tuple[float, float]:
    return ((ndc_x * 0.5 + 0.5) * DESIGN_SIZE[0],
            (0.5 - ndc_y * 0.5) * DESIGN_SIZE[1])


def _project_design(camera: VirtualCamera, world_point_m) -> tuple[float, float]:
    """A glass-plane world point (metres) -> design px, through the
    camera's CURRENT (possibly scoped) view/projection matrices."""
    mvp = camera.projection_matrix() @ camera.view_matrix()
    p = np.array([world_point_m[0], world_point_m[1], world_point_m[2], 1.0])
    clip = mvp @ p
    ndc = clip[:3] / clip[3]
    return _ndc_to_design(float(ndc[0]), float(ndc[1]))


def _camera() -> VirtualCamera:
    camera = VirtualCamera(screen_width_m=GLASS_MM[0] / 1000.0,
                           screen_height_m=GLASS_MM[1] / 1000.0,
                           screen_center_m=np.array([0.0, 0.0, 0.0]))
    camera.set_eye_mm(np.array([0.0, 0.0, 600.0]))
    return camera


def _scope_in(camera: VirtualCamera, crosshair_design, z: float) -> None:
    sx, sy = design_to_glass_m(crosshair_design, GLASS_MM, CENTRE_MM)
    camera.scope_active = True
    camera.scope_zoom = z
    camera.scope_crosshair_m = np.array([sx, sy, 0.0])
    camera.scope_eye_anchor_mm = camera.eye_position_mm.copy()


# ---- G7: projection within 0.5 design px --------------------------------
@pytest.mark.parametrize("crosshair_design,g_design", [
    ((640.0, 360.0), (640.0, 360.0)),    # centre, same point as crosshair
    ((640.0, 360.0), (980.0, 120.0)),    # centre, aim near a corner
    ((200.0, 600.0), (260.0, 580.0)),    # off-centre, aim near an edge
])
def test_scoped_projection_matches_c_plus_z_times_g_minus_c(crosshair_design, g_design):
    camera = _camera()
    z = 2.5
    g_mm = design_to_glass_m(g_design, GLASS_MM, CENTRE_MM)
    g_world = (g_mm[0], g_mm[1], 0.0)

    _scope_in(camera, crosshair_design, z)
    actual = _project_design(camera, g_world)

    expected = (crosshair_design[0] + z * (g_design[0] - crosshair_design[0]),
               crosshair_design[1] + z * (g_design[1] - crosshair_design[1]))
    assert actual[0] == pytest.approx(expected[0], abs=0.5)
    assert actual[1] == pytest.approx(expected[1], abs=0.5)


def test_crosshair_itself_is_a_fixed_point_of_the_scope():
    """The scene point UNDER the crosshair is unchanged by scoping — the
    whole reason shot marks/the hit fx (anchored exactly at the
    crosshair) need no transform (see the report's overlay list)."""
    camera = _camera()
    crosshair_design = (900.0, 250.0)
    unscoped = _project_design(camera, (
        *design_to_glass_m(crosshair_design, GLASS_MM, CENTRE_MM), 0.0))
    _scope_in(camera, crosshair_design, 2.5)
    scoped = _project_design(camera, (
        *design_to_glass_m(crosshair_design, GLASS_MM, CENTRE_MM), 0.0))
    assert scoped[0] == pytest.approx(unscoped[0], abs=0.5)
    assert scoped[1] == pytest.approx(unscoped[1], abs=0.5)
    assert scoped[0] == pytest.approx(crosshair_design[0], abs=0.5)
    assert scoped[1] == pytest.approx(crosshair_design[1], abs=0.5)


def test_sub_window_corners_stay_inside_the_real_glass():
    """``c`` inside the design area => the warped pa'/pb'/pc' stay inside
    the glass rectangle (convex combination of c and the real corner) —
    "the view never shows past the photo edge"."""
    camera = _camera()
    _scope_in(camera, (900.0, 600.0), 2.5)
    pa, pb, pc = camera._effective_corners()
    half_w, half_h = GLASS_MM[0] / 2000.0, GLASS_MM[1] / 2000.0
    for corner in (pa, pb, pc):
        assert -half_w - 1e-9 <= corner[0] <= half_w + 1e-9
        assert -half_h - 1e-9 <= corner[1] <= half_h + 1e-9


# ---- G7: hit-test equality with/without the scope -------------------------
def _mask() -> SpriteMask:
    rgba = np.full((16, 16, 4), 255, dtype=np.uint8)
    return SpriteMask.from_rgba(rgba)


def test_hit_test_same_result_with_and_without_scope_same_crosshair_and_eye():
    """"The hit-test gives the same result with and without the scope for
    the same crosshair and eye" — with no head jitter (eye == anchor),
    ``steadied_eye`` is the identity, so the hit test (which never reads
    scope/zoom at all, only eye_mm/glass_mm/centre_mm) behaves
    identically either way."""
    eye_mm = (0.0, 0.0, 600.0)
    mask = _mask()
    quad = sprite_quad((0.0, 0.0, -180.0), 12.0, 1.0, eye_mm, GLASS_MM, CENTRE_MM)
    point = (quad.centre_x, quad.centre_y)
    bugs = {0: ((0.0, 0.0, -180.0), 12.0, 1.0, mask)}

    unscoped = pick_target(bugs, point, eye_mm, (), GLASS_MM, CENTRE_MM)

    # Scoped: no head jitter, so the steadied eye equals the raw eye —
    # the hit test must land on the exact same bug.
    steady = steadied_eye(eye_mm, eye_mm, 2.5)
    scoped = pick_target(bugs, point, steady, (), GLASS_MM, CENTRE_MM)

    assert unscoped == scoped == (0, False)


def test_steadied_eye_is_the_identity_with_no_head_movement():
    anchor = (0.0, 0.0, 600.0)
    assert steadied_eye(anchor, anchor, 2.5) == anchor


def test_steadied_eye_shrinks_jitter_by_z():
    anchor = (0.0, 0.0, 600.0)
    jittered = (4.0, -2.0, 600.0)
    steady = steadied_eye(jittered, anchor, 2.5)
    assert steady[0] == pytest.approx(anchor[0] + 4.0 / 2.5)
    assert steady[1] == pytest.approx(anchor[1] - 2.0 / 2.5)


# ---- sensitivity mapping (finer aim) --------------------------------------
def test_sensitivity_mapping_divides_mouse_delta_by_zoom():
    scope = ScopeState()
    scope.enter((640.0, 360.0))
    scope.zoom.x = 2.5          # pretend the spring already settled
    new = scope.update_crosshair((640.0, 360.0), (650.0, 370.0))
    assert new == pytest.approx((640.0 + 10.0 / 2.5, 360.0 + 10.0 / 2.5))


def test_sensitivity_mapping_is_a_per_frame_delta_not_a_jump_on_zoom_change():
    """Moving the mouse, THEN letting z change with no further mouse
    movement, must not move the crosshair again."""
    scope = ScopeState()
    scope.enter((640.0, 360.0))
    scope.zoom.x = 1.5
    once = scope.update_crosshair((640.0, 360.0), (660.0, 360.0))
    scope.zoom.x = 2.5          # zoom keeps rising, mouse hasn't moved
    twice = scope.update_crosshair(once, (660.0, 360.0))
    assert twice == pytest.approx(once)


def test_crosshair_clamped_to_the_design_area():
    scope = ScopeState()
    scope.enter((10.0, 10.0))
    scope.zoom.x = 1.0
    new = scope.update_crosshair((10.0, 10.0), (-500.0, -500.0))
    assert new == (0.0, 0.0)
    new2 = scope.update_crosshair(new, (5000.0, 5000.0))
    assert new2 == (DESIGN_SIZE[0], DESIGN_SIZE[1])


def test_zoom_spring_rises_toward_2_5_and_falls_back_on_exit():
    scope = ScopeState()
    scope.enter((640.0, 360.0))
    for _ in range(200):
        scope.step(0.01)
    assert scope.z == pytest.approx(2.5, abs=0.05)
    scope.exit()
    for _ in range(200):
        scope.step(0.01)
    assert scope.z == pytest.approx(1.0, abs=0.05)
