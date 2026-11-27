"""Virtual room behind the photo: pure geometry and colours (no GL)."""

from __future__ import annotations

import math

import numpy as np
import pytest

from src.renderer import virtual_room as vr

E0 = 450.0
SW, SH = 344.0, 193.5     # screen size, mm
D = 600.0                 # farthest layer depth, mm


def _project_x(x, z, eye):
    """Project world point (x, z<=0) from ``eye`` onto the screen plane z=0."""
    ex, _, ez = eye
    return ex + (x - ex) * ez / (ez + abs(z))


# ---- layer size ------------------------------------------------------------

def test_farthest_layer_fills_the_back_wall_at_s1():
    w, h = vr.layer_size_mm(SW, SH, 1.0, E0, D, -D)
    assert (w, h) == pytest.approx((SW, SH))


def test_layer_at_screen_plane_is_screen_times_e0_over_e0_plus_d():
    w, h = vr.layer_size_mm(SW, SH, 1.0, E0, D, 0.0)
    f = E0 / (E0 + D)
    assert (w, h) == pytest.approx((SW * f, SH * f))


def test_s_scales_layer_size_linearly():
    for z in (0.0, -200.0, -D):
        w1, h1 = vr.layer_size_mm(SW, SH, 1.0, E0, D, z)
        w2, h2 = vr.layer_size_mm(SW, SH, 0.6, E0, D, z)
        assert (w2, h2) == pytest.approx((0.6 * w1, 0.6 * h1))


def test_s_max_is_e0_plus_d_over_e0_and_gives_todays_size_at_centre():
    assert vr.s_max(E0, D) == pytest.approx((E0 + D) / E0)
    w, _ = vr.layer_size_mm(SW, SH, vr.s_max(E0, D), E0, D, 0.0)
    assert w == pytest.approx(SW)


# ---- alignment -------------------------------------------------------------

def test_layers_line_up_into_the_photo_from_the_design_eye():
    eye = (0.0, 0.0, E0)
    rects = []
    for z in (0.0, -100.0, -350.0, -D):
        w, h = vr.layer_size_mm(SW, SH, 1.0, E0, D, z)
        x0 = _project_x(-w / 2, z, eye)
        x1 = _project_x(w / 2, z, eye)
        rects.append((x0, x1))
    for r in rects:
        assert r == pytest.approx(rects[0], abs=0.01)
    half = SW / 2 * E0 / (E0 + D)
    assert rects[0] == pytest.approx((-half, half), abs=0.01)


# ---- clip rectangle --------------------------------------------------------

def test_clip_rect_at_screen_plane_is_the_screen():
    r = vr.clip_rect_mm(0.0, (37.0, -20.0, E0), SW / 2, SH / 2)
    assert r == pytest.approx((-SW / 2, SW / 2, -SH / 2, SH / 2))


def test_clip_rect_is_smaller_when_deeper():
    eye = (0.0, 0.0, E0)
    prev = None
    for z in (0.0, -100.0, -300.0, -D):
        x0, x1, _, _ = vr.clip_rect_mm(z, eye, SW / 2, SH / 2)
        if prev is not None:
            assert (x1 - x0) < prev
        prev = x1 - x0


def test_clip_rect_worked_example():
    # hx = 172, eye x = +100, E = 450, z = -450:
    #   scale = 450 / 900 = 0.5 -> half width 86
    #   centre = 100 * (1 - 450 / 900) = 50 -> x in [-36, 136]
    # hy = 96.75, eye y = -40: half height 48.375, centre -40 * 0.5 = -20.
    x0, x1, y0, y1 = vr.clip_rect_mm(-450.0, (100.0, -40.0, 450.0), 172.0, 96.75)
    assert (x0, x1) == pytest.approx((-36.0, 136.0))
    assert (y0, y1) == pytest.approx((-20.0 - 48.375, -20.0 + 48.375))


def test_clip_rect_shifts_toward_the_eye_side():
    left = vr.clip_rect_mm(-300.0, (-120.0, 0.0, E0), SW / 2, SH / 2)
    right = vr.clip_rect_mm(-300.0, (120.0, 0.0, E0), SW / 2, SH / 2)
    assert (left[0] + left[1]) / 2 < 0 < (right[0] + right[1]) / 2
    z = 300.0
    assert (right[0] + right[1]) / 2 == pytest.approx(120.0 * (1 - E0 / (E0 + z)))


# ---- grid ------------------------------------------------------------------

def test_cell_is_screen_width_over_16():
    assert vr.grid_cell_mm(SW) == pytest.approx(SW / 16)


def test_division_counts():
    cell = SW / 16
    assert vr.grid_divisions(SW, SH, D) == (
        16, max(1, round(SH / cell)), max(1, round(D / cell)))
    # a very shallow room still gets one division
    assert vr.grid_divisions(SW, SH, 1.0)[2] == 1


def test_line_counts_per_wall():
    hx, hy = SW / 2, SH / 2
    nx, ny, nz = vr.grid_divisions(SW, SH, D)
    walls = vr.grid_lines(hx, hy, D)
    assert set(walls) == {"floor", "ceiling", "left", "right", "back"}
    assert len(walls["back"]) == (nx + 1) + (ny + 1)
    assert len(walls["floor"]) == (nx + 1) + (nz + 1)
    assert len(walls["ceiling"]) == (nx + 1) + (nz + 1)
    assert len(walls["left"]) == (ny + 1) + (nz + 1)
    assert len(walls["right"]) == (ny + 1) + (nz + 1)


def test_every_endpoint_lies_on_its_wall_plane():
    hx, hy = SW / 2, SH / 2
    plane = {
        "floor": (1, -hy), "ceiling": (1, hy),
        "left": (0, -hx), "right": (0, hx), "back": (2, -D),
    }
    for name, segs in vr.grid_lines(hx, hy, D).items():
        axis, value = plane[name]
        for a, b in segs:
            assert a[axis] == pytest.approx(value)
            assert b[axis] == pytest.approx(value)


def test_grid_spans_the_room_and_is_evenly_spaced():
    hx, hy = SW / 2, SH / 2
    segs = vr.grid_lines(hx, hy, D)["floor"]
    # lines running along z sit at evenly spaced x from -hx to +hx
    xs = sorted({round(a[0], 6) for a, b in segs if a[0] == b[0]})
    assert xs[0] == pytest.approx(-hx) and xs[-1] == pytest.approx(hx)
    gaps = np.diff(xs)
    assert gaps == pytest.approx(np.full_like(gaps, SW / 16))
    zs = [a[2] for a, b in segs if a[0] == b[0]] + [b[2] for a, b in segs if a[0] == b[0]]
    assert min(zs) == pytest.approx(-D) and max(zs) == pytest.approx(0.0)


# ---- colours ---------------------------------------------------------------

def _photo(top, bottom, h=100, w=20):
    img = np.zeros((h, w, 3), dtype=np.uint8)
    img[: h // 2] = top
    img[h // 2:] = bottom
    return img


BLUE = (40, 80, 200)
BROWN = (120, 70, 30)


def test_ceiling_and_floor_come_from_the_top_and_bottom_8_percent():
    img = _photo(BLUE, BROWN)
    img[10:90] = (255, 0, 0)         # the middle must not matter
    c = vr.room_colors(img)
    assert c.ceiling == pytest.approx(BLUE)
    assert c.floor == pytest.approx(BROWN)


def test_line_colour_max_channel_is_215_and_hue_is_kept():
    c = vr.room_colors(_photo(BLUE, BROWN))
    ceil = c.line("ceiling")
    floor = c.line("floor")
    assert max(ceil) == pytest.approx(215.0)
    assert max(floor) == pytest.approx(215.0)
    assert ceil == pytest.approx([v * 215 / 200 for v in BLUE])
    assert floor == pytest.approx([v * 215 / 120 for v in BROWN])
    assert ceil[2] > ceil[0] and floor[0] > floor[2]      # blue vs brown


def test_fill_colour_max_channel_is_26():
    c = vr.room_colors(_photo(BLUE, BROWN))
    assert max(c.fill("ceiling")) == pytest.approx(26.0)
    assert max(c.fill("floor")) == pytest.approx(26.0)


def test_side_wall_blends_floor_to_ceiling():
    c = vr.room_colors(_photo(BLUE, BROWN))
    assert c.line("side", 0.0) == pytest.approx(c.line("floor"))
    assert c.line("side", 1.0) == pytest.approx(c.line("ceiling"))
    mid = (np.array(c.line("floor")) + np.array(c.line("ceiling"))) / 2
    assert c.line("side", 0.5) == pytest.approx(mid)
    assert c.line("back") == pytest.approx(mid)
    midf = (np.array(c.fill("floor")) + np.array(c.fill("ceiling"))) / 2
    assert c.fill("side", 0.5) == pytest.approx(midf)


def test_black_photo_gives_grey():
    c = vr.room_colors(np.zeros((50, 10, 3), dtype=np.uint8))
    assert c.line("ceiling") == pytest.approx((215.0,) * 3)
    assert c.line("floor") == pytest.approx((215.0,) * 3)
    assert c.fill("ceiling") == pytest.approx((26.0,) * 3)
    assert c.fill("floor") == pytest.approx((26.0,) * 3)


def test_rgba_photo_is_accepted():
    rgba = np.dstack([_photo(BLUE, BROWN), np.full((100, 20), 255, np.uint8)])
    assert vr.room_colors(rgba).ceiling == pytest.approx(BLUE)


# ---- state and keys (demo only) --------------------------------------------

def test_demo_room_is_on_by_default_and_toggle_flips_it():
    st = vr.RoomState(demo=True)
    assert st.shown and st.s == 1.0
    assert st.toggle() is False and not st.shown
    assert st.toggle() is True and st.shown


def test_album_room_is_off_and_keys_change_nothing():
    st = vr.RoomState(demo=False)
    assert not st.shown
    st.toggle()
    assert not st.shown and st.s == 1.0 and st.on is False


def test_clamp_follows_a_smaller_s_max():
    st = vr.RoomState(demo=True)
    st.s = 2.0
    st.clamp(1.5)
    assert st.s == 1.5


# ---- placing a layer in the room -------------------------------------------

def _layout(eye=(0.0, 0.0, E0), s=1.0, centre=(0.0, 0.0)):
    return vr.RoomLayout(hx=SW / 2, hy=SH / 2, depth_mm=D, e0_mm=E0, s=s,
                         eye_mm=eye, centre_mm=centre, width_px=1000, height_px=500)


def test_layer_matrix_resizes_and_recentres_the_quad():
    lay = _layout()
    # a quad 0.5 m wide centred at (0.01, -0.02) m, at the back
    qh = 0.5 * SH / SW                           # same aspect as the screen
    m = lay.layer_matrix(depth_mm=D, quad_w_m=0.5, quad_h_m=qh, centre_x_m=0.01,
                         centre_y_m=-0.02, plane_z_m=0.0)
    r = (SW / 1000.0) / 0.5                       # back layer fills the wall
    corner = m @ np.array([0.01 + 0.25, -0.02 + 0.1, -0.6, 1.0])
    assert corner[0] == pytest.approx(0.25 * r)
    assert corner[1] == pytest.approx(0.1 * r)
    assert corner[2] == pytest.approx(-0.6)


def test_layer_matrix_keeps_the_plane_z_and_follows_the_screen_centre():
    lay = _layout(centre=(30.0, -10.0))
    m = lay.layer_matrix(depth_mm=0.0, quad_w_m=0.344, quad_h_m=0.1935,
                         centre_x_m=0.0, centre_y_m=0.0, plane_z_m=-0.1)
    p = m @ np.array([0.0, 0.0, 0.0, 1.0])
    assert p[:3] == pytest.approx([0.03, -0.01, -0.1])


def test_scissor_covers_the_screen_at_depth_zero_and_shrinks_with_depth():
    lay = _layout()
    assert lay.scissor_px(0.0) == (0, 0, 1000, 500)
    x, y, w, h = lay.scissor_px(D)
    f = E0 / (E0 + D)
    assert w == pytest.approx(1000 * f, abs=2)
    assert h == pytest.approx(500 * f, abs=2)
    assert x == pytest.approx(500 - w / 2, abs=2)


def test_scissor_moves_with_the_eye():
    right = _layout(eye=(120.0, 0.0, E0)).scissor_px(300.0)
    centre = _layout().scissor_px(300.0)
    assert right[0] > centre[0]
    # shift = 120 * (1 - E/(E+300)) mm, and 1000 px = SW mm
    assert right[0] - centre[0] == pytest.approx(
        120 * (1 - E0 / (E0 + 300)) * 1000 / SW, abs=2)


def test_layer_depth_in_the_layout_is_clamped_at_the_glass():
    lay = _layout()
    assert lay.layer_size_mm(-50.0) == lay.layer_size_mm(0.0)


def test_a_taller_photo_fits_inside_the_room_instead_of_spilling():
    lay = _layout()
    # 3:2 photo (taller than the 16:9 screen): height limits the size
    qw, qh = 0.6, 0.4
    m = lay.layer_matrix(depth_mm=D, quad_w_m=qw, quad_h_m=qh, centre_x_m=0.0,
                         centre_y_m=0.0, plane_z_m=0.0)
    top = m @ np.array([qw / 2, qh / 2, -0.6, 1.0])
    assert top[1] == pytest.approx(SH / 2000.0)          # touches the ceiling
    assert top[0] <= SW / 2000.0 + 1e-9                  # and not wider than the wall


# ---- album guard: the game renders as before -------------------------------

class _PlaneStub:
    room_layout = "left over"

    def layer_records(self):
        return [type("L", (), {"z_world_m": -0.6})()]


def _renderer(state):
    from src.renderer.renderer import SceneRenderer

    r = object.__new__(SceneRenderer)
    r.room_state = state
    r._video_plane = _PlaneStub()
    return r


def test_album_renderer_never_gets_a_room_layout():
    for state in (None, vr.RoomState(demo=False)):
        r = _renderer(state)
        assert r._prepare_room() is None
        assert r._video_plane.room_layout is None      # today's rendering


def test_demo_with_the_room_off_renders_as_today():
    st = vr.RoomState(demo=True)
    st.toggle()
    r = _renderer(st)
    assert r._prepare_room() is None and r._video_plane.room_layout is None
