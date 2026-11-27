"""The room flares when s > 1 so zooming never cuts the photo (pure logic)."""

from __future__ import annotations

import inspect

import pytest

from src.renderer import virtual_room as vr

E0, D = 450.0, 600.0
HX, HY = 344.0 / 2, 194.0 / 2
S_MAX = vr.s_max(E0, D)
SCALES = (0.6, 0.8, 1.0, 1.3, S_MAX)
TOL = 1e-6


def _eye_project(p, eye):
    """Project a world point (z <= 0) from ``eye`` onto the screen plane."""
    ex, ey, ez = eye
    k = ez / (ez + abs(p[2]))
    return ex + (p[0] - ex) * k, ey + (p[1] - ey) * k


@pytest.mark.parametrize("s", SCALES)
@pytest.mark.parametrize("z", (0.0, D / 2, D))
def test_layers_stay_inside_the_room_cross_section(s, z):
    lw, lh = vr.layer_size_mm(2 * HX, 2 * HY, s, E0, D, -z)
    rx, ry = vr.cross_section_half(HX, HY, D, s, -z)
    assert lw / 2 <= rx + TOL
    assert lh / 2 <= ry + TOL


def test_flare_is_one_up_to_s1_and_s_above():
    assert vr.flare(0.6) == 1.0 and vr.flare(1.0) == 1.0
    assert vr.flare(1.7) == 1.7


@pytest.mark.parametrize("s", (0.6, 0.8, 1.0))
def test_room_is_the_straight_box_when_s_is_at_most_1(s):
    c = vr.room_corners(HX, HY, D, s)
    assert c["front"] == [(-HX, -HY, 0.0), (HX, -HY, 0.0), (HX, HY, 0.0), (-HX, HY, 0.0)]
    assert c["back"] == [(-HX, -HY, -D), (HX, -HY, -D), (HX, HY, -D), (-HX, HY, -D)]


def test_room_flares_the_back_wall_by_s():
    c = vr.room_corners(HX, HY, D, 1.5)
    assert c["back"][2] == pytest.approx((HX * 1.5, HY * 1.5, -D))
    assert c["front"][2] == pytest.approx((HX, HY, 0.0))
    assert vr.cross_section_half(HX, HY, D, 1.5, -D / 2) == pytest.approx(
        (HX * 1.25, HY * 1.25))


def test_walls_are_edge_on_at_s_max_from_the_design_eye():
    eye = (0.0, 0.0, E0)
    walls = vr.grid_lines(HX, HY, D, S_MAX)
    for name in ("left", "right", "floor", "ceiling"):
        for a, b in walls[name]:
            for p in (a, b):
                x, y = _eye_project(p, eye)
                if name in ("left", "right"):
                    assert abs(x) == pytest.approx(HX, abs=0.01)
                else:
                    assert abs(y) == pytest.approx(HY, abs=0.01)


@pytest.mark.parametrize("s", (0.6, 1.0, 1.5, S_MAX))
def test_grid_line_counts_do_not_depend_on_s(s):
    nx, ny, nz = vr.grid_divisions(2 * HX, 2 * HY, D)
    w = vr.grid_lines(HX, HY, D, s)
    assert len(w["back"]) == (nx + 1) + (ny + 1)
    assert len(w["floor"]) == len(w["ceiling"]) == (nx + 1) + (nz + 1)
    assert len(w["left"]) == len(w["right"]) == (ny + 1) + (nz + 1)


def test_default_grid_is_the_old_straight_grid():
    assert vr.grid_lines(HX, HY, D) == vr.grid_lines(HX, HY, D, 1.0)


@pytest.mark.parametrize("s", (0.6, 1.0, 1.5, S_MAX))
def test_depth_lines_join_a_front_point_to_its_scaled_back_point(s):
    f = vr.flare(s)
    walls = vr.grid_lines(HX, HY, D, s)
    for name in ("floor", "ceiling", "left", "right"):
        depth_lines = [(a, b) for a, b in walls[name] if a[2] == 0.0 and b[2] == -D]
        assert depth_lines, name
        for a, b in depth_lines:
            assert (b[0], b[1]) == pytest.approx((a[0] * f, a[1] * f))


@pytest.mark.parametrize("s", (0.6, 1.0, 1.5, S_MAX))
def test_cross_lines_lie_on_their_wall_at_their_depth(s):
    walls = vr.grid_lines(HX, HY, D, s)
    for name, segs in walls.items():
        for a, b in segs:
            if a[2] != b[2]:
                continue
            hx, hy = vr.cross_section_half(HX, HY, D, s, a[2])
            if name == "floor":
                assert a[1] == b[1] == pytest.approx(-hy)
                assert sorted((a[0], b[0])) == pytest.approx([-hx, hx])
            elif name == "ceiling":
                assert a[1] == b[1] == pytest.approx(hy)
                assert sorted((a[0], b[0])) == pytest.approx([-hx, hx])
            elif name in ("left", "right"):
                sign = -1.0 if name == "left" else 1.0
                assert a[0] == b[0] == pytest.approx(sign * hx)
                assert sorted((a[1], b[1])) == pytest.approx([-hy, hy])
            else:                       # back wall spans the back rectangle
                assert a[2] == pytest.approx(-D)


def test_back_wall_grid_spans_the_back_rectangle():
    f = 1.5
    pts = [p for a, b in vr.grid_lines(HX, HY, D, f)["back"] for p in (a, b)]
    assert min(p[0] for p in pts) == pytest.approx(-HX * f)
    assert max(p[0] for p in pts) == pytest.approx(HX * f)
    assert max(p[1] for p in pts) == pytest.approx(HY * f)


def test_layer_pass_gets_no_clip_rectangle():
    from src.renderer.content_plane import ContentPlaneRenderer as ContentPlane

    assert not hasattr(vr.RoomLayout, "scissor_px")
    src = inspect.getsource(ContentPlane._draw_layers)
    assert "glScissor" not in src and "GL_SCISSOR_TEST" not in src


def test_room_buffers_depend_on_s():
    from src.renderer.virtual_room_gl import VirtualRoomRenderer

    colors = vr.room_colors(__import__("numpy").zeros((4, 4, 3), "uint8"))
    a = VirtualRoomRenderer.build(colors, HX, HY, D, (0.0, 0.0), 1.0)
    b = VirtualRoomRenderer.build(colors, HX, HY, D, (0.0, 0.0), 1.5)
    assert a[0].shape == b[0].shape and a[1].shape == b[1].shape
    assert not (a[1] == b[1]).all()
    assert abs(float(b[1].reshape(-1, 6)[:, 0].max()) - HX * 1.5 / 1000.0) < 1e-6
