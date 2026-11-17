"""
TASK_game_feel_1118 Phase 1: live redraw while the window is being
resized. On Windows, dragging a window edge runs a modal loop inside
``glfw.poll_events()`` and no frame is drawn until it returns — GLFW's
window-refresh callback (wired through ``window.set_refresh_handler``)
is the only way to draw DURING that modal loop. ``_draw_frame_only``
must never read input, tick the game/match/PvP, read the webcam, or call
``pump_events`` — see its docstring in ``src/engine/app.py``.
"""

from __future__ import annotations

from src.engine.app import HcpdApplication
from src.renderer.window import GlfwWindow
from src.config.schema import RendererSettings


class _FakeAlbum:
    def __init__(self, has_viewport: bool = True) -> None:
        self._has_viewport = has_viewport
        self.draw_calls: list = []
        self.hidden_scene_calls: list = []
        self.tick_calls = 0

    def has_viewport(self) -> bool:
        return self._has_viewport

    def draw(self, fb_size, scene_draw=None) -> None:
        self.draw_calls.append((fb_size, scene_draw))

    def draw_hidden_scene(self, scene_draw) -> None:
        self.hidden_scene_calls.append(scene_draw)

    def tick(self) -> None:          # pragma: no cover - must never run
        self.tick_calls += 1


class _FakeWindow:
    def __init__(self, width: int = 800, height: int = 600) -> None:
        self.width = width
        self.height = height
        self.swap_count = 0

    def swap_buffers(self) -> None:
        self.swap_count += 1


def make_app(**overrides) -> HcpdApplication:
    app = HcpdApplication.__new__(HcpdApplication)
    app._album = overrides.get("album", _FakeAlbum())
    app._window = overrides.get("window", _FakeWindow())
    app._last_scene = overrides.get("last_scene")
    app._drawing = False
    return app


def test_draw_frame_only_draws_exactly_one_frame_at_new_size():
    album = _FakeAlbum()
    window = _FakeWindow(width=1304, height=745)   # "the new size"
    app = make_app(album=album, window=window)
    calls = []
    scene = lambda: calls.append("scene")           # noqa: E731

    app._draw_frame_only(scene)

    assert len(album.draw_calls) == 1
    fb_size, scene_draw = album.draw_calls[0]
    assert fb_size == (1304, 745)
    assert scene_draw is scene
    assert window.swap_count == 1


def test_draw_frame_only_hides_scene_when_album_has_no_viewport():
    album = _FakeAlbum(has_viewport=False)
    app = make_app(album=album)
    scene = lambda: None                              # noqa: E731

    app._draw_frame_only(scene)

    assert album.hidden_scene_calls == [scene]
    assert len(album.draw_calls) == 1


def test_draw_frame_only_draws_scene_directly_with_no_album():
    app = make_app(album=None)
    calls = []
    scene = lambda: calls.append("scene")             # noqa: E731

    app._draw_frame_only(scene)

    assert calls == ["scene"]
    assert app._window.swap_count == 1


def test_refresh_does_not_tick_the_album():
    """A refresh must not advance game time, input, or the match —
    ``_draw_frame_only``/``_on_refresh`` never call ``album.tick()``."""
    album = _FakeAlbum()
    app = make_app(album=album, last_scene=lambda: None)

    app._on_refresh()

    assert album.tick_calls == 0
    assert len(album.draw_calls) == 1


def test_refresh_with_no_prior_frame_does_nothing():
    app = make_app(last_scene=None)
    app._on_refresh()
    assert app._album.draw_calls == []


def test_nested_refresh_is_ignored():
    album = _FakeAlbum()
    app = make_app(album=album)
    outer_scene = lambda: None                        # noqa: E731
    app._last_scene = outer_scene

    def scene_that_tries_to_refresh_again():
        # Simulates GLFW firing another refresh WHILE this frame is
        # still being drawn (nested modal loop re-entrance).
        app._on_refresh()

    album.draw = lambda fb_size, scene_draw=None: (
        album.draw_calls.append((fb_size, scene_draw)),
        scene_that_tries_to_refresh_again(),
    )

    app._on_refresh()

    # Only the outer call actually drew; the nested one saw _drawing=True
    # and did nothing.
    assert len(album.draw_calls) == 1


def test_refresh_during_main_loops_own_draw_is_ignored():
    """``_drawing`` is also set around the MAIN LOOP's own
    ``_draw_frame_only`` call, so a refresh firing from inside that draw
    (e.g. GLFW delivering it synchronously) is a no-op too."""
    album = _FakeAlbum()
    app = make_app(album=album)
    scene = lambda: None                              # noqa: E731

    app._drawing = True
    try:
        app._on_refresh()
    finally:
        app._drawing = False

    assert album.draw_calls == []


def test_set_refresh_handler_stores_without_a_real_window():
    """Constructing ``GlfwWindow`` and calling ``set_refresh_handler``
    before ``create()`` must not touch GLFW — mirrors
    ``test_window_viewport.py``'s no-``create()`` pattern."""
    window = GlfwWindow(RendererSettings(window_width=1280, window_height=720))
    calls = []
    window.set_refresh_handler(lambda: calls.append("refresh"))
    assert window._refresh_handler is not None
    assert calls == []  # not invoked just by registering
