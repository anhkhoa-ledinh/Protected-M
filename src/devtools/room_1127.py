"""``HCPD_ROOM_1127``: scripted run of the virtual room in ``--demo``.

``HCPD_ROOM_1127="step;step;..."`` with ``HCPD_ROOM_OUT=<dir>`` for pictures.
Steps (each waits for 8 drawn frames and 1 s after the previous one):

``photo:<path>``   load a photo through the normal path, wait for its layers
``pose:x,y[,z]``   pin the eye (mm, relative to the screen centre; z defaults to
                   the design distance); ``pose:free`` releases it
``room:on|off``    the B key
``relief:on|off``  set the per-layer relief (the demo F10 key) to that state
``s:<v>|max|1``    photo scale (like + - 0, clamped)
``shot:<stem>``    framebuffer -> ``<out>/<stem>.png`` (what F12 saves) and a
                   line with its size and pixel standard deviation
``bench:<n>:<tag>``  n frames with vsync off; logs the median frame time
``quit``
"""

from __future__ import annotations

import logging
import os
import statistics
import time
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

_MIN_FRAMES = 8
_MIN_SECONDS = 1.0
_LAYER_WAIT_S = 180.0


class RoomScript:
    def __init__(self, hooks, spec: str, out_dir: str) -> None:
        self._app = hooks
        self._steps = [s.strip() for s in spec.split(";") if s.strip()]
        self._out = Path(out_dir) if out_dir else None
        self._i = 0
        self._frames = 0
        self._last = time.perf_counter()
        self._pinned: np.ndarray | None = None
        self._bench: dict | None = None
        self._started = None

    # -- called by app.py every frame ---------------------------------------

    def pinned_eye(self):
        return self._pinned

    def tick(self, now: float) -> None:
        app = self._app
        if app.window.is_iconified():
            return
        if self._bench is not None:
            self._tick_bench()
            return
        self._frames += 1
        if self._i >= len(self._steps):
            return
        if self._frames < _MIN_FRAMES or time.perf_counter() - self._last < _MIN_SECONDS:
            return
        step = self._steps[self._i]
        if not self._run(step, now):
            return                                   # step wants more frames
        self._i += 1
        self._frames = 0
        self._last = time.perf_counter()

    # -- steps --------------------------------------------------------------

    def _run(self, step: str, now: float) -> bool:
        app = self._app
        name, _, arg = step.partition(":")
        if name == "photo":
            if self._started is None:
                self._started = time.perf_counter()
                app.try_load_media(Path(arg), now)
                return False
            if app.renderer.layer_records():
                self._started = None
                logger.info("Room script: photo ready (%d layers)",
                            len(app.renderer.layer_records()))
                return True
            if time.perf_counter() - self._started > _LAYER_WAIT_S:
                logger.error("Room script: no layers after %.0f s", _LAYER_WAIT_S)
                app.window.request_close()
                return True
            return False
        if name == "pose":
            if arg == "free":
                self._pinned = None
            else:
                v = [float(x) for x in arg.split(",")]
                z = v[2] if len(v) > 2 else float(
                    app.content.overscan_design_distance_mm)
                self._pinned = np.array([v[0], v[1], z], dtype=np.float64)
            logger.info("Room script: eye pinned at %s", self._pinned)
        elif name == "room":
            app.room.on = arg == "on"
            logger.info("Room script: room %s", arg)
        elif name == "relief":
            want = arg == "on"
            if app.renderer.relief_enabled != want:
                app.renderer.toggle_relief()
            logger.info("Room script: relief %s", arg)
        elif name == "s":
            smax = app.renderer.room_s_max() or 1.0
            app.room.s = smax if arg == "max" else float(arg)
            app.room.clamp(smax)
            logger.info("Room script: s=%.3f (s_max %.3f)", app.room.s, smax)
        elif name == "shot":
            self._shot(arg)
        elif name == "bench":
            n, _, tag = arg.partition(":")
            import glfw

            glfw.swap_interval(0)
            self._bench = {"n": int(n), "tag": tag, "times": [],
                           "t": time.perf_counter(), "skip": 30}
            return True
        elif name == "quit":
            app.window.request_close()
        else:
            logger.error("Room script: unknown step %r", step)
        return True

    def _shot(self, stem: str) -> None:
        from OpenGL import GL

        from src.ui_album.shots import save_framebuffer

        app = self._app
        width, height = app.window.framebuffer_size()
        GL.glFinish()
        raw = GL.glReadPixels(0, 0, int(width), int(height), GL.GL_RGB,
                              GL.GL_UNSIGNED_BYTE)
        std = float(np.frombuffer(raw, dtype=np.uint8).std())
        path = save_framebuffer(self._out / f"{stem}.png", width, height)
        logger.info("Room shot %s: %dx%d std=%.2f room=%s s=%.3f eye=%s", path.name,
                    width, height, std, app.room.shown, app.room.s,
                    None if self._pinned is None else self._pinned.tolist())

    def _tick_bench(self) -> None:
        b = self._bench
        t = time.perf_counter()
        if b["skip"] > 0:                            # settle after the switch
            b["skip"] -= 1
        else:
            b["times"].append((t - b["t"]) * 1000.0)
        b["t"] = t
        if len(b["times"]) >= b["n"]:
            import glfw

            glfw.swap_interval(1)
            ts = b["times"]
            logger.info("Room bench %s: frames=%d median=%.3f ms mean=%.3f "
                        "p95=%.3f ms room=%s", b["tag"], len(ts),
                        statistics.median(ts), statistics.fmean(ts),
                        float(np.percentile(ts, 95)), self._app.room.shown)
            self._bench = None
            self._frames = 0
            self._last = time.perf_counter()


def from_env(hooks) -> RoomScript | None:
    spec = os.environ.get("HCPD_ROOM_1127", "")
    return RoomScript(hooks, spec, os.environ.get("HCPD_ROOM_OUT", "")) if spec else None
