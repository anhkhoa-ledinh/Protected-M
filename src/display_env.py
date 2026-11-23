"""Pick the window-system backend before GLFW and PyOpenGL are imported.

Native Wayland GLFW cannot read or set the window position, but the
head-coupled screen plane (D2) is built from it. On a Wayland desktop that has
XWayland, use GLFW's X11 backend instead: it reports the real position.
"""

from __future__ import annotations

import os
import sys
from collections.abc import MutableMapping


def prefer_xwayland(environ: MutableMapping[str, str] | None = None,
                    platform: str | None = None) -> bool:
    """Select pyGLFW's X11 library (and a matching GLX context for PyOpenGL).

    Only on Linux, only inside a Wayland session that also offers XWayland
    (``WAYLAND_DISPLAY`` and ``DISPLAY`` both set). Values the user already
    exported are kept. Returns True when it changed the environment.
    """
    env = os.environ if environ is None else environ
    plat = sys.platform if platform is None else platform
    if not plat.startswith("linux"):
        return False
    if not (env.get("WAYLAND_DISPLAY") and env.get("DISPLAY")):
        return False
    before = dict(env)
    env.setdefault("PYGLFW_LIBRARY_VARIANT", "x11")
    env.setdefault("PYOPENGL_PLATFORM", "glx")
    return dict(env) != before
