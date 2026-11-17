"""Sakuga VFX playback for Protected M (TASK_ui_sakuga_1117).

Pure Python, no GL (``spec.py``, ``anim.py``, ``player.py``, ``screens.py``)
— the GL texture layer lives in ``src/fx/library.py`` and the drawing hook
in ``src/ui_album/v3.py``/``app.py``, kept separate so this package stays
importable and testable on CI with no OpenGL context.
"""

from __future__ import annotations
