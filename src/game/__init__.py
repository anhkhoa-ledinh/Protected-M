"""Plain-Python match/weapon rules (TASK_arch_refactor_1120 R5) — match
start/end, weapons, firing, hit resolution, damage, scoring. May import
``numpy`` and ``src.config``; nothing from ``src.engine``, ``src.renderer``,
``src.ui_album``, ``src.fx``, ``src.net``, OpenGL, glfw or PIL. Callers
(``src.engine.app``, ``src.devtools``) pass in what this layer needs (time,
aim point, hit test result) rather than this layer reaching up for it."""
