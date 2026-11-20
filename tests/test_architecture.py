"""TASK_arch_refactor_1120 R2: the dependency rules the refactor is working
toward — layers depend downward only (``app`` -> ``ui``/``game``/``net`` ->
``render3d``/``tracking``/``content``). Parses imports with ``ast`` (no
execution).

Scope: only ``src/`` and ``main.py`` — the PRODUCTION layering this task is
about. Tests and ``scripts/`` legitimately construct ``HcpdApplication``
directly for integration testing/tooling; that is not a layering violation
of the kind this file checks for.

Each later step (R3-R6) tightens this test as more of the target
architecture lands (``game/`` isolation in R5, ``net`` not depending on
``ui_album`` in R6, …).
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts import import_graph  # noqa: E402

#: Package-level cycles R2 fixed (TASK_arch_refactor_1120) — must never
#: reappear. ``net -> ui_album`` (and anything needing ``game/``) is
#: deliberately NOT checked here yet; R5/R6 own that.
_FIXED_PAIRS = [
    ("src.engine", "src.renderer"),
    ("src.engine", "src.ui_album"),
    ("src.fx", "src.ui_album"),
    ("src.renderer", "src.scene"),
    ("src.renderer", "src.ui_album"),
]


def _production_py_files() -> list[Path]:
    files = [p for p in import_graph.iter_py_files(ROOT / "src")]
    main_py = ROOT / "main.py"
    if main_py.exists():
        files.append(main_py)
    return files


def test_no_cycle_among_the_packages_r2_fixed():
    graph = import_graph.build_package_graph()
    offenders = []
    for a, b in _FIXED_PAIRS:
        if b in graph.get(a, ()) and a in graph.get(b, ()):
            offenders.append((a, b))
    assert offenders == [], f"cycle(s) reappeared: {offenders}"


def test_engine_app_is_imported_only_from_main():
    """``src.engine.app`` is the one module every OTHER layer is forbidden
    from reaching into directly — only ``main.py`` (the thin entry point)
    and ``src/engine/__init__.py`` (re-exporting its own submodule) may."""
    allowed = {ROOT / "main.py", ROOT / "src" / "engine" / "__init__.py"}
    offenders = []
    for path in _production_py_files():
        if path in allowed:
            continue
        for imp in import_graph.extract_imports(path):
            if imp == "src.engine.app" or imp.startswith("src.engine.app."):
                offenders.append(str(path.relative_to(ROOT)))
    assert offenders == [], f"src.engine.app imported outside main.py: {offenders}"


@pytest.mark.parametrize("pkg_a,pkg_b", _FIXED_PAIRS)
def test_fixed_pair_has_no_reverse_edge(pkg_a, pkg_b):
    """Belt-and-suspenders per-pair version of the cycle check above, so a
    single broken pair names itself in the failure instead of a list."""
    graph = import_graph.build_package_graph()
    assert not (pkg_b in graph.get(pkg_a, ()) and pkg_a in graph.get(pkg_b, ())), (
        f"{pkg_a} <-> {pkg_b} cycle reappeared")


#: TASK_arch_refactor_1120 R5: ``src/game`` is plain Python — match rules,
#: weapons, firing, hit resolution, scoring. It may import ``numpy`` and
#: ``src.config`` (and, transitively, ``src.content``, which itself only
#: depends on ``src.config``); nothing from the layers above it, and none
#: of the GL/windowing/image libraries those layers use.
_GAME_FORBIDDEN_PREFIXES = (
    "src.engine", "src.renderer", "src.ui_album", "src.fx", "src.net",
    "OpenGL", "glfw", "PIL",
)


def _all_top_level_imports(path: Path) -> list[str]:
    """Every import this file makes (unlike ``extract_imports``, not just
    ``src.*``) — needed to also catch ``OpenGL``/``glfw``/``PIL``."""
    import ast as _ast

    tree = _ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    out = []
    for node in _ast.walk(tree):
        if isinstance(node, _ast.Import):
            out.extend(alias.name for alias in node.names)
        elif isinstance(node, _ast.ImportFrom) and node.module and not node.level:
            out.append(node.module)
    return out


def test_game_package_has_no_upward_dependency():
    offenders = []
    for path in import_graph.iter_py_files(ROOT / "src" / "game"):
        for imp in _all_top_level_imports(path):
            if any(imp == p or imp.startswith(p + ".") for p in _GAME_FORBIDDEN_PREFIXES):
                offenders.append((str(path.relative_to(ROOT)), imp))
    assert offenders == [], f"src/game imports an upper layer: {offenders}"
