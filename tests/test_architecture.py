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


#: TASK_arch_refactor_1120 R6: ``net`` depends on ``game`` for match
#: rules/constants, never on the UI layer directly (the task's exact
#: wording names these three).
_NET_FORBIDDEN_PREFIXES = ("src.ui_album", "src.engine", "src.renderer")


def test_net_package_does_not_depend_on_ui_or_engine():
    offenders = []
    for path in import_graph.iter_py_files(ROOT / "src" / "net"):
        for imp in import_graph.extract_imports(path):
            if any(imp == p or imp.startswith(p + ".") for p in _NET_FORBIDDEN_PREFIXES):
                offenders.append((str(path.relative_to(ROOT)), imp))
    assert offenders == [], f"src/net imports ui_album/engine/renderer: {offenders}"


# ---- TASK_hardening_1124: one door for user photos ---------------------------

#: Modules that open an image file but never a user's photo: shipped art
#: (UI atlas, fx packs) or files this app wrote itself (bake and fill caches).
#: Anything else must get its pixels from ``media_gate``.
_OWN_FILE_IMAGE_READERS = {
    "src/ui_album/hud_cards.py", "src/ui_album/textures.py",
    "src/ui_album/board.py", "src/ui_album/v3_hover.py",
    "src/fx/spec.py", "src/fx/library.py", "src/fx/player.py",
    "src/content/bake.py", "src/content/lama_fill.py",
}
_GATE = "src/content/media_gate.py"
_IMAGE_DECODERS = {"cv2.imread", "cv2.imdecode", "PIL.Image.open",
                   "PIL.Image.frombytes", "PIL.Image.frombuffer"}


def _dotted(node) -> str:
    import ast as _ast

    parts = []
    while isinstance(node, _ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if isinstance(node, _ast.Name):
        parts.append(node.id)
    return ".".join(reversed(parts))


def _aliases(tree) -> dict[str, str]:
    """``import numpy as np`` -> {"np": "numpy"}; ``from cv2 import imwrite
    as w`` -> {"w": "cv2.imwrite"}. A call is judged by what it really is,
    not by the local name."""
    import ast as _ast

    out: dict[str, str] = {}
    for node in _ast.walk(tree):
        if isinstance(node, _ast.Import):
            for alias in node.names:
                out[(alias.asname or alias.name).split(".")[0]] = (
                    alias.name if alias.asname else alias.name.split(".")[0])
        elif isinstance(node, _ast.ImportFrom) and node.module and not node.level:
            for alias in node.names:
                out[alias.asname or alias.name] = f"{node.module}.{alias.name}"
    return out


def _canonical(node, aliases: dict[str, str]) -> str:
    name = _dotted(node)
    head, _, rest = name.partition(".")
    full = aliases.get(head, head)
    return f"{full}.{rest}" if rest else full


def _decoder_calls() -> dict[str, set[str]]:
    import ast as _ast

    found: dict[str, set[str]] = {}
    for path in _production_py_files():
        tree = _ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        aliases = _aliases(tree)
        for node in _ast.walk(tree):
            if (isinstance(node, _ast.Call)
                    and _canonical(node.func, aliases) in _IMAGE_DECODERS):
                found.setdefault(str(path.relative_to(ROOT)).replace("\\", "/"),
                                 set()).add(_canonical(node.func, aliases))
    return found


def test_only_the_media_gate_decodes_user_photos():
    offenders = {mod: calls for mod, calls in _decoder_calls().items()
                 if mod != _GATE and mod not in _OWN_FILE_IMAGE_READERS}
    assert offenders == {}, (
        f"decode a user photo through src.content.media_gate, not directly: {offenders}")


def test_cv2_imdecode_exists_only_in_the_media_gate():
    uses = {mod for mod, calls in _decoder_calls().items() if "cv2.imdecode" in calls}
    assert uses <= {_GATE}, f"cv2.imdecode used outside the gate: {uses}"


def test_the_own_file_reader_list_has_no_stale_entries():
    calls = _decoder_calls()
    stale = sorted(m for m in _OWN_FILE_IMAGE_READERS if m not in calls)
    assert stale == [], f"remove from _OWN_FILE_IMAGE_READERS (no decoder call): {stale}"


# ---- TASK_hardening_1124: camera data stays local -----------------------------

#: What may hold camera frames or landmarks. ``src.renderer.webcam_preview``
#: only shows the frame on screen (GL texture).
_CAMERA_PREFIXES = ("src.camera", "src.tracking", "src.pose",
                    "src.renderer.webcam_preview")
#: Code that talks to the network, stores accounts or shows the album UI must
#: never see a camera frame, directly or through anything it imports.
_CAMERA_FREE = ("src/net", "src/account", "src/content/shared_album.py",
                "src/content/shared_album_worker.py", "src/ui_album")


def _rel(path: Path) -> str:
    return str(path.relative_to(ROOT)).replace("\\", "/")


def _module_files() -> dict[str, Path]:
    return {import_graph.module_name(p): p for p in _production_py_files()
            if p.name != "main.py"}


def _is_camera(module: str) -> bool:
    return any(module == p or module.startswith(p + ".") for p in _CAMERA_PREFIXES)


def _resolve(name: str, modules: dict[str, Path]) -> str | None:
    """``src.a.b.c`` -> the longest prefix that is a real module/package."""
    parts = name.split(".")
    while parts:
        candidate = ".".join(parts)
        if candidate in modules:
            return candidate
        parts.pop()
    return None


def _camera_paths(start: Path, modules: dict[str, Path]) -> list[str]:
    """The import chain from ``start`` to a camera module, or ``[]``."""
    first = import_graph.module_name(start)
    queue, seen = [(first, [first])], {first}
    while queue:
        current, chain = queue.pop(0)
        path = modules.get(current)
        if path is None:
            continue
        for imp in import_graph.extract_imports(path):
            if _is_camera(imp):
                return chain + [imp]
            target = _resolve(imp, modules)
            if target and target not in seen:
                seen.add(target)
                queue.append((target, chain + [target]))
    return []


def test_net_account_shared_album_and_album_ui_never_reach_camera_code():
    modules = _module_files()
    offenders = {}
    for path in _production_py_files():
        rel = _rel(path)
        if any(rel == p or rel.startswith(p + "/") for p in _CAMERA_FREE):
            chain = _camera_paths(path, modules)
            if chain:
                offenders[rel] = " -> ".join(chain)
    assert offenders == {}, f"camera/tracking/pose reachable from: {offenders}"


#: Calls that write an image or an array to disk.
_FRAME_WRITERS = {"cv2.imwrite", "cv2.VideoWriter", "numpy.save", "numpy.savez",
                  "numpy.savez_compressed", "imageio.imwrite", "imageio.v3.imwrite"}
#: The only modules that write one, and what they write: baked layers of a
#: photo (bake, lama_fill), the photo gate's clean copy, F12 screenshots of the
#: GL framebuffer. ``src/devtools`` is exempt (evidence tooling).
_FRAME_WRITER_MODULES = {"src/content/bake.py", "src/content/lama_fill.py",
                         "src/content/media_gate.py", "src/ui_album/shots.py"}


def _writer_calls() -> dict[str, set[str]]:
    import ast as _ast

    found: dict[str, set[str]] = {}
    for path in _production_py_files():
        tree = _ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        aliases = _aliases(tree)
        for node in _ast.walk(tree):
            if not isinstance(node, _ast.Call):
                continue
            name = _canonical(node.func, aliases)
            on_array_image = (isinstance(node.func, _ast.Attribute)
                              and node.func.attr == "save"
                              and isinstance(node.func.value, _ast.Call)
                              and _canonical(node.func.value.func, aliases)
                              .endswith("fromarray"))
            if name in _FRAME_WRITERS or on_array_image:
                found.setdefault(_rel(path), set()).add(name or "Image.fromarray().save")
    return found


def test_only_known_modules_write_images_and_none_outside_devtools_sees_a_camera():
    outside = {m: c for m, c in _writer_calls().items()
               if not m.startswith("src/devtools/")}
    unknown = {m: c for m, c in outside.items() if m not in _FRAME_WRITER_MODULES}
    assert unknown == {}, f"new image/array writer outside devtools: {unknown}"

    modules = _module_files()
    touching = {}
    for module in outside:
        chain = _camera_paths(ROOT / module, modules)
        if chain:
            touching[module] = " -> ".join(chain)
    assert touching == {}, f"a module that writes images can reach the camera: {touching}"


#: B1. ``src.fx`` may not import ``ui_album`` (see ``_FIXED_PAIRS``), so its
#: screen table keeps copies; ``tests/test_ui_ids.py`` checks they match.
_ID_LITERAL_EXEMPT = {"src/fx/screens.py", "src/ui_album/ids.py"}


def test_no_ui_id_literal_outside_ids_py():
    import ast
    import importlib

    ids = importlib.import_module("src.ui_album.ids")
    wanted = {v for k, v in vars(ids).items() if k.isupper() and isinstance(v, str)}
    wanted |= set(ids.NOT_USED_IN_CODE)
    offenders = []
    for path in _production_py_files():
        rel = path.relative_to(ROOT).as_posix()
        if rel in _ID_LITERAL_EXEMPT:
            continue
        tree = ast.parse(path.read_text("utf-8"))
        docstrings = {id(n.body[0].value) for n in ast.walk(tree)
                      if isinstance(n, (ast.Module, ast.ClassDef, ast.FunctionDef,
                                        ast.AsyncFunctionDef))
                      and n.body and isinstance(n.body[0], ast.Expr)}
        for node in ast.walk(tree):
            if (isinstance(node, ast.Constant) and isinstance(node.value, str)
                    and node.value in wanted and id(node) not in docstrings):
                offenders.append(f"{rel}:{node.lineno} {node.value!r}")
    assert offenders == [], "use src.ui_album.ids instead of the literal:\n" + "\n".join(offenders)


def test_ui_album_never_imports_the_account_package():
    """B2: the UI gets an ``AccountService`` injected; it must not import
    ``src.account`` (the rules it needs live in ``src.shared.account_rules``)."""
    offenders = []
    for path in _production_py_files():
        if "src/ui_album/" not in path.as_posix():
            continue
        for imp in import_graph.extract_imports(path):
            if imp == "src.account" or imp.startswith("src.account."):
                offenders.append(f"{path.relative_to(ROOT)}: {imp}")
    assert offenders == [], offenders


def test_ui_album_imports_neither_the_shared_album_nor_prefs_modules():
    """B3: the UI asks injected services (``src.services``); it never imports
    ``src.content.shared_album`` or ``src.config.prefs`` itself."""
    banned = {"src.content.shared_album", "src.config.prefs"}
    offenders = []
    for path in _production_py_files():
        if "src/ui_album/" not in path.as_posix():
            continue
        for imp in import_graph.extract_imports(path):
            if imp in banned or any(imp.startswith(b + ".") for b in banned):
                offenders.append(f"{path.relative_to(ROOT)}: {imp}")
    assert offenders == [], offenders


def test_ui_album_imports_only_game_view_from_src_game():
    """B4: the UI reads a ``MatchView`` snapshot; the one ``src.game`` module
    it may import is ``src.game.view``."""
    offenders = []
    for path in _production_py_files():
        if "src/ui_album/" not in path.as_posix():
            continue
        for imp in import_graph.extract_imports(path):
            if (imp == "src.game" or imp.startswith("src.game.")) and imp != "src.game.view":
                offenders.append(f"{path.relative_to(ROOT)}: {imp}")
    assert offenders == [], offenders
