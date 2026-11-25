"""Guards against tests that only mean something on one platform.

CI must be green because code and tests are right on Windows and Linux, not
because a check was avoided. This file fails on:

* a new ``skip`` / ``skipif`` / ``xfail`` / ``importorskip`` in tests/ that is
  not in ``ALLOWED_SKIPS`` (today's resource skips: no OpenGL driver, no
  webcam, missing model files, and the few listed below);
* a POSIX mode-bit check (``st_mode``, ``S_IMODE``, ``& 0o077``) in tests/ --
  ask ``src.shared.private_files.is_private`` instead;
* a ``subprocess`` call to ``bash`` / ``sh`` in tests/;
* a workflow that runs fewer tests (``continue-on-error``, ``--deselect``, ``-k``,
  an ``if:`` on the pytest step).
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TESTS = ROOT / "tests"
WORKFLOW = ROOT / ".github" / "workflows" / "tests.yml"

#: Files that may look at mode bits / use ``bash``: they test the helper itself.
EXEMPT = {"tests/test_private_files.py", "tests/test_portability.py"}

#: (file, enclosing function or class, kind) -> how many. These are skips for a
#: missing *resource*, each with a reason in the call. Today's lines (2025-11-25):
#:   conftest.py:98,103,111,139 gl_context   conftest.py:155,166 webcam
#:   conftest.py:216,222 model files          test_live_resize.py:361-387 GL window
#:   test_v3ui_fx_hooks.py:86-105 GL window   test_depth_backend.py:58 ONNX file
#:   test_media_gate.py:41,86 crafted sample the OS cannot create (symlinks)
#:   test_prebleed_invariant.py:43 file not in git HEAD yet
ALLOWED_SKIPS = {
    ("tests/conftest.py", "gl_context", "skip"): 4,
    ("tests/conftest.py", "webcam", "skip"): 2,
    ("tests/conftest.py", "require_depth_anything", "skip"): 1,
    ("tests/conftest.py", "require_lama", "skip"): 1,
    ("tests/test_live_resize.py", "msaa_gl_window", "skip"): 5,
    ("tests/test_v3ui_fx_hooks.py", "_module_gl_window", "skip"): 4,
    ("tests/test_depth_backend.py", "TestOnnxMatchesTorch", "skipUnless"): 1,
    ("tests/test_media_gate.py", "test_crafted_file_is_rejected_without_a_crash", "skip"): 1,
    ("tests/test_media_gate.py",
     "test_symlink_that_leaves_the_photo_folder_is_rejected", "skip"): 1,
    ("tests/test_prebleed_invariant.py",
     "test_prebleed_only_changes_rgb_under_alpha_zero", "skip"): 1,
}

_SKIP_NAMES = {"skip", "skipif", "xfail", "importorskip", "skipUnless", "skipIf",
               "skipTest"}
_SUBPROCESS = {"run", "Popen", "call", "check_call", "check_output"}
_SHELLS = {"bash", "sh", "/bin/bash", "/bin/sh"}


def _skip_kind(node: ast.AST) -> str | None:
    """``skip``/``xfail``/... when ``node`` is such a call, or a bare decorator
    (``@pytest.mark.xfail``). ``pytest.skip.Exception`` is neither."""
    target = node.func if isinstance(node, ast.Call) else node
    if not isinstance(target, (ast.Attribute, ast.Name)):
        return None
    last = target.attr if isinstance(target, ast.Attribute) else target.id
    return last if last in _SKIP_NAMES else None


def _walk(node: ast.AST, scope: str):
    """Yield ``(node, innermost enclosing def/class name)``. A decorator belongs
    to the thing it decorates; a bare decorator is yielded as a counted skip."""
    yield node, scope
    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
        for dec in node.decorator_list:
            yield from _walk(dec, node.name)
        for sub in node.body:
            yield from _walk(sub, node.name)
        return
    for child in ast.iter_child_nodes(node):
        yield from _walk(child, scope)


def _skips_in(tree: ast.Module):
    """``(scope, kind)`` for every skip call and every bare skip decorator."""
    for node, scope in _walk(tree, "<module>"):
        if isinstance(node, ast.Call) and _skip_kind(node):
            yield scope, _skip_kind(node)
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            for dec in node.decorator_list:
                if not isinstance(dec, ast.Call) and _skip_kind(dec):
                    yield node.name, _skip_kind(dec)


def _first_literal(call: ast.Call):
    if not call.args:
        return None
    arg = call.args[0]
    if isinstance(arg, (ast.List, ast.Tuple)) and arg.elts:
        arg = arg.elts[0]
    if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
        return arg.value
    return None


def scan_source(source: str, rel: str) -> dict:
    """What ``source`` (a test file at ``rel``) does that the guard cares about."""
    tree = ast.parse(source)
    found = {"skips": {}, "mode_bits": [], "shell": []}
    for scope, kind in _skips_in(tree):
        key = (rel, scope, kind)
        found["skips"][key] = found["skips"].get(key, 0) + 1
    for node in ast.walk(tree):
        if isinstance(node, (ast.Attribute, ast.Name)):
            name = node.attr if isinstance(node, ast.Attribute) else node.id
            if name in ("st_mode", "S_IMODE"):
                found["mode_bits"].append((node.lineno, name))
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.BitAnd):
            for side in (node.left, node.right):
                text = ast.get_source_segment(source, side) or ""
                if isinstance(side, ast.Constant) and isinstance(side.value, int) \
                        and text.lower().startswith("0o"):
                    found["mode_bits"].append((node.lineno, f"& {text}"))
        if isinstance(node, ast.Call):
            name = ast.unparse(node.func)
            lit = _first_literal(node)
            shell_lit = lit is not None and (lit in _SHELLS or lit.startswith(("bash ", "sh ")))
            if shell_lit and ((name.startswith("subprocess.")
                               and name.split(".")[-1] in _SUBPROCESS)
                              or name in ("os.system", "os.popen")):
                found["shell"].append((node.lineno, name))
    return found


def _scan_tests():
    skips: dict = {}
    mode_bits, shell = [], []
    for path in sorted(TESTS.rglob("*.py")):
        rel = path.relative_to(ROOT).as_posix()
        found = scan_source(path.read_text(encoding="utf-8"), rel)
        for key, n in found["skips"].items():
            skips[key] = skips.get(key, 0) + n
        if rel in EXEMPT:
            continue
        mode_bits += [(rel, ln, what) for ln, what in found["mode_bits"]]
        shell += [(rel, ln, what) for ln, what in found["shell"]]
    return skips, mode_bits, shell


# ---- the real tests/ tree ---------------------------------------------------------

def test_no_new_skip_or_xfail():
    skips, _, _ = _scan_tests()
    new = {k: n for k, n in skips.items() if n > ALLOWED_SKIPS.get(k, 0)}
    assert not new, (
        "new skip/skipif/xfail in tests (make the test pass on every OS instead, or, "
        f"for a missing resource, ask for it to be added to ALLOWED_SKIPS): {new}")


def test_allowed_skips_still_exist():
    """An allow-list entry that no longer matches hides a future new skip."""
    skips, _, _ = _scan_tests()
    stale = {k: n for k, n in ALLOWED_SKIPS.items() if skips.get(k, 0) != n}
    assert not stale, f"update ALLOWED_SKIPS, these no longer match: {stale}"


def test_no_posix_mode_bit_checks_outside_private_files():
    _, mode_bits, _ = _scan_tests()
    assert not mode_bits, (
        f"mode bits are POSIX-only; assert is_private(path) instead: {mode_bits}")


def test_tests_never_run_bash():
    _, _, shell = _scan_tests()
    assert not shell, f"bash is not on every runner (and the Windows one is a stub): {shell}"


def test_workflow_runs_every_test():
    text = WORKFLOW.read_text(encoding="utf-8")
    assert "continue-on-error" not in text
    assert "--deselect" not in text and "--ignore" not in text
    lines = text.splitlines()
    at = [i for i, ln in enumerate(lines) if "python -m pytest" in ln]
    assert len(at) == 1 and lines[at[0]].strip() == "- run: python -m pytest -q -ra"
    assert not lines[at[0] - 1].strip().startswith("if:")     # not conditional


# ---- the scanner itself catches the things it is for -----------------------------------

BAD_SKIPIF = '''
import pytest, sys
@pytest.mark.skipif(sys.platform == "win32", reason="posix only")
def test_x():
    pass
'''
BAD_XFAIL = "import pytest\n@pytest.mark.xfail\ndef test_x():\n    pass\n"
BAD_CALL_SKIP = "import pytest\ndef test_x():\n    pytest.skip('no')\n"
BAD_IMPORTORSKIP = "import pytest\nnp = pytest.importorskip('numpy')\n"
BAD_MODE = "import os\ndef test_x(p):\n    assert os.stat(p).st_mode & 0o077 == 0\n"
BAD_IMODE = "import stat, os\ndef test_x(p):\n    assert stat.S_IMODE(os.stat(p)) == 0o600\n"
BAD_BASH = "import subprocess\ndef test_x():\n    subprocess.run(['bash', 'x.sh'])\n"
GOOD = '''
import pytest
def test_x(request):
    with pytest.raises(pytest.skip.Exception):
        request.getfixturevalue("gl_context")
'''


def test_scanner_flags_skipif_xfail_skip_and_importorskip():
    for src, kind in ((BAD_SKIPIF, "skipif"), (BAD_XFAIL, "xfail"),
                      (BAD_CALL_SKIP, "skip"), (BAD_IMPORTORSKIP, "importorskip")):
        skips = scan_source(src, "tests/test_new.py")["skips"]
        assert skips, kind
        assert all(k not in ALLOWED_SKIPS for k in skips), kind


def test_scanner_flags_mode_bits_and_bash():
    assert scan_source(BAD_MODE, "t.py")["mode_bits"]
    assert scan_source(BAD_IMODE, "t.py")["mode_bits"]
    assert scan_source(BAD_BASH, "t.py")["shell"]


def test_scanner_accepts_catching_a_skip_in_a_fixture_test():
    found = scan_source(GOOD, "t.py")
    assert not found["skips"] and not found["mode_bits"] and not found["shell"]
