"""src/shared/private_files: one property ("only I can read this") on both OSes.

The first tests use real files and are the proof on each OS. The Windows
decision logic runs everywhere through the seams of the module.
"""

from __future__ import annotations

import ast
import os
import subprocess
from pathlib import Path

import pytest

from src.shared import private_files as pf

ME = "S-1-5-21-1-2-3-1001"
OTHER = "S-1-1-0"


# ---- real files, whatever the OS ------------------------------------------------

def test_a_file_made_private_is_private(tmp_path):
    f = tmp_path / "secret.json"
    f.write_text("{}")
    pf.make_private(f)
    assert pf.is_private(f), pf.acl_report(f)


def test_a_folder_made_private_is_private(tmp_path):
    d = tmp_path / "data"
    d.mkdir()
    pf.make_private(d)
    assert pf.is_private(d), pf.acl_report(d)


# ---- a widened file is not private, then it is again ------------------------------

def _widen(path: Path) -> None:
    """Let other users read ``path``, in each system's own terms."""
    if os.name == "nt":
        subprocess.run(["icacls", str(path), "/grant", "*S-1-1-0:R"],
                       check=True, capture_output=True)
    else:
        os.chmod(path, 0o755 if path.is_dir() else 0o644)


@pytest.mark.parametrize("kind", ["file", "folder"])
def test_widening_breaks_privacy_and_make_private_restores_it(tmp_path, kind):
    target = tmp_path / "t"
    target.mkdir() if kind == "folder" else target.write_text("x")
    _widen(target)
    assert not pf.is_private(target), pf.acl_report(target)
    pf.make_private(target)
    assert pf.is_private(target), pf.acl_report(target)


# ---- Windows branch through its seams (runs on any OS) --------------------------

ALLOW, DENY = 0, 1
OI_CI = 0x3
INHERITED = 0x10
FULL = 0x001F01FF


def ace(kind=ALLOW, flags=0, sid=ME, mask=FULL):
    return (kind, flags, mask, sid)


@pytest.fixture
def win(monkeypatch):
    monkeypatch.setattr(pf, "_is_windows", lambda: True)
    monkeypatch.setattr(pf, "_current_sid", lambda: ME)


def test_windows_make_private_writes_a_file_dacl_for_a_file(win, monkeypatch, tmp_path):
    f = tmp_path / "remote_session.json"
    f.write_text("{}")
    calls = []
    monkeypatch.setattr(pf, "_write_private_dacl", lambda p, is_dir: calls.append((p, is_dir)))
    pf.make_private(f)
    assert calls == [(f, False)]


def test_windows_make_private_writes_an_inheritable_dacl_for_a_folder(win, monkeypatch,
                                                                      tmp_path):
    calls = []
    monkeypatch.setattr(pf, "_write_private_dacl", lambda p, is_dir: calls.append((p, is_dir)))
    pf.make_private(tmp_path)
    assert calls == [(tmp_path, True)]


def test_windows_a_failing_api_call_is_an_error_not_silence(win, monkeypatch, tmp_path):
    f = tmp_path / "f"
    f.write_text("x")

    def boom(*args, **kwargs):
        raise pf.PrivateFilesError("SetNamedSecurityInfoW failed (5)")

    monkeypatch.setattr(pf, "_write_private_dacl", boom)
    monkeypatch.setattr(pf, "_read_dacl", boom)
    with pytest.raises(pf.PrivateFilesError):
        pf.make_private(f)
    with pytest.raises(pf.PrivateFilesError):
        pf.is_private(f)


def test_windows_is_private_reads_the_dacl_and_the_current_sid(win, monkeypatch, tmp_path):
    seen = []
    monkeypatch.setattr(pf, "_read_dacl", lambda p: seen.append(p) or (True, [ace()]))
    assert pf.is_private(tmp_path)
    assert seen == [tmp_path]


def test_windows_only_me_protected_is_private():
    assert pf._dacl_is_private((True, [ace()]), ME)


def test_windows_a_folder_entry_that_children_inherit_is_private():
    assert pf._dacl_is_private((True, [ace(flags=OI_CI)]), ME)


def test_windows_inherited_entries_are_not_private():
    assert not pf._dacl_is_private((True, [ace(flags=INHERITED)]), ME)
    assert not pf._dacl_is_private((True, [ace(flags=OI_CI | INHERITED)]), ME)


def test_windows_other_principals_are_not_private():
    assert not pf._dacl_is_private((True, [ace(), ace(sid=OTHER)]), ME)
    assert not pf._dacl_is_private((True, [ace(sid=OTHER)]), ME)


def test_windows_a_deny_entry_is_not_private():
    assert not pf._dacl_is_private((True, [ace(kind=DENY)]), ME)
    assert not pf._dacl_is_private((True, [ace(), ace(kind=DENY)]), ME)


def test_windows_an_unprotected_dacl_is_not_private():
    assert not pf._dacl_is_private((False, [ace()]), ME)


def test_windows_the_default_unprotected_acl_is_not_private():
    default = (False, [ace(flags=INHERITED, sid="S-1-5-18"),
                       ace(flags=INHERITED, sid="S-1-5-32-544"),
                       ace(flags=INHERITED)])
    assert not pf._dacl_is_private(default, ME)


def test_windows_a_null_dacl_is_not_private():
    assert not pf._dacl_is_private(None, ME)


def test_windows_an_empty_acl_is_not_private():
    assert not pf._dacl_is_private((True, []), ME)


# ---- the ctypes table -----------------------------------------------------------

def _called_api_names() -> set[str]:
    tree = ast.parse(Path(pf.__file__).read_text(encoding="utf-8"))
    return {node.args[0].value for node in ast.walk(tree)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
            and node.func.id == "_api" and node.args
            and isinstance(node.args[0], ast.Constant)}


def test_every_win32_function_the_module_calls_is_in_the_typed_table():
    called = _called_api_names()
    assert called, "no _api(...) call found: the check is blind"
    assert called <= set(pf._WIN_API), sorted(called - set(pf._WIN_API))


def test_every_table_entry_has_a_dll_argtypes_and_restype():
    for name, (dll, argtypes, restype) in pf._WIN_API.items():
        assert dll in ("advapi32", "kernel32"), name
        assert isinstance(argtypes, list), name
        assert restype is not None, name


def test_no_win32_call_bypasses_the_table():
    tree = ast.parse(Path(pf.__file__).read_text(encoding="utf-8"))
    dll_attrs = {node.attr for node in ast.walk(tree) if isinstance(node, ast.Attribute)
                 and isinstance(node.value, ast.Name) and node.value.id in ("windll", "WinDLL")}
    assert not dll_attrs, dll_attrs


def test_linux_never_loads_a_windows_dll(monkeypatch, tmp_path):
    # The POSIX branch is simulated, so its mode bits are faked too: a real
    # chmod/stat on an NTFS host cannot show 0o600.
    monkeypatch.setattr(pf, "_is_windows", lambda: False)
    f = tmp_path / "f"
    f.write_text("x")
    monkeypatch.setattr(pf, "_dlls", {})

    chmod_calls = {}
    monkeypatch.setattr(pf.os, "chmod", lambda p, mode: chmod_calls.update({p: mode}))
    pf.make_private(f)
    assert chmod_calls == {f: 0o600}

    class _Stat:
        def __init__(self, mode):
            self.st_mode = mode

    with monkeypatch.context() as m:
        m.setattr(Path, "stat", lambda self, *a, **k: _Stat(0o100600))
        assert pf.is_private(f)
    with monkeypatch.context() as m:
        m.setattr(Path, "stat", lambda self, *a, **k: _Stat(0o100644))
        assert not pf.is_private(f)

    assert pf._dlls == {}
