"""src/shared/private_files: one property ("only I can read this") on both OSes.

The POSIX tests use real files. The Windows branch runs everywhere by faking
``icacls``: we check the exact command line and the parsing of its output.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest

from src.shared import private_files as pf

USER = "PC\\anh"


# ---- real files, whatever the OS ------------------------------------------------

def test_a_file_made_private_is_private(tmp_path):
    f = tmp_path / "secret.json"
    f.write_text("{}")
    pf.make_private(f)
    assert pf.is_private(f)


def test_a_folder_made_private_is_private(tmp_path):
    d = tmp_path / "data"
    d.mkdir()
    pf.make_private(d)
    assert pf.is_private(d)


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
    assert not pf.is_private(target)
    pf.make_private(target)
    assert pf.is_private(target)


# ---- Windows branch, with a fake icacls (runs on any OS) -----------------------

class FakeIcacls:
    def __init__(self, listing: str = "", fail: bool = False):
        self.calls: list[list[str]] = []
        self.listing = listing
        self.fail = fail

    def __call__(self, args):
        self.calls.append(list(args))
        if self.fail:
            raise pf.PrivateFilesError("icacls failed (5): Access is denied.")
        return self.listing if len(args) == 1 else "Successfully processed 1 files\n"


@pytest.fixture
def win(monkeypatch):
    def install(listing="", fail=False):
        fake = FakeIcacls(listing, fail)
        monkeypatch.setattr(pf, "_is_windows", lambda: True)
        monkeypatch.setattr(pf, "_current_user", lambda: USER)
        monkeypatch.setattr(pf, "_icacls", fake)
        return fake
    return install


def test_windows_file_drops_inheritance_and_grants_only_me(win, tmp_path):
    f = tmp_path / "remote_session.json"
    f.write_text("{}")
    fake = win()
    pf.make_private(f)
    assert fake.calls == [[str(f), "/inheritance:r", "/grant:r", f"{USER}:F"]]


def test_windows_folder_grant_is_inherited_by_its_content(win, tmp_path):
    fake = win()
    pf.make_private(tmp_path)
    assert fake.calls == [[str(tmp_path), "/inheritance:r", "/grant:r",
                           f"{USER}:(OI)(CI)F"]]


def test_windows_a_failing_icacls_is_an_error_not_silence(win, tmp_path):
    f = tmp_path / "f"
    f.write_text("x")
    win(fail=True)
    with pytest.raises(pf.PrivateFilesError):
        pf.make_private(f)
    with pytest.raises(pf.PrivateFilesError):
        pf.is_private(f)


def _listing(path: Path, *aces: str) -> str:
    lines = [f"{path} {aces[0]}"] + [f"{' ' * (len(str(path)) + 1)}{a}" for a in aces[1:]]
    return "\n".join(lines) + "\n\nSuccessfully processed 1 files; Failed processing 0 files\n"


def test_windows_only_me_is_private(win, tmp_path):
    f = tmp_path / "f"
    win(_listing(f, f"{USER}:(F)"))
    assert pf.is_private(f)


def test_windows_inherited_entries_are_not_private(win, tmp_path):
    f = tmp_path / "f"
    win(_listing(f, f"{USER}:(I)(F)"))
    assert not pf.is_private(f)


def test_windows_other_principals_are_not_private(win, tmp_path):
    f = tmp_path / "f"
    win(_listing(f, f"{USER}:(F)", "BUILTIN\\Users:(R)"))
    assert not pf.is_private(f)
    win(_listing(f, f"{USER}:(F)", "NT AUTHORITY\\SYSTEM:(F)"))
    assert not pf.is_private(f)


def test_windows_the_default_unprotected_acl_is_not_private(win, tmp_path):
    f = tmp_path / "f"
    win(_listing(f, "NT AUTHORITY\\SYSTEM:(I)(F)", "BUILTIN\\Administrators:(I)(F)",
                 f"{USER}:(I)(F)"))
    assert not pf.is_private(f)


def test_windows_an_empty_listing_is_not_private(win, tmp_path):
    f = tmp_path / "f"
    win("Successfully processed 1 files; Failed processing 0 files\n")
    assert not pf.is_private(f)


def test_windows_folder_flags_are_parsed(win, tmp_path):
    win(_listing(tmp_path, f"{USER}:(OI)(CI)(F)"))
    assert pf.is_private(tmp_path)
