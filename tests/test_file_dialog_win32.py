"""The Windows file picker goes through comdlg32, never tkinter.

Runs on every OS: the platform and ``comdlg32`` are injected, no real dialog,
no real DLL, and nothing here looks at the real platform.
"""

from __future__ import annotations

import ctypes
import sys
from pathlib import Path

import pytest

from src.gui import file_dialog


@pytest.fixture(autouse=True)
def _no_glfw(monkeypatch):
    monkeypatch.setattr(file_dialog, "_owner_hwnd", lambda: None)


class FakeComdlg32:
    """``GetOpenFileNameW`` fills the buffer (or not) and returns ``result``."""

    def __init__(self, result=1, error=0, path="C:\\pics\\a.png"):
        self.result, self.error, self.path = result, error, path
        self.seen = None

    def GetOpenFileNameW(self, ofn_ptr):  # noqa: N802
        ofn = ofn_ptr.contents
        self.seen = {
            "flags": ofn.Flags,
            "size": ofn.lStructSize,
            "max_file": ofn.nMaxFile,
            "owner": ofn.hwndOwner,
            "filter": ctypes.wstring_at(ofn.lpstrFilter,
                                        len(file_dialog.win32_filter())),
        }
        if self.result:
            buf = (ctypes.c_wchar * ofn.nMaxFile).from_address(ofn.lpstrFile)
            buf.value = self.path
        return self.result

    def CommDlgExtendedError(self):  # noqa: N802
        return self.error


def _no_tk(monkeypatch):
    for name in ("tkinter", "tkinter.filedialog"):
        monkeypatch.delitem(sys.modules, name, raising=False)

    class Guard:
        def find_spec(self, name, path=None, target=None):
            if name.split(".")[0] == "tkinter":
                raise AssertionError("tkinter was imported")
            return None

    guard = Guard()
    monkeypatch.setattr(sys, "meta_path", [guard, *sys.meta_path])


def test_backend_choice_by_platform():
    assert file_dialog.choose_backend("win32") is file_dialog.pick_win32
    assert file_dialog.choose_backend("linux") is file_dialog.pick_tk
    assert file_dialog.choose_backend("darwin") is file_dialog.pick_tk


def test_win32_never_imports_tkinter(monkeypatch):
    _no_tk(monkeypatch)
    fake = FakeComdlg32()
    monkeypatch.setattr(file_dialog, "_load_comdlg32", lambda: fake)
    result = file_dialog.pick_media_file("win32")
    assert result == Path("C:\\pics\\a.png")
    assert "tkinter" not in sys.modules


def test_filter_string_is_built_from_the_shared_extension_list():
    exts = file_dialog.IMAGE_EXTS
    expected = "Images\0" + ";".join(f"*.{e}" for e in exts) + "\0\0"
    assert file_dialog.win32_filter() == expected
    assert expected.startswith("Images\0*.")
    assert expected.endswith("\0\0")


def test_tk_filetypes_use_the_same_extension_list():
    images = dict(file_dialog.tk_filetypes())["Images"]
    assert images.split() == [f"*.{e}" for e in file_dialog.IMAGE_EXTS]


def test_filter_reaches_the_dialog_intact(monkeypatch):
    fake = FakeComdlg32()
    monkeypatch.setattr(file_dialog, "_load_comdlg32", lambda: fake)
    file_dialog.pick_media_file("win32")
    want = file_dialog.win32_filter()
    assert fake.seen["filter"] == want


def test_flags_and_buffer(monkeypatch):
    fake = FakeComdlg32()
    monkeypatch.setattr(file_dialog, "_load_comdlg32", lambda: fake)
    file_dialog.pick_media_file("win32")
    flags = fake.seen["flags"]
    for bit in (file_dialog.OFN_NOCHANGEDIR, file_dialog.OFN_FILEMUSTEXIST,
                file_dialog.OFN_PATHMUSTEXIST):
        assert flags & bit == bit
    assert file_dialog.OFN_NOCHANGEDIR == 0x8
    assert fake.seen["max_file"] >= 32768
    assert fake.seen["size"] == ctypes.sizeof(file_dialog.make_ofn_type())


def test_cancel_is_none(monkeypatch):
    fake = FakeComdlg32(result=0, error=0)
    monkeypatch.setattr(file_dialog, "_load_comdlg32", lambda: fake)
    result = file_dialog.pick_media_file("win32")
    assert result is None and result is not file_dialog.PICK_FAILED


def test_real_error_is_pick_failed_and_logged(monkeypatch, caplog):
    fake = FakeComdlg32(result=0, error=0x3002)
    monkeypatch.setattr(file_dialog, "_load_comdlg32", lambda: fake)
    with caplog.at_level("ERROR"):
        result = file_dialog.pick_media_file("win32")
    assert result is file_dialog.PICK_FAILED
    assert "0x3002" in caplog.text


def test_no_dll_is_pick_failed(monkeypatch):
    def boom():
        raise OSError("no comdlg32")

    monkeypatch.setattr(file_dialog, "_load_comdlg32", boom)
    assert file_dialog.pick_media_file("win32") is file_dialog.PICK_FAILED


def test_owner_window_is_passed_when_known(monkeypatch):
    fake = FakeComdlg32()
    monkeypatch.setattr(file_dialog, "_load_comdlg32", lambda: fake)
    monkeypatch.setattr(file_dialog, "_owner_hwnd", lambda: 1234)
    file_dialog.pick_media_file("win32")
    assert fake.seen["owner"] == 1234
