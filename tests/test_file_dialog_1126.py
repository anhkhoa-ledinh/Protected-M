"""
A broken file picker (``tk.Tk()`` raising ``TclError``) must not close the
game: ``pick_media_file`` never raises, and tells "failed" from "cancelled".

Logic-only: a fake ``tkinter`` module, no real window.
"""

from __future__ import annotations

import sys
import types
from pathlib import Path
from types import SimpleNamespace

import pytest

from src.engine.app import PICKER_FAILED_TEXT, HcpdApplication
from src.gui import file_dialog
from src.ui_album import v3_flow as F


class FakeTclError(Exception):
    pass


def install_fake_tk(monkeypatch, *, tk_raises=None, dialog=None):
    """Put a fake ``tkinter`` in ``sys.modules``; returns the roots it made."""
    roots: list = []

    class Root:
        def __init__(self) -> None:
            if tk_raises is not None:
                raise tk_raises
            self.destroyed = 0
            roots.append(self)

        def withdraw(self) -> None:
            pass

        def attributes(self, *_a) -> None:
            pass

        def destroy(self) -> None:
            self.destroyed += 1

    tk = types.ModuleType("tkinter")
    tk.Tk = Root
    tk.TclError = FakeTclError
    fd = types.ModuleType("tkinter.filedialog")
    fd.askopenfilename = dialog or (lambda **_k: "")
    tk.filedialog = fd
    monkeypatch.setitem(sys.modules, "tkinter", tk)
    monkeypatch.setitem(sys.modules, "tkinter.filedialog", fd)
    return roots


def test_tk_cannot_start_returns_failed(monkeypatch, caplog):
    install_fake_tk(monkeypatch, tk_raises=FakeTclError("Can't find a usable init.tcl"))
    with caplog.at_level("ERROR"):
        result = file_dialog.pick_media_file()
    assert result is file_dialog.PICK_FAILED
    assert "init.tcl" in caplog.text


def test_dialog_raises_destroys_root_and_returns_failed(monkeypatch):
    def boom(**_k):
        raise FakeTclError("dialog broke")

    roots = install_fake_tk(monkeypatch, dialog=boom)
    assert file_dialog.pick_media_file() is file_dialog.PICK_FAILED
    assert [r.destroyed for r in roots] == [1]


def test_cancel_is_none_not_failed(monkeypatch):
    roots = install_fake_tk(monkeypatch, dialog=lambda **_k: "")
    result = file_dialog.pick_media_file()
    assert result is None
    assert result is not file_dialog.PICK_FAILED
    assert [r.destroyed for r in roots] == [1]


def test_pick_returns_path(monkeypatch):
    install_fake_tk(monkeypatch, dialog=lambda **_k: "/x/photo.png")
    assert file_dialog.pick_media_file() == Path("/x/photo.png")


# --- the app action -------------------------------------------------------

class FakeAlbum:
    flow = F

    def __init__(self) -> None:
        self.state = F.V3Flow()
        self.state.screen = F.PHOTOS
        self.actions: list[str] = []


def make_app(monkeypatch, picked):
    monkeypatch.setattr("src.engine.app.pick_media_file", lambda: picked)
    app = HcpdApplication.__new__(HcpdApplication)
    app._album = FakeAlbum()
    app._album_options = SimpleNamespace(is_v3=True)
    toasts: list = []
    app._renderer = SimpleNamespace(
        show_toast=lambda text, is_error=False: toasts.append((text, is_error)))
    app._window = SimpleNamespace(restore_and_focus=lambda: None)
    app._album_action = lambda action, now: app._album.actions.append(action)
    return app, toasts


def test_action_failed_stays_on_photos_and_toasts(monkeypatch):
    app, toasts = make_app(monkeypatch, file_dialog.PICK_FAILED)
    app._act_pick_file("pick_file", 0.0)
    assert app._album.state.screen == F.PHOTOS
    assert app._album.state.photo is None
    assert app._album.actions == []
    assert toasts == [(PICKER_FAILED_TEXT, True)]
    assert app._album.state.message == PICKER_FAILED_TEXT


def test_action_cancel_is_silent(monkeypatch):
    app, toasts = make_app(monkeypatch, None)
    app._act_pick_file("pick_file", 0.0)
    assert app._album.state.screen == F.PHOTOS
    assert app._album.actions == []
    assert toasts == []


def test_action_picked_starts_pipeline(monkeypatch):
    app, toasts = make_app(monkeypatch, Path("/x/a.png"))
    app._act_pick_file("pick_file", 0.0)
    assert app._album.actions == ["start_pipeline"]
    assert toasts == []
