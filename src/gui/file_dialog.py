"""Native OS file picker for media content."""

from __future__ import annotations

import ctypes
import logging
import sys
from pathlib import Path

logger = logging.getLogger(__name__)


class _PickFailed:
    """Type of ``PICK_FAILED``: the picker could not run (not "cancelled")."""

    def __repr__(self) -> str:
        return "PICK_FAILED"

    def __bool__(self) -> bool:
        return False


#: Returned instead of a path when Tk could not start or the dialog broke.
#: ``None`` still means the player cancelled.
PICK_FAILED = _PickFailed()


#: The ONE list of image extensions; both backends build their filter from it.
IMAGE_EXTS = ("jpg", "jpeg", "png", "webp")
VIDEO_EXTS = ("mp4", "mov", "avi", "mkv")

# OPENFILENAMEW flags (commdlg.h). OFN_NOCHANGEDIR keeps the dialog from
# changing the process's working directory (relative asset paths).
OFN_NOCHANGEDIR = 0x00000008
OFN_PATHMUSTEXIST = 0x00000800
OFN_FILEMUSTEXIST = 0x00001000
OFN_EXPLORER = 0x00080000
#: Wide chars in the path buffer: long paths fit.
WIN32_PATH_CHARS = 32768


def _globs(exts) -> str:
    return " ".join(f"*.{e}" for e in exts)


def tk_filetypes() -> list:
    return [
        ("Media", _globs(IMAGE_EXTS + VIDEO_EXTS)),
        ("Images", _globs(IMAGE_EXTS)),
        ("Videos", _globs(VIDEO_EXTS)),
        ("All files", "*.*"),
    ]


def win32_filter() -> str:
    """``Images\0*.jpg;*.png...\0\0`` as ``GetOpenFileNameW`` wants it."""
    return "Images\0" + ";".join(f"*.{e}" for e in IMAGE_EXTS) + "\0\0"


def _current_platform() -> str:
    return sys.platform


def make_ofn_type():
    """The ``OPENFILENAMEW`` struct (built lazily: ``wintypes`` only here)."""
    from ctypes import wintypes as w

    class OPENFILENAMEW(ctypes.Structure):
        _fields_ = [
            ("lStructSize", w.DWORD), ("hwndOwner", w.HWND),
            ("hInstance", w.HINSTANCE), ("lpstrFilter", ctypes.c_void_p),
            ("lpstrCustomFilter", ctypes.c_void_p), ("nMaxCustFilter", w.DWORD),
            ("nFilterIndex", w.DWORD), ("lpstrFile", ctypes.c_void_p),
            ("nMaxFile", w.DWORD), ("lpstrFileTitle", ctypes.c_void_p),
            ("nMaxFileTitle", w.DWORD), ("lpstrInitialDir", w.LPCWSTR),
            ("lpstrTitle", w.LPCWSTR), ("Flags", w.DWORD),
            ("nFileOffset", w.WORD), ("nFileExtension", w.WORD),
            ("lpstrDefExt", w.LPCWSTR), ("lCustData", w.LPARAM),
            ("lpfnHook", ctypes.c_void_p), ("lpTemplateName", w.LPCWSTR),
            ("pvReserved", ctypes.c_void_p), ("dwReserved", w.DWORD),
            ("FlagsEx", w.DWORD),
        ]

    return OPENFILENAMEW


def _load_comdlg32():
    from ctypes import wintypes as w

    dll = ctypes.WinDLL("comdlg32")          # Windows only
    dll.GetOpenFileNameW.argtypes = [ctypes.POINTER(make_ofn_type())]
    dll.GetOpenFileNameW.restype = w.BOOL
    dll.CommDlgExtendedError.argtypes = []
    dll.CommDlgExtendedError.restype = w.DWORD
    return dll


def _owner_hwnd():
    """The game window's HWND from GLFW, or ``None``."""
    try:
        import glfw

        window = glfw.get_current_context()
        return int(glfw.get_win32_window(window)) if window else None
    except Exception:
        return None


def pick_win32() -> Path | None | _PickFailed:
    """The Win32 common dialog through ``ctypes``; no Tk. Same results as
    ``pick_tk``: a path, ``None`` for cancel, ``PICK_FAILED`` for an error."""
    try:
        comdlg32 = _load_comdlg32()
        ofn_type = make_ofn_type()
        filt = win32_filter()
        filt_buf = (ctypes.c_wchar * len(filt))(*filt)
        path_buf = ctypes.create_unicode_buffer(WIN32_PATH_CHARS)
        ofn = ofn_type()
        ofn.lStructSize = ctypes.sizeof(ofn_type)
        ofn.hwndOwner = _owner_hwnd()
        ofn.lpstrFilter = ctypes.addressof(filt_buf)
        ofn.nFilterIndex = 1
        ofn.lpstrFile = ctypes.addressof(path_buf)
        ofn.nMaxFile = WIN32_PATH_CHARS
        ofn.lpstrTitle = "Open image"
        ofn.Flags = (OFN_FILEMUSTEXIST | OFN_PATHMUSTEXIST | OFN_NOCHANGEDIR
                     | OFN_EXPLORER)
        if comdlg32.GetOpenFileNameW(ctypes.pointer(ofn)):
            return Path(path_buf.value)
        code = int(comdlg32.CommDlgExtendedError())
    except Exception as exc:
        logger.error("file dialog failed: %s", exc)
        return PICK_FAILED
    if code == 0:
        return None                           # the player cancelled
    logger.error("file dialog failed: CommDlgExtendedError 0x%04X", code)
    return PICK_FAILED


def choose_backend(platform: str | None = None):
    """``pick_win32`` on Windows, else ``pick_tk``; ``platform`` is injectable."""
    platform = _current_platform() if platform is None else platform
    return pick_win32 if platform.startswith("win") else pick_tk


def pick_media_file(platform: str | None = None) -> Path | None | _PickFailed:
    """Open the OS native file dialog and return a selected media path.

    Windows uses the Win32 common dialog (Tk's ``init.tcl`` is broken in a
    Python 3.13 venv there); other systems use Tk. Blocks until the user
    picks or cancels. Never raises: ``None`` = cancelled, ``PICK_FAILED`` =
    the dialog could not run (the reason is logged)."""
    return choose_backend(platform)()


def pick_tk() -> Path | None | _PickFailed:
    """
    Open the OS native file dialog and return a selected media path.

    Uses tkinter. Blocks until the user picks or cancels. Never raises:
    returns ``None`` when the player cancels and ``PICK_FAILED`` when Tk is
    missing or broken (e.g. ``TclError`` from ``tk.Tk()``), after logging
    the reason.
    """
    try:
        import tkinter as tk
        from tkinter import filedialog
    except ImportError:
        logger.error("tkinter unavailable — cannot open file dialog")
        return PICK_FAILED

    root = None
    try:
        root = tk.Tk()
        root.withdraw()
        try:
            root.attributes("-topmost", True)
        except tk.TclError:
            pass

        selected = filedialog.askopenfilename(
            title="Open image or video",
            filetypes=tk_filetypes(),
        )
    except tk.TclError as exc:
        logger.error("file dialog failed: %s", exc)
        return PICK_FAILED
    finally:
        if root is not None:
            try:
                root.destroy()
            except tk.TclError:
                pass

    if not selected:
        return None
    return Path(selected)
