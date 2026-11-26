"""Native OS file picker for media content."""

from __future__ import annotations

import logging
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


def pick_media_file() -> Path | None | _PickFailed:
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

        filetypes = [
            ("Media", "*.jpg *.jpeg *.png *.webp *.mp4 *.mov *.avi *.mkv"),
            ("Images", "*.jpg *.jpeg *.png *.webp"),
            ("Videos", "*.mp4 *.mov *.avi *.mkv"),
            ("All files", "*.*"),
        ]
        selected = filedialog.askopenfilename(
            title="Open image or video",
            filetypes=filetypes,
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
