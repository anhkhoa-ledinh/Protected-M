"""Native OS file picker for media content."""

from __future__ import annotations

import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def pick_media_file() -> Path | None:
    """
    Open the OS native file dialog and return a selected media path.

    Uses tkinter (stdlib on Windows). Blocks until the user picks or cancels.
    """
    try:
        import tkinter as tk
        from tkinter import filedialog
    except ImportError:
        logger.error("tkinter unavailable — cannot open file dialog")
        return None

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
    root.destroy()

    if not selected:
        return None
    return Path(selected)
