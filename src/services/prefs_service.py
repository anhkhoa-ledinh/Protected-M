"""The two user preferences the UI reads, behind an interface so the UI does
not import the prefs file module (it is injected by ``app``)."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

from src.config import prefs


class PrefsService(ABC):
    @abstractmethod
    def fx_enabled(self) -> bool: ...

    @abstractmethod
    def screen_shake_enabled(self) -> bool: ...


class LocalPrefsService(PrefsService):
    """Reads ``user_prefs.json`` under ``root`` (a read-only view)."""

    def __init__(self, root) -> None:
        self._root = Path(root)

    def fx_enabled(self) -> bool:
        return prefs.fx_enabled(self._root)

    def screen_shake_enabled(self) -> bool:
        return prefs.screen_shake_enabled(self._root)
