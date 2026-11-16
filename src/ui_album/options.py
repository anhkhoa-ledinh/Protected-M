"""
Cờ dòng lệnh của giao diện album (TASK_album_ui_1106).

Để riêng một module nhỏ vì ``main.py`` phải nhập được nó **mà không** kéo theo
OpenGL: các module vẽ chỉ nạp khi ``enabled`` bật.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class AlbumOptions:
    """Cờ ``--album*`` / ``--ui*``. Mặc định tắt = app chạy y hệt hôm nay."""

    enabled: bool = False
    guest: bool = False
    shots: bool = False
    seed_leaderboard: bool = False
    #: TASK_ui_v3_1107: bộ giao diện nào — ``"v3"`` (bản vẽ "v2 · Redesign",
    #: mặc định) hay ``"album"`` (bộ cũ của TASK_album_ui_1106, giữ nguyên để
    #: còn đường lùi cho buổi bảo vệ).
    ui: str = "v3"
    #: ``--ui-shots``: đi qua 23 màn v3, lưu một PNG mỗi màn rồi thoát.
    ui_shots: bool = False

    @property
    def is_v3(self) -> bool:
        return self.enabled and str(self.ui) == "v3"

    @property
    def is_album(self) -> bool:
        return self.enabled and str(self.ui) == "album"
