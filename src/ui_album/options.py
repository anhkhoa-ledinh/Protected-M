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
    #: TASK_ui_v3_1107: bộ giao diện nào — chỉ còn ``"v3"`` (bản vẽ
    #: "v2 · Redesign"; bộ cũ của TASK_album_ui_1106 đã xoá ở R1 của
    #: TASK_arch_refactor_1120).
    ui: str = "v3"
    #: ``--ui-shots``: đi qua 23 màn v3, lưu một PNG mỗi màn rồi thoát.
    ui_shots: bool = False

    @property
    def is_v3(self) -> bool:
        return self.enabled and str(self.ui) == "v3"
