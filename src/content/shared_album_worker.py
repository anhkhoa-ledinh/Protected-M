"""
Chạy MỘT việc mạng của ``shared_album`` (upload/list/download) ở luồng riêng.

Cùng lý do với ``DepthWorker``/``BakeWorker``: ``urllib.request`` có thể chặn
tới 15 s (``shared_album.TIMEOUT_S``), và luồng vẽ GL không được đứng chờ.
Việc này ngắn và một-lần (không phải vòng lặp liên tục như depth), nên một
``threading.Thread`` daemon đơn giản là đủ — không cần tiến trình con.
"""

from __future__ import annotations

import threading
from typing import Any, Callable

#: Trả về khi việc chưa xong. Khác ``None``: ``list_remote``/``download``
#: TRẢ ``None`` khi thật sự thất bại, và đó vẫn phải đọc được qua đây.
PENDING = object()


class SharedAlbumWorker:
    """Một việc một lúc: ``submit`` việc mới trong lúc đang bận thì bị bỏ qua."""

    def __init__(self) -> None:
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()
        self._result: Any = PENDING

    @property
    def busy(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def submit(self, fn: Callable[..., Any], *args: Any) -> bool:
        """Chạy ``fn(*args)`` ở luồng nền. ``False`` nếu đang bận việc khác."""
        if self.busy:
            return False
        with self._lock:
            self._result = PENDING

        def run() -> None:
            value = fn(*args)
            with self._lock:
                self._result = value

        self._thread = threading.Thread(target=run, daemon=True,
                                        name="shared-album")
        self._thread.start()
        return True

    def pop_result(self) -> Any:
        """Kết quả việc vừa xong (lấy một lần); ``PENDING`` nếu chưa xong."""
        with self._lock:
            if self._result is PENDING:
                return PENDING
            value, self._result = self._result, PENDING
            return value
