"""TASK_perf_textures_1119 Fix 3: decode PNGs off the render thread.

A single background worker thread does the CPU work (``load_rgba`` —
decode, bleed-if-needed, resize) that used to run synchronously inside
``V3Painter.bind_screen()`` the first time each screen drew. The render
thread never decodes; it only ever calls ``request()`` (cheap, just
queues) and ``poll()`` (drains whatever decode jobs finished since the
last call, non-blocking) before doing the GL upload itself — GL calls
must stay on the thread that owns the context.

Pure Python/threading, no GL — testable without a real window.
"""
from __future__ import annotations

import queue
import threading

from src.ui_album.textures import load_rgba

#: Lower number = served first. The current screen's own textures jump
#: ahead of prefetched neighbours.
PRIORITY_CURRENT = 0
PRIORITY_PREFETCH = 1


class AsyncTextureLoader:
    """Queues ``(path, max_width)`` decode jobs; a single worker thread
    runs ``load_rgba`` on them; ``poll()`` drains finished ones."""

    def __init__(self) -> None:
        self._queue: queue.PriorityQueue = queue.PriorityQueue()
        self._pending: set[tuple] = set()
        self._ready: queue.Queue = queue.Queue()
        self._seq = 0
        self._lock = threading.Lock()
        self._stop = threading.Event()
        #: TASK_perf_textures_1119 Fix 3: measured on this machine — the
        #: worker thread's PIL/cv2 calls DO release the GIL (verified: the
        #: main thread keeps making steady progress while they run), but
        #: scrypt hashing (``src/account``, a different background thread
        #: for sign-in/up) apparently does not release it as cleanly, and
        #: the two competing pushed single frames during the accounts flow
        #: past 50ms even with decode off the render thread. ``pause()``
        #: lets ``V3UI.tick()`` idle this thread for the ~100-300ms a hash
        #: job runs, so it never competes for the GIL during that window.
        self._allowed = threading.Event()
        self._allowed.set()
        self._thread = threading.Thread(target=self._run, daemon=True,
                                        name="texture-loader")
        self._thread.start()

    def pause(self) -> None:
        self._allowed.clear()

    def resume(self) -> None:
        self._allowed.set()

    def request(self, key: tuple, path, max_width, priority: int = PRIORITY_CURRENT) -> None:
        """Queues a decode job for ``key`` unless already pending or in
        flight. Cheap — never touches the file itself."""
        with self._lock:
            if key in self._pending:
                return
            self._pending.add(key)
            self._seq += 1
            seq = self._seq
        self._queue.put((priority, seq, key, path, max_width))

    def is_pending(self, key: tuple) -> bool:
        with self._lock:
            return key in self._pending

    def poll(self) -> list[tuple]:
        """Non-blocking: every ``(key, array)`` finished since the last
        call. ``array`` is ``None`` if the file failed to decode."""
        out = []
        while True:
            try:
                item = self._ready.get_nowait()
            except queue.Empty:
                break
            out.append(item)
        return out

    def stop(self) -> None:
        """Signals the worker to exit; does not block (tests/``release()``
        don't need to wait for an in-flight decode to finish)."""
        self._stop.set()
        self._allowed.set()                           # unblock if paused
        self._queue.put((-1, -1, None, None, None))   # wake it up

    def _run(self) -> None:
        while not self._stop.is_set():
            priority, seq, key, path, max_width = self._queue.get()
            _ = (priority, seq)
            if key is None:
                continue
            self._allowed.wait()
            if self._stop.is_set():
                break
            try:
                array = load_rgba(path, max_width=max_width or 0)
            finally:
                with self._lock:
                    self._pending.discard(key)
            self._ready.put((key, array))
