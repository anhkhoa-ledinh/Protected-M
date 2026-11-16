r"""
Scratch (khong commit): kiem tra truc quan nut "Random from album" +
ba duong (khong cau hinh / rong / khong mang / thanh cong) cho
TASK_shared_album_1111 Phase 3.

    python scripts/_shared_album_1111_phase3_shots.py
"""
from __future__ import annotations

import io
import json
import sys
import time
import urllib.error
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

OUT = ROOT / "results" / "shared_album_1111"

from PIL import Image  # noqa: E402

from src.config import merge_user_config  # noqa: E402
from src.content import shared_album  # noqa: E402
from src.engine.app import HcpdApplication as AppClass  # noqa: E402
from src.ui_album.options import AlbumOptions  # noqa: E402
from src.ui_album.shots import save_framebuffer  # noqa: E402
from src.ui_album import v3_flow as F  # noqa: E402

DEFAULT_CONFIG = ROOT / "assets" / "config" / "default.json"
USER_CONFIG = ROOT / "assets" / "config" / "user.json"


class _FakeResponse:
    def __init__(self, payload: bytes) -> None:
        self._payload, self._pos = payload, 0

    def read(self, n=None):
        if n is None:
            chunk, self._pos = self._payload[self._pos:], len(self._payload)
            return chunk
        chunk = self._payload[self._pos:self._pos + n]
        self._pos += len(chunk)
        return chunk

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return None


def _jpeg_bytes():
    buffer = io.BytesIO()
    Image.new("RGB", (200, 150), "orange").save(buffer, "JPEG")
    return buffer.getvalue()


def save(app, name: str) -> None:
    app._window.restore_and_focus()
    app._window.pump_events()
    width, height = app._window.framebuffer_size()
    save_framebuffer(OUT / f"p3_{name}.png", width, height)
    print(f"SAVED p3_{name}.png {width}x{height}")


def driver(app):
    flow = app._album.state
    while app._renderer.waiting_for_depth:
        yield
    flow.player = "anhkhoa"
    flow.photo = app._content.resolved_media_path(app._root)
    flow.go(F.PHOTOS)
    for _ in range(3):
        yield
    print("label(no config):", flow.message or "(none)")
    save(app, "01_button_not_configured")

    shared_album.load_config = lambda path=None: {
        "cloud_name": "demo", "upload_preset": "", "tag": "shared"}
    for _ in range(3):
        yield
    save(app, "02_button_configured_no_preset")

    # Rong: list_remote() tra ve [].
    shared_album.urllib.request.urlopen = lambda *a, **k: _FakeResponse(
        json.dumps({"resources": []}).encode("utf-8"))
    flow.message = ""; app._album_action(app._album.activate("btn_Random from album"), time.perf_counter())
    for i in range(60):
        yield
        if flow.message:
            break
    for _ in range(2):   # 1 khung nua de VE khung da co toast (khong chi dat state)
        yield
    print("empty toast:", flow.message)
    save(app, "03_empty_album_toast")

    # Khong mang.
    def boom(*a, **k):
        raise urllib.error.URLError("no route")

    shared_album.urllib.request.urlopen = boom
    flow.message = ""; app._album_action(app._album.activate("btn_Random from album"), time.perf_counter())
    for i in range(60):
        yield
        if flow.message:
            break
    for _ in range(2):
        yield
    print("no-network toast:", flow.message)
    save(app, "04_no_connection_toast")

    # Thanh cong: list roi tai that.
    resources = [{"public_id": "x", "version": "1", "format": "jpg"}]
    calls = {"n": 0}

    def fake_urlopen(url, timeout=None, context=None):
        calls["n"] += 1
        if calls["n"] % 2 == 1:
            return _FakeResponse(json.dumps({"resources": resources}).encode("utf-8"))
        return _FakeResponse(_jpeg_bytes())

    shared_album.urllib.request.urlopen = fake_urlopen
    before_photo = flow.photo
    flow.message = ""; app._album_action(app._album.activate("btn_Random from album"), time.perf_counter())
    for i in range(90):
        yield
        if flow.photo != before_photo or flow.message:
            break
    print("photo changed:", flow.photo != before_photo, "screen:", flow.screen,
          "message:", flow.message)
    save(app, "05_random_success")

    app._window.request_close()


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    config = merge_user_config(DEFAULT_CONFIG, USER_CONFIG)
    app = AppClass(
        config, ROOT, demo=True,
        album=AlbumOptions(enabled=True, guest=False, shots=False,
                           seed_leaderboard=False, ui="v3", ui_shots=False),
    )
    gen = {"it": None}

    def hook(now: float) -> None:
        if gen["it"] is None:
            gen["it"] = driver(app)
        try:
            next(gen["it"])
        except StopIteration:
            pass

    app._log_album_fps = hook  # type: ignore[method-assign]
    return app.run()


if __name__ == "__main__":
    sys.exit(main())
