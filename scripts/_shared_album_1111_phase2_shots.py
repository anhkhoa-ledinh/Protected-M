r"""
Scratch (khong commit): kiem tra truc quan nut "Share to album" + hop xac
nhan + toast ket qua cho TASK_shared_album_1111 Phase 2.

    python scripts/_shared_album_1111_phase2_shots.py
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

OUT = ROOT / "results" / "shared_album_1111"

from src.config import merge_user_config  # noqa: E402
from src.content import shared_album  # noqa: E402
from src.engine.app import HcpdApplication as AppClass  # noqa: E402
from src.ui_album.options import AlbumOptions  # noqa: E402
from src.ui_album.shots import save_framebuffer  # noqa: E402
from src.ui_album import v3_flow as F  # noqa: E402

DEFAULT_CONFIG = ROOT / "assets" / "config" / "default.json"
USER_CONFIG = ROOT / "assets" / "config" / "user.json"


def save(app, name: str) -> None:
    app._window.restore_and_focus()
    app._window.pump_events()
    width, height = app._window.framebuffer_size()
    save_framebuffer(OUT / f"p2_{name}.png", width, height)
    print(f"SAVED p2_{name}.png {width}x{height}")


PINNED = {"xy": None}


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

    # Bam nut (khong co config that): phai khong lam gi (khong mo hop thoai).
    app._album.activate("btn_Share to album")
    for _ in range(3):
        yield
    print("after click, pending:", repr(flow.shared_album_pending))
    save(app, "02_click_disabled_noop")

    # Gia lap co config (khong dung mang that) de kiem duong con lai.
    shared_album.load_config = lambda path=None: {
        "cloud_name": "demo", "upload_preset": "demo_preset", "tag": "shared"}
    app._album.activate("btn_Share to album")
    for _ in range(3):
        yield
    print("dialog open, pending:", repr(flow.shared_album_pending))
    save(app, "03_confirm_dialog")

    # Di chuot vao nut Yes de chup luon vong hover.
    PINNED["xy"] = (820.0, 444.0)
    for _ in range(5):
        yield
    print("mouse:", app._album.mouse, "hover:", app._album.hover())
    save(app, "04_confirm_dialog_hover_yes")
    PINNED["xy"] = None

    app._album_action(app._album.activate("btn_confirm_yes"), time.perf_counter())
    print("worker busy right after activate:", app._share_worker.busy)
    for i in range(120):          # cho worker that bai (khong URL that) + toast
        yield
        if i % 20 == 0:
            print(f"  t={i} busy={app._share_worker.busy} msg={flow.message!r}")
        if flow.message:
            break
    print("toast:", flow.message)
    save(app, "05_toast_after_confirm")

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
        if PINNED["xy"] is not None:
            app._album.mouse = PINNED["xy"]

    app._log_album_fps = hook  # type: ignore[method-assign]
    return app.run()


if __name__ == "__main__":
    sys.exit(main())
