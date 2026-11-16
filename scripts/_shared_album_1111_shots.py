r"""
Scratch (khong commit): chup truoc/sau cho TASK_shared_album_1111 Phase 1.

    python scripts/_shared_album_1111_shots.py

Lai mot HcpdApplication thuc (--demo --album --ui v3), ghep `_log_album_fps`
(goi moi khung, khong anh huong gi khac) thanh mot may trang thai tu drive
flow truc tiep (dung cach v3_live.py da lam: gan field cua V3Flow roi
`flow.go(...)`), vi ba dau hieu can chup deu la TRANG THAI SONG (focus/hover/
mot van con du lieu) ma `--ui-shots` khong ve (README: "no hover ring, no
focus ring" trong che do chup).

Dau (c): bat mot van THAT bang HCPD_ALBUM_AUTOMATCH=1 (moc do da co san cho
dung viec "tu kiem bang mat"), roi goi `_album_action("logout", ...)` -
DUNG duong san xuat cua nut "Sign out" - va chup man sign-in ngay sau do.
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

os.environ["HCPD_ALBUM_AUTOMATCH"] = "1"

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

OUT = ROOT / "results" / "shared_album_1111"
TAG = sys.argv[1] if len(sys.argv) > 1 else "shot"

from src.config import merge_user_config  # noqa: E402
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
    path = OUT / f"{TAG}_{name}.png"
    save_framebuffer(path, width, height)
    print(f"SAVED {path} {width}x{height}")


def driver(app):
    """Mot generator: moi `next()` la mot buoc, cho den khi can cho khung sau."""
    flow = app._album.state  # V3Flow

    # 0) cho tach lop xong (demo tu nap anh mac dinh luc khoi dong)
    while app._renderer.waiting_for_depth:
        yield

    # 1) nhay thang vao PLAY (nhu v3_live.apply) de moc AUTOMATCH mo van thuc
    flow.player = "anhkhoa"
    flow.weapon = 0
    flow.items = ["Antidote", "Flare"]
    flow.level = 1
    flow.go(F.PLAY)
    for _ in range(10):
        yield
        if app._album.game is not None:
            break
    game = app._game
    print("GAME:", None if game is None else f"{len(game.props)} bug(s)")

    # 2) dang xuat bang DUNG duong san xuat cua nut "Sign out"
    app._album_action("logout", time.perf_counter())
    for _ in range(3):
        yield
    print("AFTER LOGOUT: screen=", flow.screen,
          "self._game=", None if app._game is None else len(app._game.props))
    save(app, "c_sign_in_after_logout")

    # 3) (a) vong focus cua o nhap tren man dang nhap
    flow.go(F.SIGN_IN)
    flow.focus = "field_user"
    for _ in range(3):
        yield
    save(app, "a_sign_in_focus_field_user")
    flow.focus = None

    # 4) (b) vong hover cua the cap do
    flow.go(F.LEVELS)
    app._album.mouse = (932.0, 192.0)  # tam level_row_1 [648,150,568,84]
    for _ in range(3):
        yield
    save(app, "b_level_select_hover_row1")

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
