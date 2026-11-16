"""
Ván vật ẩn bị dọn giữa chừng thì ván GIAO DIỆN phải đi theo.

``_sync_prop_game`` gọi ``_abandon_game`` khi hình học lớp dựng lại giữa
trận. Trước bản sửa, nó chỉ dọn ``self._game`` còn ``self._album.game`` ở
lại: người chơi đứng trên HUD với 0 con bọ — không bao giờ thắng được — và
cú bấm chuột kế tiếp đọc ``self._game.props`` trên ``None``, giết cả vòng
vẽ. Phiên chơi thật 07/11 chết đúng kiểu đó:
``AttributeError: 'NoneType' object has no attribute 'props'``.
"""

from __future__ import annotations

from src.engine.app import HcpdApplication
from src.ui_album import v3_flow as F
from src.ui_album.options import AlbumOptions


class _Renderer:
    def __init__(self) -> None:
        self.toasts: list = []

    def set_props(self, props) -> None:
        self._props = props

    def show_toast(self, text, is_error=False) -> None:
        self.toasts.append(text)


class _UI:
    """Bề mặt tối thiểu mà ``_abandon_ui_match`` đụng tới."""

    flow = F

    def __init__(self, screen: str) -> None:
        self.state = F.V3Flow()
        self.state.screen = screen
        self.state.weapon = 0
        self.game = object()


def make_app(screen: str, *, v3: bool = True) -> HcpdApplication:
    app = HcpdApplication.__new__(HcpdApplication)
    app._album_options = AlbumOptions(enabled=True, ui="v3" if v3 else "album")
    app._album = _UI(screen)
    app._renderer = _Renderer()
    app._game = None
    app._game_planes = ()
    app._watch_until_s = None
    return app


def test_match_is_cleared_when_props_are_abandoned():
    """Không dọn ván giao diện thì cú bấm sau đó sập app."""
    app = make_app(F.PLAY)
    app._abandon_game(0.0, "geometry")
    assert app._album.game is None


def test_player_is_moved_off_a_hud_that_cannot_be_won():
    """Đứng lại trên HUD 0 bọ là bế tắc — đưa về màn trang bị."""
    app = make_app(F.PLAY)
    app._abandon_game(0.0, "geometry")
    assert app._album.state.screen == F.LOADOUT


def test_v3_message_does_not_mention_the_removed_G_key():
    """Giao diện v3 đã bỏ phím G; đừng bảo người chơi bấm nó."""
    app = make_app(F.PLAY)
    app._abandon_game(0.0, "geometry")
    said = app._album.state.message
    assert said
    assert "G" not in said.split()


def test_other_screens_are_not_yanked_around():
    """Đang ở màn kết quả thì dọn ván ngầm, không giật người chơi đi đâu cả."""
    app = make_app(F.WIN)
    app._abandon_game(0.0, "new_image")
    assert app._album.game is None
    assert app._album.state.screen == F.WIN


def test_no_ui_match_is_harmless():
    """Không có ván giao diện thì hàm không được nổ."""
    app = make_app(F.PLAY)
    app._album.game = None
    app._abandon_game(0.0, "exit")
    assert app._album.state.screen == F.PLAY


def test_shoot_is_safe_when_the_prop_game_is_gone():
    """``_album_shoot`` không được đọc ``props`` trên ``None``."""
    app = make_app(F.PLAY)
    app._album.game = object()          # ván giao diện con song
    app._game = None                    # ...ma van vat an da bi don
    assert app._album_shoot(0.0) is True   # nuot cu bam, khong nem loi
