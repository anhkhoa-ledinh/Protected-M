#!/usr/bin/env python3
"""
HCPD — Head-Coupled Perspective Display.

The only visible window is an OpenGL virtual room.
The webcam is internal only and is NEVER displayed.
"""

from __future__ import annotations

import argparse
import logging
import sys
import threading
from pathlib import Path

from src.config import merge_user_config
from src.engine import HcpdApplication
from src.gui import run_calibration_wizard
from src.ui_album.options import AlbumOptions

PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_CONFIG = PROJECT_ROOT / "assets" / "config" / "default.json"
USER_CONFIG = PROJECT_ROOT / "assets" / "config" / "user.json"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Head-Coupled Perspective Display — virtual room window",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG,
        help="Path to JSON/YAML configuration file",
    )
    parser.add_argument(
        "--calibrate",
        action="store_true",
        help="Launch screen/webcam calibration wizard (separate tool)",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable debug logging",
    )
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Demo mode (PB-055): content.demo.json, no config writes, "
             "only G/T/F6/drag-drop/F4/Esc",
    )
    # Giao diện "album bọc nhựa" (TASK_album_ui_1106). Không có --album thì
    # app chạy **y hệt hôm nay**: mọi cờ dưới đây chỉ có nghĩa khi --album bật.
    parser.add_argument(
        "--album",
        action="store_true",
        help="Giao dien album Protected M (assets/ui_album)",
    )
    parser.add_argument(
        "--album-guest",
        action="store_true",
        help="Bo qua man dang nhap, vao thang bia album",
    )
    parser.add_argument(
        "--album-shots",
        action="store_true",
        help="Di qua moi man o ca hai theme, luu mot PNG moi man roi thoat",
    )
    parser.add_argument(
        "--album-seed-leaderboard",
        action="store_true",
        help="Them vai dong mau (danh dau 'mau') vao bang xep hang",
    )
    # TASK_ui_v3_1107: bo giao dien v3 ("v2 · Redesign") la bo duy nhat
    # con lai (TASK_arch_refactor_1120 R1 da xoa bo cu "album").
    parser.add_argument(
        "--ui",
        choices=("v3",),
        default="v3",
        help="Bo giao dien khi --album bat (chi co v3)",
    )
    parser.add_argument(
        "--ui-shots",
        action="store_true",
        help="Di qua 23 man v3, luu mot PNG moi man roi thoat",
    )
    # TASK_pvp_lan_1116 P2: khong mo webcam — dau theo dang mo phong ngay tu
    # dau. De chay HAI tien trinh game tren CUNG mot may khi kiem PvP: chi
    # mot tien trinh mo duoc webcam that, tien trinh con lai phai dung co nay.
    parser.add_argument(
        "--no-webcam",
        action="store_true",
        help="Khong mo webcam that, dung dau mo phong "
             "(chay hai tien trinh PvP tren cung mot may)",
    )
    args = parser.parse_args(argv)
    if args.ui_shots and args.ui != "v3":
        parser.error("--ui-shots chi co nghia voi --ui v3 (bo cu dung --album-shots)")
    if args.album_shots and args.ui != "album":
        parser.error("--album-shots chi co nghia voi --ui album")
    if args.demo and args.calibrate:
        # Wizard hiệu chuẩn ghi user.json; demo không được ghi file config nào.
        parser.error("--demo khong dung chung voi --calibrate")
    if args.calibrate and args.album:
        parser.error("--album khong dung chung voi --calibrate")
    return args


def main(argv: list[str] | None = None) -> int:
    """Application entry point."""
    args = parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        # TASK_hud_fix_1119 Fix 4: ms-precision timestamps -- without them
        # a real double-click and an actual duplicate-fire bug look
        # identical in the log (both just two "Ban ..." lines in a row).
        format="%(asctime)s.%(msecs)03d | %(levelname)s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )
    # TASK_pvp_lan_1116: no free text slot on the Settings screen for this
    # (checked both texts keys — webcam_name/volume_value, both already
    # bound), so log it at startup per the task's own fallback.
    from src import VERSION

    logging.getLogger(__name__).info("Protected M v%s", VERSION)

    def _thread_excepthook(args: threading.ExceptHookArgs) -> None:
        logging.getLogger("thread").error(
            "Unhandled exception in thread %s",
            args.thread.name if args.thread else "?",
            exc_info=(args.exc_type, args.exc_value, args.exc_traceback),
        )

    threading.excepthook = _thread_excepthook

    config = merge_user_config(args.config, USER_CONFIG)

    if args.calibrate:
        run_calibration_wizard(config, USER_CONFIG)
        return 0

    app = HcpdApplication(
        config, PROJECT_ROOT, demo=args.demo, no_webcam=args.no_webcam,
        album=AlbumOptions(enabled=args.album, guest=args.album_guest,
                           shots=args.album_shots,
                           seed_leaderboard=args.album_seed_leaderboard,
                           ui=args.ui, ui_shots=args.ui_shots),
    )
    try:
        return app.run()
    except Exception:
        logging.getLogger(__name__).exception("Application terminated with an error")
        return 1


if __name__ == "__main__":
    sys.exit(main())
