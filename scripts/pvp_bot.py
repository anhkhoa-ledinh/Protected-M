r"""
TASK_pvp_lan_1116 — headless PvP test client. No GL, no webcam, no renderer.

Joins a queue or a room code, plays one match (or several, with
``--matches``) by claiming kills on a fixed timer, and logs round-trip time
for every kill plus the final score both sides reported. Dev tool only —
the bug positions it sends when it must place them (host in Competitive,
hider in Hunt & Hide once the setup window starts) are synthetic, there is
no real photo or layer geometry behind them.

    .\.venv\Scripts\python.exe scripts\pvp_bot.py --mode Competitive --level 1
    .\.venv\Scripts\python.exe scripts\pvp_bot.py --mode "Hunt & Hide" --room 1234
"""

from __future__ import annotations

import argparse
import hashlib
import io
import logging
import random
import statistics
import sys
import time
from base64 import b64encode
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.net import client as pvp_client  # noqa: E402
from src.net import protocol as proto  # noqa: E402
from src.game.pvp import LEVEL_TABLE, PVP_MODES  # noqa: E402

logger = logging.getLogger("pvp_bot")

COMPETITIVE, HUNT_HIDE = PVP_MODES
#: How long to wait for one expected message before giving up.
WAIT_S = 10.0
#: Hunt & Hide setup window the server told everyone about (task: 20 s), and
#: for Competitive, the time for the *other* side (a real game, not this
#: bot) to bake a never-before-seen photo — depth inference + clustering +
#: disocclusion fill, which the README documents can take "a minute or
#: more" on a slow machine, and includes a one-time model download/load the
#: first time a process bakes anything at all. 25 s was measured to be too
#: short and stalled real matches during G2 measurement (TASK_pvp_lan_1116);
#: this is deliberately generous rather than racing the bake.
SETUP_WAIT_S = 90.0


def _tiny_photo() -> tuple[str, str]:
    """A throwaway valid JPEG — just enough to exercise the wire format.
    ``(jpeg_b64, sha1)``, same shape as ``shared_album._reencode`` output.

    FIXED content, not a fresh random image each call: a different sha1
    every time this bot hosts means the receiver's bake cache
    (``assets/cache/maps/<sha1>/``) never hits, forcing a full ~tens-of-
    seconds depth bake on EVERY match the bot hosts — easily outlasting
    ``SETUP_WAIT_S`` and stalling the match (observed during G2 measurement,
    TASK_pvp_lan_1116). A real host resending the same photo on a rematch
    would hit the cache; this should too.

    A real gradient, not a flat colour: a flat-colour image has no depth
    cues at all, so if a game ever ends up *hosting* with this photo (it
    can receive it as a guest in one match and host the next, since
    there's no "pick a PvP photo" screen yet — see README's Known issues),
    bug placement finds zero valid spots and the match correctly, but
    uselessly, aborts. A gradient at least gives the depth model something
    to split into layers.
    """
    from PIL import Image

    size = 192
    image = Image.new("RGB", (size, size))
    pixels = image.load()
    for y in range(size):
        for x in range(size):
            pixels[x, y] = (40 + x, 60 + y // 2, 90)
    buf = io.BytesIO()
    image.save(buf, "JPEG", quality=85)
    data = buf.getvalue()
    return b64encode(data).decode("ascii"), hashlib.sha1(data).hexdigest()


def _synthetic_bugs(level: int) -> list[dict]:
    """Valid-but-fake bug placements — this bot has no layer geometry."""
    count = LEVEL_TABLE[level][1]
    return [{
        "id": i,
        "x": round(random.uniform(0.15, 0.85), 4),
        "y": round(random.uniform(0.15, 0.85), 4),
        "z": round(random.uniform(0.0, 1.0), 4),
        "layer": random.randint(0, 2),
        "size_k": round(random.uniform(0.6, 1.8), 2),
        "camo_rgb": [round(random.random(), 3) for _ in range(3)],
    } for i in range(count)]


def _wait_for(client: pvp_client.PvpClient, types: set[str], *,
              timeout_s: float = WAIT_S) -> dict | None:
    """Poll until a message whose ``t"`` is in ``types`` arrives (or an
    ``error``/disconnect status, returned immediately so the caller can
    react). ``None`` on timeout."""
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        for msg in client.poll():
            t = msg.get("t")
            if t == proto.ERROR or (t == pvp_client.STATUS
                                    and msg.get("status") != "connected"):
                return msg
            if t in types:
                return msg
        time.sleep(0.02)
    return None


class _Match:
    """One match's local bookkeeping: who's alive, kill RTT, final result."""

    def __init__(self, name: str, role: str, mode: str) -> None:
        self.name = name
        self.role = role
        self.mode = mode
        self.alive: set[int] = set()
        self.sent_at: dict[int, float] = {}
        self.rtts_ms: list[float] = []
        self.kills_mine = 0
        self.time_left = 0.0
        self.result: dict | None = None

    def can_shoot(self) -> bool:
        return self.mode == COMPETITIVE or self.role == "hunter"


def _play_one_match(client: pvp_client.PvpClient, name: str, mode: str,
                     level: int, kill_every_s: float) -> dict | None:
    matched = _wait_for(client, {proto.MATCHED})
    if matched is None or matched.get("t") != proto.MATCHED:
        logger.error("Khong vao duoc tran: %s", matched)
        return None
    role = matched["role"]
    logger.info("Vao tran: room=%s role=%s doi thu=%s", matched["room"], role,
               matched["opponent_name"])
    match = _Match(name, role, mode)

    is_photo_host = role in ("host", "hunter")   # true on any fresh match
    if is_photo_host:
        jpeg_b64, sha1 = _tiny_photo()
        client.send({"t": proto.PHOTO, "jpeg_b64": jpeg_b64, "sha1": sha1})
    if mode == COMPETITIVE and role == "host":
        client.send({"t": proto.BUGS, "bugs": _synthetic_bugs(level)})
    client.send({"t": proto.READY})

    if mode == HUNT_HIDE:
        # The server sends an empty-bugs "setup" start to BOTH players
        # (hunter included — it only shows "bugs are hiding…"); only the
        # hider has to answer it with placements.
        setup = _wait_for(client, {proto.START}, timeout_s=SETUP_WAIT_S)
        if setup is None or setup.get("t") != proto.START:
            logger.error("Khong thay giai doan dat bo: %s", setup)
            return None
        if role == "hider":
            client.send({"t": proto.BUGS, "bugs": _synthetic_bugs(level)})

    start = _wait_for(client, {proto.START}, timeout_s=SETUP_WAIT_S)
    if start is None or start.get("t") != proto.START or not start.get("bugs"):
        logger.error("Khong vao duoc pha san: %s", start)
        return None
    match.alive = {int(b["id"]) for b in start["bugs"]}
    logger.info("Bat dau: %d bo, %.0f s", len(match.alive), start["duration_s"])

    next_shot = time.monotonic()
    while True:
        for msg in client.poll():
            t = msg.get("t")
            if t == proto.KILLED:
                bug_id = msg["bug"]
                match.alive.discard(bug_id)
                sent = match.sent_at.pop(bug_id, None)
                if sent is not None and not msg.get("late") and msg.get("by") == name:
                    rtt_ms = (time.monotonic() - sent) * 1000.0
                    match.rtts_ms.append(rtt_ms)
                    match.kills_mine += 1
                    logger.info("Ha bo #%d — RTT %.1f ms", bug_id, rtt_ms)
            elif t == proto.TICK:
                match.time_left = msg["time_left"]
            elif t == proto.END:
                match.result = msg
            elif t == proto.ERROR:
                logger.error("Loi tu server: %s", msg.get("message"))
            elif t == pvp_client.STATUS and msg.get("status") != "connected":
                logger.error("Mat ket noi: %s", msg.get("message"))
                return _summarize(match)
        if match.result is not None:
            return _summarize(match)
        now = time.monotonic()
        if match.can_shoot() and match.alive and now >= next_shot:
            bug_id = min(match.alive)
            client.send({"t": proto.HIT, "bug": bug_id})
            match.sent_at[bug_id] = now
            next_shot = now + kill_every_s
        time.sleep(0.02)


def _summarize(match: _Match) -> dict:
    scores = (match.result or {}).get("scores", {})
    rtt_p95 = (statistics.quantiles(match.rtts_ms, n=20)[18]
              if len(match.rtts_ms) >= 2 else (match.rtts_ms[0] if match.rtts_ms else 0.0))
    summary = {
        "role": match.role, "winner": (match.result or {}).get("winner"),
        "you_score": scores.get("you"), "opp_score": scores.get("opp"),
        "opp_name": (match.result or {}).get("opp_name"),
        "kills_confirmed": match.kills_mine, "kill_rtt_ms": list(match.rtts_ms),
        "kill_rtt_p95_ms": rtt_p95,
    }
    logger.info("KET QUA: %s", summary)
    return summary


def run(args: argparse.Namespace) -> int:
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(levelname)s | %(name)s | %(message)s")
    if args.host:
        host, port = args.host, args.port
    else:
        found = pvp_client.resolve_server(ROOT)
        if found is None:
            logger.error("Khong tim duoc server (pvp.json / UDP discovery)")
            return 1
        host, port = found
    client = pvp_client.PvpClient(args.name)
    client.connect(host, port)
    status = _wait_for(client, {pvp_client.STATUS}, timeout_s=WAIT_S)
    if status is None or status.get("status") != "connected":
        logger.error("Khong ket noi duoc %s:%d — %s", host, port, status)
        return 1
    logger.info("Da ket noi %s:%d", host, port)

    results = []
    for i in range(args.matches):
        if args.room and i == 0:
            client.send({"t": proto.JOIN_ROOM, "code": args.room})
        else:
            client.send({"t": proto.QUEUE, "mode": args.mode, "level": args.level})
        result = _play_one_match(client, args.name, args.mode, args.level,
                                 args.kill_every)
        if result is None:
            client.close()
            return 1
        results.append(result)
    client.close()
    wins = sum(1 for r in results if r["winner"] == "you")
    logger.info("TOM TAT %d tran: %d thang. RTT p95 moi tran: %s", len(results),
               wins, [round(r["kill_rtt_p95_ms"], 1) for r in results])
    return 0


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--mode", choices=PVP_MODES, default=COMPETITIVE)
    p.add_argument("--level", type=int, default=1, choices=sorted(LEVEL_TABLE))
    p.add_argument("--room", default="", help="4-digit room code to join; "
                   "default is to join the matchmaking queue")
    p.add_argument("--kill-every", type=float, default=2.0,
                  help="Seconds between claimed kills (default 2.0)")
    p.add_argument("--matches", type=int, default=1,
                  help="Play this many matches back to back (re-queues after "
                       "each end; --room only applies to the first)")
    p.add_argument("--name", default=f"Bot{random.randint(100, 999)}")
    p.add_argument("--host", default="", help="Skip discovery, connect here")
    p.add_argument("--port", type=int, default=proto.TCP_PORT)
    p.add_argument("-v", "--verbose", action="store_true")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    return run(parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
