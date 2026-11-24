"""Server-authoritative economy: tickets, the wallet, scores, the board.

The client never says how many points it earned. ``start_match`` charges the
items and hands out a one-time ticket; ``finish_match`` turns the ticket and
the **server's own clock** into a score with the shared formula
(``src.game.scoring``). Every wallet change is one SQLite transaction with a
conditional ``UPDATE``, and the table has ``CHECK (wallet >= 0)`` on top.

What it cannot know: that the match was really won (docs/server.md)."""

from __future__ import annotations

import json
import secrets
import sqlite3

from src.account.crypto import hash_token
from src.game.levels import LEVELS
from src.game.pvp import ITEM_WEIGHT, PVP_MODES, PVP_UNLOCK_LEVELS, item_price
from src.game.scoring import OVERTIME_MAX_S, match_score
from src.server.store import Store

SOLO = "solo"
MODES = (SOLO, *PVP_MODES)

#: A ticket is good for the level's time, the level-5 overtime, and this long
#: on top (loading the photo, a slow link).
TICKET_GRACE_S = 120.0
#: A "win" this soon after the start is refused: no human finds every bug
#: faster than this per bug. A modified client can still wait it out.
MIN_WIN_S_PER_BUG = 1.0
#: Lateness tolerated before a claimed win counts as a loss (levels 1-4 end
#: at the time limit in the game).
LATE_GRACE_S = 10.0
MAX_USES_PER_ITEM = 50
LEADERBOARD_ROWS = 20


class EconomyError(Exception):
    def __init__(self, code: str, msg: str = "") -> None:
        super().__init__(code)
        self.code = code
        self.msg = msg or code


def level_seconds(level: int) -> float:
    return float(LEVELS[level].seconds)


def _cleared(row: sqlite3.Row) -> set[int]:
    try:
        return {int(n) for n in json.loads(row["levels_cleared"])}
    except (ValueError, TypeError):
        return set()


def price(items: list[str]) -> int:
    return sum(item_price(name) for name in items)


def start_match(store: Store, key: str, mode: str, level: int, items: list[str],
                now: float) -> tuple[str, dict]:
    """Charge ``items`` and issue a ticket. Returns ``(nonce, info)``."""
    if mode not in MODES or level not in LEVELS:
        raise EconomyError("bad_request", "unknown mode or level")
    rules = LEVELS[level]
    if len(items) > rules.slots or len(set(items)) != len(items) \
            or any(name not in ITEM_WEIGHT for name in items):
        raise EconomyError("bad_request", "bad items")
    cost = price(items)
    nonce = secrets.token_urlsafe(16)
    expires = now + level_seconds(level) + OVERTIME_MAX_S + TICKET_GRACE_S
    with store.transaction() as conn:
        row = conn.execute("SELECT * FROM accounts WHERE key = ?", (key,)).fetchone()
        if row is None:
            raise EconomyError("forbidden", "no such account")
        if mode != SOLO and not set(PVP_UNLOCK_LEVELS) <= _cleared(row):
            raise EconomyError("locked_mode", "win levels 1-5 to unlock PvP")
        # An older ticket that was never finished is void (its items stay paid).
        conn.execute("UPDATE tickets SET used = 1 WHERE account_key = ? AND used = 0",
                     (key,))
        conn.execute("DELETE FROM tickets WHERE account_key = ? AND"
                     " (used = 1 AND expires_at <= ?)", (key, now))
        if cost:
            spent = conn.execute(
                "UPDATE accounts SET wallet = wallet - ? WHERE key = ? AND wallet >= ?",
                (cost, key, cost)).rowcount
            if spent != 1:
                raise EconomyError("no_funds", "not enough points")
        conn.execute("INSERT INTO tickets VALUES (?,?,?,?,?,?,?,?,0,0)",
                     (hash_token(nonce), key, mode, level, json.dumps(items),
                      cost, now, expires))
        wallet = conn.execute("SELECT wallet FROM accounts WHERE key = ?",
                              (key,)).fetchone()[0]
    return nonce, {"level": level, "mode": mode, "items": items, "cost": cost,
                   "wallet": wallet, "started_at": now, "expires_at": expires,
                   "seconds": rules.seconds}


def begin_play(store: Store, key: str, nonce: str, now: float) -> float:
    """Move the ticket's start to now (the photo has loaded). Only forward,
    only once, only before the match is finished."""
    with store.transaction() as conn:
        row = _live_ticket(conn, key, nonce, now)
        if row["began"]:
            raise EconomyError("bad_ticket", "play already began")
        conn.execute("UPDATE tickets SET started_at = ?, began = 1"
                     " WHERE nonce_hash = ?", (now, row["nonce_hash"]))
    return now


def _live_ticket(conn, key: str, nonce: str, now: float) -> sqlite3.Row:
    row = conn.execute(
        "SELECT * FROM tickets WHERE nonce_hash = ? AND account_key = ? AND used = 0",
        (hash_token(nonce), key)).fetchone()
    if row is None or row["expires_at"] <= now:
        raise EconomyError("bad_ticket", "unknown, used or expired ticket")
    return row


def finish_match(store: Store, key: str, nonce: str, won: bool,
                 items_used: dict, now: float) -> dict:
    """Spend the ticket and, for a won solo match, credit the wallet."""
    with store.transaction() as conn:
        row = _live_ticket(conn, key, nonce, now)
        # Spend it first: whatever follows, this ticket never works again.
        conn.execute("UPDATE tickets SET used = 1 WHERE nonce_hash = ?",
                     (row["nonce_hash"],))
        level, mode = row["level"], row["mode"]
        bought = set(json.loads(row["items"]))
        used = {}
        for name, count in items_used.items():
            if name not in bought or isinstance(count, bool) \
                    or not isinstance(count, int) \
                    or not 0 <= count <= MAX_USES_PER_ITEM:
                raise EconomyError("bad_request", "bad items_used")
            if count:
                used[name] = count
        elapsed = max(0.0, now - row["started_at"])
        seconds = level_seconds(level)
        limit = seconds + (OVERTIME_MAX_S if level == 5 else 0.0) + LATE_GRACE_S
        reason = ""
        if won and elapsed < LEVELS[level].bugs * MIN_WIN_S_PER_BUG:
            won, reason = False, "too_fast"
        elif won and elapsed > limit:
            won, reason = False, "time"
        score = match_score(level, not won, seconds - elapsed, used, ITEM_WEIGHT)
        result = {"won": bool(won), "score": score, "elapsed": round(elapsed, 3),
                  "level": level, "mode": mode}
        if reason:
            result["reason"] = reason
        if won and mode == SOLO:
            account = conn.execute("SELECT * FROM accounts WHERE key = ?",
                                   (key,)).fetchone()
            cleared = sorted(_cleared(account) | {level})
            conn.execute(
                "UPDATE accounts SET wallet = wallet + ?, earned_total = earned_total + ?,"
                " levels_cleared = ? WHERE key = ?",
                (score, score, json.dumps(cleared), key))
            conn.execute("INSERT INTO scores (account_key, level, score, seconds,"
                         " items_used, at) VALUES (?,?,?,?,?,?)",
                         (key, level, score, elapsed, json.dumps(used), now))
        account = conn.execute("SELECT wallet, earned_total, levels_cleared FROM"
                               " accounts WHERE key = ?", (key,)).fetchone()
        result.update(wallet=account["wallet"], earned_total=account["earned_total"],
                      levels_cleared=sorted(json.loads(account["levels_cleared"])))
    return result


def leaderboard(store: Store, key: str | None) -> dict:
    """Solo board: ``earned_total`` high to low, ties share a rank and sort by
    name (the same rule as ``ui_album.board.collect``)."""
    conn = store.connect()
    rows = conn.execute(
        "SELECT key, display, earned_total, levels_cleared FROM accounts"
        " ORDER BY earned_total DESC, lower(display), key LIMIT ?",
        (LEADERBOARD_ROWS,)).fetchall()
    entries, last_total, last_rank = [], None, 0
    for index, row in enumerate(rows):
        if row["earned_total"] != last_total:
            last_total, last_rank = row["earned_total"], index + 1
        try:
            best = max((int(n) for n in json.loads(row["levels_cleared"])), default=0)
        except (ValueError, TypeError):
            best = 0
        entries.append({"rank": last_rank, "name": row["display"],
                        "earned": row["earned_total"], "best_level": best})
    you = None
    me = conn.execute("SELECT earned_total FROM accounts WHERE key = ?",
                      (key,)).fetchone() if key else None
    if me is not None and me["earned_total"] > 0:
        you = 1 + conn.execute("SELECT COUNT(*) FROM accounts WHERE earned_total > ?",
                               (me["earned_total"],)).fetchone()[0]
    return {"entries": entries, "you": you}
