"""Black-box security checks against a RUNNING central server over TLS
(TASK_deploy_1125 G9): roles table, ticket replay, concurrent spending,
concurrent finishing. Same cases as tests/test_server_{skeleton,economy}.py
but through the wire, so it can be pointed at a container.

    python tools/server_remote_checks.py --port 47870 --cert cert.pem

Start the server with the per-IP/rate limits raised; sign-ups are paced to
respect the server's own heavy-request budget. Creates accounts "chk*"."""

from __future__ import annotations

import argparse
import json
import socket
import ssl
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.game.pvp import ITEM_WEIGHT, item_price          # noqa: E402
from src.net.protocol import LineReader, encode            # noqa: E402
from src.server import permissions                         # noqa: E402

PW = "correct horse 9"
SCAN = "Thermal scan"


class Conn:
    def __init__(self, port, cert, host="127.0.0.1"):
        ctx = ssl.create_default_context(cafile=str(cert))
        ctx.check_hostname = False
        self.sock = ctx.wrap_socket(socket.create_connection((host, port), 8),
                                    server_hostname=host)
        self.sock.settimeout(10)
        self.reader = LineReader(1 << 20)
        self.call({"t": "hello", "proto": 1})

    def call(self, msg):
        self.sock.sendall(encode(msg))
        while True:
            got = self.reader.feed(self.sock.recv(65536) or b"")
            if got:
                return got[0]

    def close(self):
        self.sock.close()


def code(reply):
    return reply.get("code", "ok") if reply.get("t") != "ok" else "ok"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, required=True)
    ap.add_argument("--cert", type=Path, required=True)
    ap.add_argument("--out", type=Path)
    a = ap.parse_args()
    new = lambda: Conn(a.port, a.cert)                      # noqa: E731
    results: dict[str, object] = {}

    def signup(name):
        time.sleep(1.2)                                     # heavy budget
        c = new()
        r = c.call({"t": "sign_up", "username": name, "password": PW})
        assert r["t"] == "ok", r
        return c, r["token"]

    # 1. roles: every row above anonymous is refused to an anonymous caller
    bad = {}
    anon = new()
    for mtype, need in sorted(permissions.PERMISSIONS.items()):
        if need == permissions.ANONYMOUS:
            continue
        r = anon.call({"t": mtype})
        if r.get("code") != "forbidden":
            bad[mtype] = r
    # admin / unknown rows are never served: the server closes on them
    closed = []
    for mtype in ("admin", "list_accounts", "delete_account", "unlock", "import_accounts"):
        c = new()
        c.sock.sendall(encode({"t": mtype}))
        try:
            data = c.sock.recv(4096)
        except OSError:
            data = b""
        closed.append((mtype, b'"ok"' not in data))
        c.close()
    results["roles"] = {"rows_checked": sum(1 for n in permissions.PERMISSIONS.values()
                                            if n != permissions.ANONYMOUS),
                        "not_forbidden": bad,
                        "admin_like_types_served": [m for m, ok in closed if not ok]}

    # 2. ticket replay and another account's ticket
    c1, tok1 = signup("chkone")
    c2, _ = signup("chktwo")
    t = c1.call({"t": "start_match", "level": 1, "items": []})["ticket"]
    other = c2.call({"t": "finish_match", "ticket": t, "won": False})
    first = c1.call({"t": "finish_match", "ticket": t, "won": False})
    replay = c1.call({"t": "finish_match", "ticket": t, "won": False})
    t2 = c1.call({"t": "start_match", "level": 1, "items": []})["ticket"]
    made_up = c1.call({"t": "finish_match", "ticket": "0" * 32, "won": True})
    owner_ok = c1.call({"t": "finish_match", "ticket": t2, "won": False})
    results["tickets"] = {"other_account": code(other), "first": code(first),
                          "replay": code(replay), "made_up": code(made_up),
                          "owner_still_ok": code(owner_ok)}

    # 3. concurrent spending: wallet 100, item 80 -> exactly one buys
    _, tok3 = signup("chkspend")
    conns = []
    for _ in range(16):
        c = new()
        assert c.call({"t": "resume", "token": tok3})["t"] == "ok"
        conns.append(c)
    outcomes, lock, barrier = [], threading.Lock(), threading.Barrier(16)

    def spend(c):
        barrier.wait()
        r = c.call({"t": "start_match", "level": 1, "items": [SCAN]})
        with lock:
            outcomes.append(code(r))

    ths = [threading.Thread(target=spend, args=(c,)) for c in conns]
    [t_.start() for t_ in ths]
    [t_.join() for t_ in ths]
    wallet = conns[0].call({"t": "me"})["wallet"]
    results["spending"] = {"ok": outcomes.count("ok"), "no_funds": outcomes.count("no_funds"),
                           "other": [o for o in outcomes if o not in ("ok", "no_funds")],
                           "wallet_after": wallet, "price": item_price(SCAN)}

    # 4. one ticket finished by 12 connections at once pays once
    _, tok4 = signup("chkpay")
    conns = []
    for _ in range(12):
        c = new()
        assert c.call({"t": "resume", "token": tok4})["t"] == "ok"
        conns.append(c)
    t4 = conns[0].call({"t": "start_match", "level": 1, "items": []})["ticket"]
    time.sleep(9)                                            # slower than the human minimum
    outcomes, barrier = [], threading.Barrier(12)

    def finish(c):
        barrier.wait()
        r = c.call({"t": "finish_match", "ticket": t4, "won": True})
        with lock:
            outcomes.append(code(r))

    ths = [threading.Thread(target=finish, args=(c,)) for c in conns]
    [t_.start() for t_ in ths]
    [t_.join() for t_ in ths]
    me = conns[0].call({"t": "me"})
    results["finishing"] = {"ok": outcomes.count("ok"), "bad_ticket": outcomes.count("bad_ticket"),
                            "other": [o for o in outcomes if o not in ("ok", "bad_ticket")],
                            "earned_total": me["earned_total"]}

    r = results
    ok = (not r["roles"]["not_forbidden"] and not r["roles"]["admin_like_types_served"]
          and r["tickets"] == {"other_account": "bad_ticket", "first": "ok",
                               "replay": "bad_ticket", "made_up": "bad_ticket",
                               "owner_still_ok": "ok"}
          and r["spending"]["ok"] == 1 and r["spending"]["no_funds"] == 15
          and not r["spending"]["other"] and r["spending"]["wallet_after"] == 100 - item_price(SCAN) >= 0
          and r["finishing"]["ok"] == 1 and r["finishing"]["bad_ticket"] == 11
          and not r["finishing"]["other"] and r["finishing"]["earned_total"] > 0)
    r["PASS"] = ok
    print(json.dumps(r, indent=2))
    if a.out:
        a.out.write_text(json.dumps(r, indent=2))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
