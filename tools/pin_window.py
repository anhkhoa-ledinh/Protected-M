#!/usr/bin/env python3
"""Dev tool: float the game window at an exact size and position on Hyprland.

    python tools/pin_window.py [--size 1091x613] [--pos 107,100] [--wait 60]

Start it BEFORE (or while) the game launches: it installs a window rule
(Hyprland 0.57 Lua, via ``hyprctl eval``) for the game's window class, then
waits until ``hyprctl clients -j`` reports exactly the asked geometry and
prints it. If the window already exists it is moved and resized with a
dispatcher. Exit 0 = geometry reached, 1 = timeout.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time

CLASS = "HCPD — Virtual Room"


def hypr(*args: str) -> str:
    return subprocess.run(["hyprctl", *args], capture_output=True, text=True,
                          check=False).stdout


def find_client() -> dict | None:
    try:
        clients = json.loads(hypr("clients", "-j"))
    except json.JSONDecodeError:
        return None
    for client in clients:
        if client.get("class") == CLASS:
            return client
    return None


def install_rule(w: int, h: int, x: int, y: int) -> None:
    lua = ('hl.window_rule({ match = { class = "^(%s)$" }, float = true, '
           'size = { %d, %d }, move = { %d, %d } })' % (CLASS, w, h, x, y))
    out = hypr("eval", lua).strip()
    if out and out != "ok":
        print(f"hyprctl eval: {out}", file=sys.stderr)


def nudge(client: dict, w: int, h: int, x: int, y: int) -> None:
    addr = f"address:{client['address']}"
    if not client.get("floating"):
        hypr("dispatch", "setfloating", addr)
    hypr("dispatch", "resizewindowpixel", f"exact {w} {h},{addr}")
    hypr("dispatch", "movewindowpixel", f"exact {x} {y},{addr}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--size", default="1091x613")
    parser.add_argument("--pos", default="107,100")
    parser.add_argument("--wait", type=float, default=60.0)
    args = parser.parse_args()
    w, h = (int(v) for v in args.size.lower().split("x"))
    x, y = (int(v) for v in args.pos.split(","))
    install_rule(w, h, x, y)
    deadline, last_nudge = time.time() + args.wait, 0.0
    while time.time() < deadline:
        client = find_client()
        if client is not None:
            at, size = client["at"], client["size"]
            if at == [x, y] and size == [w, h] and client.get("floating"):
                print(json.dumps({"at": at, "size": size, "floating": True}))
                return 0
            if time.time() - last_nudge > 1.0:
                nudge(client, w, h, x, y)
                last_nudge = time.time()
        time.sleep(0.2)
    print(f"timeout; last client: {find_client()}", file=sys.stderr)
    return 1


if __name__ == "__main__":
    sys.exit(main())
