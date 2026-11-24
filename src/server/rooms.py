"""PvP rooms on the central server: who is in which room, and a push to the
other player. In memory only, thread-safe, bounded (docs/server.md)."""

from __future__ import annotations

import secrets
import threading

_ALPHABET = "23456789ABCDEFGHJKMNPQRSTVWXYZ"
CODE_LEN = 6


class RoomError(Exception):
    def __init__(self, code: str, msg: str = "") -> None:
        super().__init__(code)
        self.code, self.msg = code, msg or code


class Rooms:
    def __init__(self, max_rooms: int = 50) -> None:
        self.max_rooms = max_rooms
        self._lock = threading.Lock()
        self._rooms: dict[str, dict] = {}
        self._where: dict[str, str] = {}          # account key -> room code

    def count(self) -> int:
        with self._lock:
            return len(self._rooms)

    def players(self) -> int:
        with self._lock:
            return len(self._where)

    def create(self, state, mode: str) -> str:
        with self._lock:
            if state.account in self._where:
                raise RoomError("in_room", "leave your room first")
            if len(self._rooms) >= self.max_rooms:
                raise RoomError("busy", "no free rooms right now")
            while True:
                code = "".join(secrets.choice(_ALPHABET) for _ in range(CODE_LEN))
                if code not in self._rooms:
                    break
            self._rooms[code] = {"mode": mode, "seats": [state, None]}
            self._where[state.account] = code
            state.room = code
            return code

    def join(self, state, code: str) -> tuple[object, str]:
        with self._lock:
            if state.account in self._where:
                raise RoomError("in_room", "leave your room first")
            room = self._rooms.get(code)
            if room is None or room["seats"][1] is not None:
                raise RoomError("no_such_room", "no such room, or it is full")
            room["seats"][1] = state
            self._where[state.account] = code
            state.room = code
            return room["seats"][0], room["mode"]

    def peer_of(self, state):
        with self._lock:
            room = self._rooms.get(self._where.get(state.account, ""))
            if room is None:
                return None
            a, b = room["seats"]
            return b if a is state else a if b is state else None

    def leave(self, state):
        """Remove ``state`` from its room; returns the player left behind
        (to tell them), or ``None``."""
        with self._lock:
            code = self._where.pop(state.account, None) if state.account else None
            state.room = None
            room = self._rooms.get(code or "")
            if room is None:
                return None
            a, b = room["seats"]
            other = b if a is state else a
            if other is None:                       # last one out closes the room
                del self._rooms[code]
                return None
            room["seats"] = [other, None] if a is state else [a, None]
            if a is state:                          # the host left: the room ends
                del self._rooms[code]
                self._where.pop(other.account, None)
                other.room = None
            return other
