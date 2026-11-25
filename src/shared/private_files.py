"""Make a file or folder readable by the current user only, on every OS.

POSIX: mode ``0o600`` for files, ``0o700`` for folders.
Windows: drop the inherited permissions and grant only the current user,
with the built-in ``icacls`` (no extra package).

``is_private`` checks the same property in each system's own terms, so one
test can assert it everywhere.
"""

from __future__ import annotations

import os
import re
import subprocess
from pathlib import Path


class PrivateFilesError(OSError):
    """The permissions could not be set or read."""


# ---- seams the tests replace --------------------------------------------------

def _is_windows() -> bool:
    return os.name == "nt"


def _current_user() -> str:
    name = os.environ.get("USERNAME") or os.environ.get("USER") or ""
    if not name:
        raise PrivateFilesError("cannot tell the current user (USERNAME is empty)")
    domain = os.environ.get("USERDOMAIN", "")
    return f"{domain}\\{name}" if domain else name


def _icacls(args: list[str]) -> str:
    """Run ``icacls`` and return its stdout; raise if it fails."""
    try:
        done = subprocess.run(["icacls", *args], capture_output=True, text=True,
                              timeout=30)
    except (OSError, subprocess.SubprocessError) as exc:
        raise PrivateFilesError(f"icacls could not run: {exc}") from exc
    if done.returncode != 0:
        raise PrivateFilesError(
            f"icacls failed ({done.returncode}): {(done.stderr or done.stdout).strip()}")
    return done.stdout


# ---- public API ---------------------------------------------------------------

def make_private(path) -> None:
    path = Path(path)
    if _is_windows():
        grant = "(OI)(CI)F" if path.is_dir() else "F"
        _icacls([str(path), "/inheritance:r", "/grant:r", f"{_current_user()}:{grant}"])
        return
    os.chmod(path, 0o700 if path.is_dir() else 0o600)


def is_private(path) -> bool:
    path = Path(path)
    if _is_windows():
        return _windows_is_private(path)
    return (path.stat().st_mode & 0o077) == 0


def acl_report(path) -> str:
    """Describe the permissions of ``path`` for a failing assertion. Never raises."""
    try:
        path = Path(path)
        if not _is_windows():
            return f"{path}: mode {oct(path.stat().st_mode & 0o7777)}"
        parts = [f"{path}:"]
        try:
            done = subprocess.run(["icacls", str(path)], capture_output=True,
                                  text=True, timeout=30)
            parts.append(f"icacls rc={done.returncode}\n{done.stdout}{done.stderr}")
        except Exception as exc:
            parts.append(f"icacls could not run: {exc!r}")
        try:
            parts.append(f"parsed: {_aces(path, _icacls([str(path)]))}")
        except Exception as exc:
            parts.append(f"parsed: error {exc!r}")
        try:
            parts.append(f"user: {_current_user()}")
        except Exception as exc:
            parts.append(f"user: error {exc!r}")
        return "\n".join(parts)
    except Exception as exc:
        return f"acl_report failed: {exc!r}"


# ---- Windows: parse `icacls <path>` -------------------------------------------

_ACE = re.compile(r"^(?P<who>.+?):(?P<flags>(?:\([^)]*\))+)\s*$")


def _aces(path: Path, output: str) -> list[tuple[str, list[str]]]:
    """``[(principal, [flag, ...]), ...]`` from the output of ``icacls path``."""
    found = []
    for line in output.splitlines():
        line = line.rstrip()
        if line.startswith(str(path)):
            line = line[len(str(path)):]
        line = line.strip()
        if not line or line.startswith("Successfully processed"):
            continue
        m = _ACE.match(line)
        if m:
            flags = re.findall(r"\(([^)]*)\)", m.group("flags"))
            found.append((m.group("who").strip(), flags))
    return found


def _same_user(principal: str, user: str) -> bool:
    return principal.lower() == user.lower() or \
        principal.lower().rsplit("\\", 1)[-1] == user.lower().rsplit("\\", 1)[-1]


def _windows_is_private(path: Path) -> bool:
    aces = _aces(path, _icacls([str(path)]))
    if not aces:
        return False
    user = _current_user()
    for principal, flags in aces:
        if not _same_user(principal, user):
            return False                      # someone else has an entry
        if "I" in flags:
            return False                      # still inherited from the parent
    return True
