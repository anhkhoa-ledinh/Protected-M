"""
TASK_hkey_profile_1119: optional, pure-measurement per-frame stage
timing. Zero effect unless ``HCPD_FRAME_PROFILE=1`` is set — every
``stage()``/``record_ms()`` call becomes a true no-op otherwise, so
normal play is untouched.

Each named stage records its OWN ms cost for the current frame
(``begin_frame()``/``end_frame()`` bracket one frame); the history is
kept per stage name across the whole run so the caller can report
median/p95 at the end. This module holds no opinion about WHICH stages
exist — call sites in ``app.py``/``renderer.py``/``v3_ui.py`` decide
that; this is just the stopwatch + the list.
"""

from __future__ import annotations

import os
import time
from collections import defaultdict
from contextlib import contextmanager

ENABLED = os.environ.get("HCPD_FRAME_PROFILE") == "1"

_history: dict[str, list[float]] = defaultdict(list)
_frame: dict[str, float] = {}


def begin_frame() -> None:
    if ENABLED:
        _frame.clear()


def record_ms(name: str, ms: float) -> None:
    if not ENABLED:
        return
    _frame[name] = _frame.get(name, 0.0) + float(ms)


@contextmanager
def stage(name: str):
    if not ENABLED:
        yield
        return
    t0 = time.perf_counter()
    try:
        yield
    finally:
        record_ms(name, (time.perf_counter() - t0) * 1000.0)


def current_frame() -> dict[str, float]:
    """Peek this frame's totals SO FAR, without flushing to history yet
    — for a derived stage (e.g. "ui_painter" = draw_total minus its own
    nested sub-stages) that needs to read siblings before adding itself,
    then still land in the SAME frame's history via one ``end_frame()``."""
    return dict(_frame)


def end_frame() -> dict[str, float]:
    """Push this frame's totals into history; return them (callers can
    derive a stage BY SUBTRACTION, e.g. "ui_painter" = draw_total minus
    the sub-stages nested inside it, without this module knowing that
    relationship itself)."""
    if not ENABLED:
        return {}
    this_frame = dict(_frame)
    for name, ms in this_frame.items():
        _history[name].append(ms)
    return this_frame


def history() -> dict[str, list[float]]:
    return {k: list(v) for k, v in _history.items()}


def reset_history() -> None:
    _history.clear()


def _percentile(sorted_values: list[float], p: float) -> float:
    if not sorted_values:
        return 0.0
    index = min(len(sorted_values) - 1, int(round(p * (len(sorted_values) - 1))))
    return sorted_values[index]


def report_lines(extra: dict[str, list[float]] | None = None) -> list[str]:
    """``["stage  n=.. median=.. p95=.. max=..", ...]``, widest-median
    first — the exact table ``TASK_hkey_profile_1119`` asks for.
    ``extra``: stage histories from OUTSIDE this module (e.g. the
    existing ``Profiler``'s webcam/tracking samples), merged in so the
    report is one table, not two."""
    merged = dict(history())
    for name, values in (extra or {}).items():
        merged.setdefault(name, []).extend(values)
    rows = []
    for name, values in merged.items():
        if not values:
            continue
        ordered = sorted(values)
        rows.append((name, len(ordered), ordered[len(ordered) // 2],
                    _percentile(ordered, 0.95), ordered[-1]))
    rows.sort(key=lambda r: r[2], reverse=True)
    return [f"{name:20s} n={n:5d}  median={median:7.3f} ms  "
           f"p95={p95:7.3f} ms  max={mx:7.3f} ms"
           for name, n, median, p95, mx in rows]
