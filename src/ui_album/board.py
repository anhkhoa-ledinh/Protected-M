"""Real leaderboard data and the runtime covers over the baked sample art.

``v3_14_leaderboard/base.png`` has a fake podium, six fake rows and two sample
captions printed into it. The art is not edited: ``covers_for`` returns flat
shapes in the panel's own colours that hide those boxes at runtime, and the
real text is drawn on top (see ``fill_state``).

Switch the covers off in ONE place: set ``COVERS_ENABLED = False`` once the
user exports a blank version of the board from Penpot.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

from src.account import crypto as account_crypto

#: The one switch for every cover below.
COVERS_ENABLED = True

SCREEN = "v3_14_leaderboard"
SEG_SOLO = "Solo"
NO_SCORE = "—"
PODIUM_SLOTS = 3
ROW_SLOTS = 6                    # ranks 4..9

INK, INK_SOFT, MUTED, PRIMARY = "#24415F", "#5D7894", "#8DA3BA", "#3B72B4"
WHITE, AVATAR_FILL, BADGE_FILL, BADGE_INK = "#FFFFFF", "#DCE9F7", "#FFF3D6", "#9A5A1E"
K = 1.5                          # art px (1920x1080) per design px (1280x720)


def _d(x0, y0, x1, y1) -> tuple:
    """A box in ART pixels (corners) -> design-pixel ``(x, y, w, h)``."""
    return (x0 / K, y0 / K, (x1 - x0) / K, (y1 - y0) / K)


# ---- the data -------------------------------------------------------------

@dataclass(frozen=True)
class Entry:
    key: str
    name: str
    earned: int
    best_level: int              # highest level won, 0 = none
    rank: int = 0


def collect(data: dict | None) -> list[Entry]:
    """Every account on this computer (never the guest), ranked by
    ``earned_total`` high to low; ties share a rank and sort by name; accounts
    that never won anything come last."""
    rows = []
    for key, record in ((data or {}).get("accounts") or {}).items():
        if key == account_crypto.GUEST_KEY or not isinstance(record, dict):
            continue
        cleared = [int(n) for n in record.get("levels_cleared") or ()
                   if isinstance(n, (int, float))]
        rows.append(Entry(key, str(record.get("display") or key),
                          max(0, int(record.get("earned_total") or 0)),
                          max(cleared, default=0)))
    rows.sort(key=lambda e: (-e.earned, e.name.casefold(), e.key))
    ranked, last_total, last_rank = [], None, 0
    for index, entry in enumerate(rows):
        if entry.earned != last_total:
            last_total, last_rank = entry.earned, index + 1
        ranked.append(Entry(entry.key, entry.name, entry.earned,
                            entry.best_level, last_rank))
    return ranked


def rank_of(data: dict | None, key: str) -> int | None:
    """Board rank of account ``key``, or ``None`` if it has won nothing yet."""
    for entry in collect(data):
        if entry.key == key:
            return entry.rank if entry.earned > 0 else None
    return None


def points(entry: Entry, suffix: str = "") -> str:
    return f"{entry.earned:,}{suffix}" if entry.earned > 0 else NO_SCORE


def badge(entry: Entry) -> str:
    return f"Level {entry.best_level} clear" if entry.best_level else ""


# ---- the covers ------------------------------------------------------------
#: Podium columns: centre x, ring-letter box, text block (all art px), and the
#: text rows (name / points / badge) as art-px centre y.
_PODIUM = (
    (516, (486, 404, 546, 466), (410, 508, 622, 628), (526, 561, 604)),     # 1st
    (264, (238, 490, 290, 546), (150, 582, 378, 698), (599, 633, 676)),     # 2nd
    (768, (742, 530, 794, 586), (680, 622, 850, 740), (637, 672, 716)),     # 3rd
)
_ROW_CY = (316, 400, 484, 568, 652, 736)        # art px, ranks 4..9
_YOU_CY = 869
_DOTS = _d(1380, 788, 1430, 815)
_SUBTITLE = (90, 162, 320, 210)                 # on the sky: "Updated every hour."
_FOOTER = (90, 955, 216, 996)                   # on the sky: "Sample data"
_SEG_TRACK = (985, 166, 1823, 228)
_PANELS = (_d(120, 290, 912, 906), _d(1000, 262, 1808, 918))


@lru_cache(maxsize=16)
def _inpaint(path: str, box: tuple):
    """RGBA patch for ``box`` (art px) of the base art: every pixel row is a
    straight blend between the colour just left of the box and the colour just
    right of it, so the sky behind a printed caption continues across it."""
    import numpy as np
    from PIL import Image

    x0, y0, x1, y1 = box
    with Image.open(path) as image:
        rgb = np.asarray(image.convert("RGB"), dtype=np.float64)
    left = rgb[y0:y1, x0 - 4:x0].mean(axis=1)           # (rows, 3)
    right = rgb[y0:y1, x1:x1 + 4].mean(axis=1)
    t = np.linspace(0.0, 1.0, x1 - x0)[None, :, None]
    patch = left[:, None, :] * (1.0 - t) + right[:, None, :] * t
    out = np.empty((y1 - y0, x1 - x0, 4), dtype=np.uint8)
    out[..., :3] = np.rint(patch).astype(np.uint8)
    out[..., 3] = 255
    return out


@lru_cache(maxsize=16)
def _row_colours(path: str, box: tuple) -> tuple:
    """Median colour of every pixel row of ``box`` (art px) of the base art:
    text strokes are the minority in each row, so this is the sky behind it."""
    from PIL import Image, ImageStat

    x0, y0, x1, y1 = box
    with Image.open(path) as image:
        image = image.convert("RGB")
        out = []
        for y in range(y0, y1):
            median = ImageStat.Stat(image.crop((x0, y, x1, y + 1))).median
            out.append("#%02X%02X%02X" % tuple(int(v) for v in median))
    return tuple(out)


def _strips(base_path, box) -> tuple:
    box = tuple(box)
    return ("patch", _d(*box), ("board_patch", str(base_path), box),
            _inpaint(str(base_path), box))


def covers_for(entries: list[Entry], seg: str, base_path) -> list:
    """Cover shapes for the board, in draw order."""
    if not COVERS_ENABLED:
        return []
    out = [_strips(base_path, _SUBTITLE), _strips(base_path, _FOOTER)]
    if seg != SEG_SOLO:
        track = _row_colours(str(base_path), (1190, 170, 1230, 171))[0]
        out += [("rect", p, WHITE) for p in _PANELS]
        out.append(("rounded", _d(*_SEG_TRACK), 31.0 / K, track))
        return out
    for column, (_cx, letter, block, _ys) in enumerate(_PODIUM):
        out.append(("rect", _d(*letter), WHITE))
        out.append(("rect", _d(*block), WHITE))
    for index, cy in enumerate(_ROW_CY):
        out.append(("rect", _d(1024, cy - 26, 1066, cy + 26), WHITE))     # rank
        out.append(("rect", _d(1146, cy - 26, 1795, cy + 26), WHITE))     # name, score
        if index + PODIUM_SLOTS < len(entries):
            out.append(("rect", _d(1094, cy - 16, 1120, cy + 16), AVATAR_FILL))
        else:
            out.append(("rect", _d(1072, cy - 30, 1144, cy + 30), WHITE))
    if len(entries) <= PODIUM_SLOTS + ROW_SLOTS:
        out.append(("rect", _DOTS, WHITE))                    # "…" not needed
    out.append(("rect", _d(1094, _YOU_CY - 14, 1120, _YOU_CY + 14), PRIMARY))
    return out


# ---- the real text ----------------------------------------------------------

def _run(text: str, font: str, weight: int, size: float, colour: str,
         spacing: float = 0.0) -> dict:
    return {"c": text, "ff": font, "fw": weight, "fs": size, "col": colour,
            "ls": spacing, "upper": False}


def fit(painter, run: dict, text: str, width: float) -> str:
    """``text`` cut with "…" so it is at most ``width`` design px wide."""
    if painter is None or painter.text_width([run], [text]) <= width:
        return text
    lo, hi, best = 0, len(text), "…"
    while lo <= hi:
        mid = (lo + hi) // 2
        candidate = text[:mid].rstrip() + "…"
        if painter.text_width([run], [candidate]) <= width:
            best, lo = candidate, mid + 1
        else:
            hi = mid - 1
    return best


def _put(state, painter, rect, run, text, align, width=None) -> None:
    text = fit(painter, run, text, width if width is not None else rect[2])
    state.extra.append((list(rect), [run], [text], align))


def _initial(name: str) -> str:
    text = (name or "").strip()
    return text[:1].upper() if text else "?"


def fill_state(state, flow, manifest, painter, data) -> None:
    """Fill ``state`` for the leaderboard: covers + real text. ``data`` is the
    accounts store; ``flow.account_key`` says which row is "you"."""
    entries = collect(data)
    seg = flow.board_seg or SEG_SOLO
    state.covers = covers_for(entries, seg, manifest.path(f"{SCREEN}/base.png"))
    state.extra.append((_d(96, 168, 520, 204), [_run(
        "Accounts on this computer", "Patrick Hand", 400, 20.0, INK_SOFT)],
        ["Accounts on this computer"], "left"))
    mine = next((e for e in entries if e.key == flow.account_key), None)
    state.texts["you_rank"] = str(mine.rank) if mine else NO_SCORE
    you_name = f"You · {flow.player or 'guest'}"
    run = _run("", "Nunito", 900, 15.0, INK)
    state.texts["you_name"] = fit(painter, run, you_name, 270.0)
    state.text_rect["you_name"] = [768, 570, 270, 18]
    state.texts["you_score"] = points(mine) if mine else NO_SCORE
    if seg != SEG_SOLO:
        state.hidden |= {"you_rank", "you_name", "you_score"}
        _segment(state, seg)
        return
    initial = _initial(flow.player or "guest")
    state.extra.append(([738 - 14, _YOU_CY / K - 10, 28, 20], [_run(
        initial, "Nunito", 900, 14.0, WHITE)], [initial], "center"))
    for slot, (cx, _letter, _block, ys) in enumerate(_PODIUM):
        if slot >= len(entries):
            continue
        entry = entries[slot]
        x = cx / K
        lx, ly = (_letter[0] + _letter[2]) / 2 / K, (_letter[1] + _letter[3]) / 2 / K
        state.extra.append(((lx - 20, ly - 17, 40, 34), [_run(
            _initial(entry.name), "Nunito", 900, 26.0, INK)],
            [_initial(entry.name)], "center"))
        _put(state, painter, (x - 75, ys[0] / K - 10, 150, 20),
             _run("", "Nunito", 800, 15.0, INK), entry.name, "center")
        _put(state, painter, (x - 75, ys[1] / K - 12, 150, 24),
             _run("", "Baloo 2", 800, 18.0, PRIMARY), points(entry, " pts"), "center")
        text = badge(entry)
        if text:
            run = _run(text, "Nunito", 900, 11.0, BADGE_INK, 0.5)
            width = (painter.text_width([run], [text]) if painter else 80.0) + 18.0
            pill = (x - width / 2, ys[2] / K - 11, width, 22)
            state.covers.append(("rounded", pill, 11.0, BADGE_FILL))
            state.extra.append(((pill[0], pill[1], pill[2], pill[3]), [run],
                                [text], "center"))
    for index in range(ROW_SLOTS):
        at = index + PODIUM_SLOTS
        if at >= len(entries):
            break
        entry, cy = entries[at], _ROW_CY[index] / K
        state.extra.append(((694 - 14, cy - 11, 28, 22), [_run(
            str(entry.rank), "Baloo 2", 800, 17.0, MUTED)], [str(entry.rank)], "center"))
        letter = _initial(entry.name)
        state.extra.append(((738 - 14, cy - 10, 28, 20), [_run(
            letter, "Nunito", 900, 13.0, PRIMARY)], [letter], "center"))
        _put(state, painter, (768, cy - 10, 270, 20),
             _run("", "Nunito", 800, 15.0, INK), entry.name, "left")
        state.extra.append(((1056, cy - 11, 136, 22), [_run(
            points(entry), "Baloo 2", 800, 18.0, INK)], [points(entry)], "right"))


#: Hotspot rects of the three segments (design px) from the manifest.
_SEGMENTS = (("Solo", (662, 114, 120, 34)),
             ("PvP · Competitive", (782, 114, 206, 34)),
             ("PvP · Hunt & Hide", (988, 114, 222, 34)))


def _segment(state, seg: str) -> None:
    """The two PvP tabs: nothing recorded yet. The baked pill sits on Solo, so
    it is covered and redrawn under the chosen tab."""
    for name, rect in _SEGMENTS:
        chosen = name == seg
        if chosen:
            state.covers.append(("rounded", rect, 17.0, PRIMARY))
        state.extra.append((rect, [_run(name, "Nunito", 800, 13.5,
                                        WHITE if chosen else PRIMARY)],
                            [name], "center"))
    for panel in _PANELS:
        x, y, w, h = panel
        state.extra.append(((x, y + h / 2 - 14, w, 28), [_run(
            "No matches yet", "Nunito", 800, 18.0, INK_SOFT)],
            ["No matches yet"], "center"))
