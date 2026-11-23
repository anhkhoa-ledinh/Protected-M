"""How many bugs a photo can really hold for a level, without opening a match.

Runs the SAME steps ``HcpdApplication._start_album_match`` runs (``place_props``
at ``count × 3``, the viewport filter, the 5×5 exposure check), so with the
same seed and rest eye the number equals what the match then places. Pure
logic, no GL; the loading screen uses it to say the real count up front.
"""

from __future__ import annotations

from src.content.hidden_props_search import exposure_count, place_props
from src.content.prop_rest_eye import z_rest_mm
from src.game.viewport_filter import filter_props


def placeable_bugs(planes, count, radius_mm, rest_eye_mm, glass_mm, rect,
                   centre_mm, seed):
    """``(winnable props, outside the viewport, dropped as unwinnable)``."""
    placed = place_props(planes, int(count) * 3, float(radius_mm),
                         tuple(rest_eye_mm[:2]),
                         z_rest=float(z_rest_mm(rest_eye_mm[2])), seed=seed)
    if not placed:
        return [], 0, 0
    kept, outside = filter_props(placed, rest_eye_mm, int(count), glass_mm,
                                 rect, centre_mm)
    winnable = [p for p in kept
                if exposure_count(planes, p["pos_mm"], p["radius_mm"],
                                  rest_eye_mm[:2], rest_eye_mm[2]) > 0]
    return winnable, outside, len(kept) - len(winnable)
