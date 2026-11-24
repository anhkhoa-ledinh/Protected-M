"""PvP-relevant constants moved out of ``src.ui_album.v3_flow``
(TASK_arch_refactor_1120 R6) so ``src.net.server`` does not have to depend
on ``ui_album`` for them. ``v3_flow.py`` imports these back into its own
namespace (a genuine consumer — the v3 UI state machine still reads
``v3_flow.LEVEL_TABLE``/``v3_flow.ITEM_WEIGHT``/``v3_flow.PVP_MODES``
everywhere), not a compatibility shim.

NOT the same table as ``src.game.match.LEVEL_TABLE``/``ITEM_WEIGHT`` —
those are the OLD v2 album UI's Vietnamese-named version (moved from
``src.ui_album.flow`` in R5; that module itself was deleted as dead code
in TASK_cleanup_readme_1121); this is the v3 UI's English-named version
(its sprite keys, e.g. ``item_tile:Flare``, need English names) — same
numbers, different display/item-name strings.
"""

from __future__ import annotations

from src.game.levels import LEVELS

#: Bảng cấp: (tên, số bọ, giây, ô vật phẩm, phản công), đọc từ ``levels.LEVELS``.
LEVEL_TABLE = {lv: (r.name_en, r.bugs, r.seconds, r.slots, r.counter)
               for lv, r in LEVELS.items()}

ITEM_WEIGHT = {"Flare": 3, "Antidote": 5, "Acid shield": 4, "Filter lens": 4,
              "Compass": 2, "Thermal scan": 8, "Energy drink": 3,
              "Smoke bomb": 6}

#: An item costs ``weight x ITEM_PRICE_PER_WEIGHT`` points, paid from the
#: account wallet when a match starts with it equipped.
ITEM_PRICE_PER_WEIGHT = 10


def item_price(item: str) -> int:
    return ITEM_WEIGHT.get(item, 0) * ITEM_PRICE_PER_WEIGHT


#: The items that really work, and for how many seconds one use lasts: the one
#: table. Every other item in ``ITEM_WEIGHT`` is "Coming soon": shown on the
#: loadout, never equippable, never charged.
ITEM_EFFECT_S = {"Flare": 8.0, "Thermal scan": 6.0}


def item_is_live(item: str) -> bool:
    return item in ITEM_EFFECT_S

PVP_MODES = ("Competitive", "Hunt & Hide")

#: PvP opens once every solo level has been won at least once.
PVP_UNLOCK_LEVELS = (1, 2, 3, 4, 5)
