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

#: Bảng cấp: (tên, số bọ, giây, ô vật phẩm, phản công). Con số giữ nguyên của
#: ``flow.py``; chỉ cái tên đổi sang tiếng Anh theo U2.
LEVEL_TABLE = {
    1: ("Easy", 6, 180, 2, False), 2: ("Normal", 8, 150, 3, False),
    3: ("Hard", 10, 150, 3, True), 4: ("Infestation", 12, 120, 4, True),
    5: ("Asian Mode", 12, 120, 4, True),
}

ITEM_WEIGHT = {"Flare": 3, "Antidote": 5, "Acid shield": 4, "Filter lens": 4,
              "Compass": 2, "Thermal scan": 8, "Energy drink": 3,
              "Smoke bomb": 6}

PVP_MODES = ("Competitive", "Hunt & Hide")
