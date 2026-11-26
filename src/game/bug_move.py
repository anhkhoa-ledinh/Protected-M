"""Bugs that move after the player has watched them for a while (levels 4-5).

A bug that has been continuously visible for ``move_after_s`` without dying
jumps to an unused spare spot (and its old spot becomes a spare): a candidate from the SAME placement run that
``_rank`` did not pick, already checked to be hidden at rest, revealed on a
lean, inside the viewport, and exposed at some eye position. At the moment of
the move it must also keep the placement spacing (3 x radius) from every other
live bug. No spare left means the bug stays.

Health lives in ``AlbumMatch.bug_hp`` keyed by bug index, so a move (which
rewrites the prop in place, same index) keeps it. The visibility timer
(``GameState.visible_since_s``) counts continuous visibility only: being shot
does not touch it, and a move resets it.

Pure logic, no GL.
"""

from __future__ import annotations

import logging
import random

from src.content.hidden_props import _nearer, is_visible

logger = logging.getLogger(__name__)

#: Same spacing rule as ``hidden_props_search._rank``.
SPACING_K = 3.0

#: The keys of a placement dict that describe WHERE the bug is.
SPOT_KEYS = ("pos_mm", "reveal_axis", "reveal_450_mm", "reveal_330_mm")


def too_close(pos_a, pos_b, radius_mm: float) -> bool:
    return (abs(pos_a[0] - pos_b[0]) < SPACING_K * radius_mm
            and abs(pos_a[1] - pos_b[1]) < SPACING_K * radius_mm)


class BugMover:
    def __init__(self, game, spares, move_after_s: float, radius_mm: float,
                 seed, flee_chance: float = 0.0,
                 flee_cooldown_s: float = 0.0) -> None:
        self.game = game
        self.spares = list(spares)
        self.move_after_s = float(move_after_s)
        self.radius_mm = float(radius_mm)
        self._rng = random.Random(f"move:{seed}")
        #: Chance (per shot) that a wounded bug flees; the level table's
        #: ``flee_chance``. Its own seeded stream, so a flee roll never shifts
        #: the shuffle the timed move uses and a match replays the same.
        self.flee_chance = float(flee_chance)
        self.flee_rng = random.Random(f"flee:{seed}")
        #: A bug that fled cannot flee again for this long (level table's
        #: ``flee_cooldown_s``); ``_fled_at`` = bug index -> time it last moved.
        self.flee_cooldown_s = float(flee_cooldown_s)
        self._fled_at: dict = {}

    def _pick(self, index, planes, eye):
        """A spare that keeps the spacing from every other live bug; one that
        is not in view at the current eye is preferred (the bug should not
        hop between two spots the player is looking at)."""
        live = [p["pos_mm"] for j, p in enumerate(self.game.props)
                if j != index and not self.game.found[j]]
        ok = [s for s in self.spares
              if not any(too_close(s["pos_mm"], q, self.radius_mm) for q in live)]
        if not ok:
            return None
        self._rng.shuffle(ok)
        for spare in ok:
            near = _nearer(planes, spare["pos_mm"][2] / 1000.0)
            if not is_visible(eye, spare["pos_mm"], self.radius_mm, near):
                return spare
        return ok[0]

    def tick(self, planes, eye, now: float) -> list:
        """Move every bug that has been seen long enough. Returns
        ``[(index, old_pos_mm, new_pos_mm, seen_s)]`` for the bugs that moved
        (``seen_s`` = how long the bug had been continuously visible)."""
        moved = []
        game = self.game
        for index, prop in enumerate(game.props):
            if game.found[index]:
                continue
            since = game.visible_since_s[index]
            if since is None or now - since < self.move_after_s:
                continue
            spare = self._pick(index, planes, eye)
            if spare is None:
                continue                      # nowhere to go: it stays
            old = self._relocate(index, spare)
            moved.append((index, old, tuple(prop["pos_mm"]), now - since))
        return moved

    def _relocate(self, index, spare) -> tuple:
        """Put bug ``index`` on ``spare`` (used up), reset its seen timer;
        returns the old position. Shared by the timed move and the flee.
        The old spot goes back into the pool AFTER the pick, so the bug can
        never move onto the spot it is leaving."""
        prop = self.game.props[index]
        old = tuple(prop["pos_mm"])
        old_spot = {key: prop[key] for key in SPOT_KEYS if key in prop}
        self.spares.remove(spare)
        for key in SPOT_KEYS:
            if key in spare:
                prop[key] = spare[key]
        self.spares.append(old_spot)
        self.game.visible_since_s[index] = None
        return old

    def flee(self, hits, planes, eye, now: float, hp_of=None) -> list:
        """After ONE shot: ``hits`` is ``match.last_shot_hits``
        (``(index, crit, killed)`` per pellet). One roll per shot, and only
        if some bug was damaged without dying; on success every such bug
        moves to a spare spot (same ``_pick`` rules as the timed move). A bug
        is eligible only if its last flee was ``flee_cooldown_s`` or more ago;
        with none eligible there is no roll at all. The cooldown starts only
        when the bug really moved. No spot means it stays, logged. Returns ``[(index, old_pos, new_pos)]``.
        ``hp_of(index)`` only feeds the log line; HP itself is never touched."""
        if self.flee_chance <= 0.0:
            return []
        dead = {i for i, _c, killed in hits if killed}
        wounded = sorted({
            i for i, _c, killed in hits
            if i not in dead
            and now - self._fled_at.get(i, float("-inf")) >= self.flee_cooldown_s})
        if not wounded:
            return []
        if self.flee_rng.random() >= self.flee_chance:
            return []
        moved = []
        for index in wounded:
            hp = "?" if hp_of is None else f"{hp_of(index):.0f}"
            spare = self._pick(index, planes, eye)
            if spare is None:
                logger.info("Bo #%d tron: khong con cho trong, bo dung yen "
                            "(stays) · mau %s", index, hp)
                continue
            old = self._relocate(index, spare)
            self._fled_at[index] = now
            new = tuple(self.game.props[index]["pos_mm"])
            logger.info("Bo #%d bi ban va tron (flee) · mau giu %s · "
                        "(%.0f, %.0f, %.0f) -> (%.0f, %.0f, %.0f)",
                        index, hp, *old, *new)
            moved.append((index, old, new))
        return moved
