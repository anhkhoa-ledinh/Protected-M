"""TASK_signin_toggles_1120: the eye icon and the Remember-me switch on
the sign-in/sign-up/sign-in-error screens.

Root causes (see the task report for the one-sentence version):
- the eye: ``field_pass``'s hotspot fully contains ``icon_show_password``'s
  and was listed first in the manifest, so ``v3_hit.hit_test`` always resolved
  a click on the icon to the field underneath it;
- the switch: nothing ever painted an OFF look — ``base.png`` only drew it
  ON, and ``state_for``/``_sign_in`` never even copied ``flow.remember``
  into the per-frame draw state.

No test here touches a real config file — everything runs against
``V3Flow``/``SimpleNamespace`` objects or the manifest data directly."""

from __future__ import annotations

import json
import unittest
from pathlib import Path
from types import SimpleNamespace

from src.ui_album import v3_hit, v3_screen_draw, v3_values, v3_routes
from src.ui_album import v3_flow as F

ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = ROOT / "assets" / "ui_v3" / "manifest.json"
MANIFEST_DATA = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))


class FakeManifest:
    def __init__(self) -> None:
        self.screens = MANIFEST_DATA["screens"]


def _new_state() -> SimpleNamespace:
    return SimpleNamespace(texts={}, sprites={}, hidden=set(), caret=None,
                           focus=None, remember=False)


# ---- bug 1: the eye's click was swallowed by the password field -----------

class EyeHitTestTest(unittest.TestCase):
    """A click on ``icon_show_password`` must resolve to the icon, not to
    ``field_pass`` sitting underneath it."""

    SCREENS = ("v3_01a_sign_in", "v3_01b_sign_up", "v3_01c_sign_in_error")

    def test_eye_wins_over_the_password_field_on_every_screen(self) -> None:
        manifest = FakeManifest()
        for screen in self.SCREENS:
            with self.subTest(screen=screen):
                hotspots = {h["n"]: h for h in manifest.screens[screen]["hotspots"]}
                icon_rect = v3_hit.hotspot_rect(screen, hotspots["icon_show_password"])
                cx = icon_rect[0] + icon_rect[2] / 2.0
                cy = icon_rect[1] + icon_rect[3] / 2.0
                self.assertEqual(v3_hit.hit_test(manifest, screen, cx, cy),
                                 "icon_show_password")

    def test_password_field_still_catches_clicks_outside_the_icon(self) -> None:
        """The fix must not turn the field itself into a dead zone — only
        the icon's own small rect should be carved out of it."""
        manifest = FakeManifest()
        hotspots = {h["n"]: h for h in manifest.screens["v3_01a_sign_in"]["hotspots"]}
        field_rect = v3_hit.hotspot_rect("v3_01a_sign_in", hotspots["field_pass"])
        point = (field_rect[0] + 5.0, field_rect[1] + 5.0)   # top-left corner
        self.assertEqual(v3_hit.hit_test(manifest, "v3_01a_sign_in", *point),
                         "field_pass")


class EyeToggleTest(unittest.TestCase):
    """``icon_show_password``'s click handler and the resulting mask."""

    def test_route_toggles_show_password_both_ways_without_moving_focus(self) -> None:
        flow = F.V3Flow()
        flow.screen = F.SIGN_IN
        flow.focus = "field_pass"
        self.assertFalse(flow.show_password)

        v3_routes.on_click(flow, "icon_show_password", 0.0)
        self.assertTrue(flow.show_password)
        self.assertEqual(flow.focus, "field_pass")   # caret/focus untouched

        v3_routes.on_click(flow, "icon_show_password", 0.0)
        self.assertFalse(flow.show_password)

    def test_mask_follows_show_password_and_value_never_changes(self) -> None:
        flow = F.V3Flow()
        flow.screen = F.SIGN_IN
        flow.player = "anhkhoa"
        flow.password = "s3cret!"

        state = _new_state()
        v3_values._sign_in(state, flow)
        self.assertEqual(state.texts["pass"], "•" * len(flow.password))

        flow.show_password = True
        state = _new_state()
        v3_values._sign_in(state, flow)
        self.assertEqual(state.texts["pass"], "s3cret!")

        flow.show_password = False
        state = _new_state()
        v3_values._sign_in(state, flow)
        self.assertEqual(state.texts["pass"], "•" * len(flow.password))
        # The typed value itself was never touched by any of the above.
        self.assertEqual(flow.password, "s3cret!")


# ---- bug 2: Remember-me never painted its OFF state ------------------------

class RememberToggleTest(unittest.TestCase):
    def test_route_toggles_remember_both_ways(self) -> None:
        flow = F.V3Flow()
        flow.screen = F.SIGN_IN
        self.assertFalse(flow.remember)

        v3_routes.on_click(flow, "switch_remember", 0.0)
        self.assertTrue(flow.remember)

        v3_routes.on_click(flow, "switch_remember", 0.0)
        self.assertFalse(flow.remember)

    def test_sign_in_state_mirrors_flow_remember(self) -> None:
        flow = F.V3Flow()
        flow.screen = F.SIGN_IN

        state = _new_state()
        v3_values._sign_in(state, flow)
        self.assertFalse(state.remember)

        flow.remember = True
        state = _new_state()
        v3_values._sign_in(state, flow)
        self.assertTrue(state.remember)


class _FakePainter:
    def __init__(self) -> None:
        self.calls: list[tuple] = []

    def rounded(self, rect, radius, colour) -> None:
        self.calls.append(("rounded", rect, radius, colour))

    def disc(self, rect, colour) -> None:
        self.calls.append(("disc", rect, colour))


class RememberSwitchPaintTest(unittest.TestCase):
    """ON needs no extra draw (the baked art already shows it); OFF must
    paint a grey track + a knob pushed to the left, over the baked art."""

    def _record(self):
        return MANIFEST_DATA["screens"]["v3_01a_sign_in"]

    def test_on_draws_nothing_extra(self) -> None:
        painter = _FakePainter()
        state = SimpleNamespace(remember=True)
        v3_screen_draw._draw_remember_switch(painter, "v3_01a_sign_in", self._record(), state)
        self.assertEqual(painter.calls, [])

    def test_off_draws_a_grey_track_and_a_left_knob(self) -> None:
        painter = _FakePainter()
        state = SimpleNamespace(remember=False)
        v3_screen_draw._draw_remember_switch(painter, "v3_01a_sign_in", self._record(), state)
        kinds = [call[0] for call in painter.calls]
        self.assertEqual(kinds, ["rounded", "disc"])
        hotspots = {h["n"]: h for h in self._record()["hotspots"]}
        track_x, track_y, track_w, track_h = v3_hit.hotspot_rect(
            "v3_01a_sign_in", hotspots["switch_remember"])
        _kind, knob_rect, _colour = painter.calls[1]
        knob_x = knob_rect[0]
        # Knob sits in the LEFT half of the track (OFF), not the right.
        self.assertLess(knob_x, track_x + track_w / 2.0)


if __name__ == "__main__":
    unittest.main()
