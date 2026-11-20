"""
TASK_game_feel_fix_1118 Fix 4: the crit impact frame must draw BEFORE
``top.png`` (where the HUD card backgrounds are actually baked) — not
after, which painted the wash straight over the opaque cards, darkening
them (``hud_layer_check.png``/G3). Real GL + manifest (``draw_screen``'s
call tree needs every real painter method) with ``_draw_crit_impact``
and ``painter.layer_region`` (Fix 10.1: PLAY's top.png now draws per
HUD card, not as one ``painter.layer`` call — see
``v3._draw_hud_top_cards``) wrapped to record call ORDER, not output.
"""

from __future__ import annotations

from pathlib import Path

from src.ui_album import v3
from src.ui_album import v3_flow as F
from src.ui_album import v3_painter
from src.ui_album import v3_screen_draw
from src.ui_album import v3_values

ROOT = Path(".").resolve()


def test_crit_impact_draws_before_the_top_layer(gl_context, monkeypatch):
    manifest = v3.V3Manifest(ROOT)
    painter = v3_painter.V3Painter(manifest, ROOT / "assets" / "shaders")
    painter.initialize()
    painter.set_frame((1280, 720))

    flow = F.V3Flow()
    flow.go(F.PLAY)
    state = v3_values.state_for(manifest, painter, flow, game=None, aim=(640.0, 360.0))
    state.crit_impact_alpha = 0.55

    order: list = []
    real_crit_impact = v3_screen_draw._draw_crit_impact
    real_layer_region = v3_painter.V3Painter.layer_region

    def tracking_crit_impact(painter_arg, state_arg):
        order.append("crit_impact")
        return real_crit_impact(painter_arg, state_arg)

    def tracking_layer_region(self, rect, relative, alpha=1.0):
        if relative == manifest.screens[F.PLAY].get("top"):
            order.append("top_layer")
        return real_layer_region(self, rect, relative, alpha)

    monkeypatch.setattr(v3, "_draw_crit_impact", tracking_crit_impact)
    monkeypatch.setattr(v3_painter.V3Painter, "layer_region", tracking_layer_region)

    painter.bind_screen(F.PLAY, v3_values.extra_files(manifest, F.PLAY, state))
    painter.begin()
    v3.draw_screen(painter, manifest, F.PLAY, state)
    painter.end()

    assert "crit_impact" in order
    assert "top_layer" in order
    assert order.index("crit_impact") < order.index("top_layer")
