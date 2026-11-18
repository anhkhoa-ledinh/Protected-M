"""
TASK_game_feel_1118 Phase 4: ``src/config/prefs.py``'s ``screen_shake``
key/``HCPD_NO_SHAKE`` off switch — same precedence as the existing
``fx``/``HCPD_NO_FX`` pair, tested directly (no GL needed) against a
throwaway ``user_prefs.json``.
"""

from __future__ import annotations

import json

from src.config import prefs


def test_screen_shake_defaults_true_with_no_prefs_file(tmp_path):
    assert prefs.screen_shake_enabled(tmp_path) is True
    assert prefs.load(tmp_path)["screen_shake"] is True


def test_screen_shake_false_in_prefs_file(tmp_path):
    path = prefs.prefs_path(tmp_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"screen_shake": False}), encoding="utf-8")
    assert prefs.screen_shake_enabled(tmp_path) is False


def test_hcpd_no_shake_env_wins_over_prefs_true(tmp_path, monkeypatch):
    monkeypatch.setenv(prefs.NO_SHAKE_ENV, "1")
    assert prefs.screen_shake_enabled(tmp_path) is False


def test_save_preserves_screen_shake_when_not_passed(tmp_path):
    prefs.save(tmp_path, "window", screen_shake=False)
    assert prefs.load(tmp_path)["screen_shake"] is False
    # A later save() that doesn't mention screen_shake must not reset it.
    prefs.save(tmp_path, "fullscreen")
    assert prefs.load(tmp_path)["screen_shake"] is False


def test_save_can_change_screen_shake(tmp_path):
    prefs.save(tmp_path, "fullscreen", screen_shake=True)
    assert prefs.load(tmp_path)["screen_shake"] is True
    prefs.save(tmp_path, "fullscreen", screen_shake=False)
    assert prefs.load(tmp_path)["screen_shake"] is False


# ---- TASK_game_feel_fix_1118 Fix 10.4: hud_compact, same shape, no env ----

def test_hud_compact_defaults_false_with_no_prefs_file(tmp_path):
    assert prefs.hud_compact_enabled(tmp_path) is False
    assert prefs.load(tmp_path)["hud_compact"] is False


def test_hud_compact_true_in_prefs_file(tmp_path):
    path = prefs.prefs_path(tmp_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"hud_compact": True}), encoding="utf-8")
    assert prefs.hud_compact_enabled(tmp_path) is True


def test_save_preserves_hud_compact_when_not_passed(tmp_path):
    prefs.save(tmp_path, "window", hud_compact=True)
    assert prefs.load(tmp_path)["hud_compact"] is True
    prefs.save(tmp_path, "fullscreen")
    assert prefs.load(tmp_path)["hud_compact"] is True


def test_save_can_change_hud_compact(tmp_path):
    prefs.save(tmp_path, "fullscreen", hud_compact=True)
    assert prefs.load(tmp_path)["hud_compact"] is True
    prefs.save(tmp_path, "fullscreen", hud_compact=False)
    assert prefs.load(tmp_path)["hud_compact"] is False
