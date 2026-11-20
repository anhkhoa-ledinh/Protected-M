"""TASK_ui_sakuga_1117 Phase 7: ``src/fx/player.py`` — pure Python, no GL."""

import math

import pytest

from src.fx.player import MAX_LIVE_EFFECTS, FxPlayer
from src.fx.spec import load_spec


@pytest.fixture(scope="module")
def spec():
    return load_spec(".")


def test_spawn_and_frame_indexing(spec):
    player = FxPlayer(spec, seed=1)
    player.spawn("hit_kill", 100.0, 200.0, variant=0, words=False)
    assert len(player._effects) == 1
    effect = player._effects[0]
    frames = spec.effects["hit_kill"]["variants"][0]["frames"]
    player.step(1.0 / spec.fps * 0.4)
    commands = player.draw_list()
    assert commands  # frame 0 (or thereabouts) should be drawn
    assert commands[0].texture_key == spec.effects["hit_kill"]["variants"][0]["sheet"]
    _ = effect, frames


def test_effect_ends_and_is_removed(spec):
    player = FxPlayer(spec, seed=2)
    player.spawn("ui_tap", 0.0, 0.0, variant=0, words=False)
    end_time = player._end_time(player._effects[0])
    player.step(end_time + 1.0 / spec.fps)
    assert len(player._effects) == 0


def test_words_false_produces_no_word_commands(spec):
    player = FxPlayer(spec, seed=3)
    player.spawn("hit_kill", 0.0, 0.0, variant=0, words=False)
    player.step(0.1)
    for cmd in player.draw_list():
        assert "words/" not in cmd.texture_key


def test_words_true_eventually_produces_word_command(spec):
    player = FxPlayer(spec, seed=4)
    player.spawn("hit_kill", 0.0, 0.0, variant=0, words=True)
    for _ in range(10):
        player.step(1.0 / spec.fps)
    commands = player.draw_list()
    assert any("words/" in c.texture_key for c in commands)


def test_max_live_effects_cap(spec):
    player = FxPlayer(spec, seed=5)
    for i in range(MAX_LIVE_EFFECTS + 3):
        player.spawn("ui_tap", float(i), 0.0, variant=0, words=False)
    assert len(player._effects) == MAX_LIVE_EFFECTS
    # Oldest 3 were dropped; the first surviving effect is index 3.
    assert player._effects[0].x == 3.0


def test_mirror_flips_offset_and_uv(spec):
    player_a = FxPlayer(spec, seed=6)
    player_a.spawn("hit_kill", 500.0, 500.0, variant=0, mirror=False, words=False)
    player_a.step(0.01)
    normal = player_a.draw_list()[0]

    player_b = FxPlayer(spec, seed=6)
    player_b.spawn("hit_kill", 500.0, 500.0, variant=0, mirror=True, words=False)
    player_b.step(0.01)
    mirrored = player_b.draw_list()[0]

    assert normal.corners != mirrored.corners
    assert normal.uv[0] == mirrored.uv[2]
    assert normal.uv[2] == mirrored.uv[0]


def test_spawn_word_bare_no_sprite(spec):
    player = FxPlayer(spec, seed=7)
    player.spawn_word("gogo", "ink", 10.0, 10.0, size=150.0, rot=-0.08, life=1.0)
    assert len(player._effects) == 1
    player.step(0.01)
    commands = player.draw_list()
    assert len(commands) == 1
    assert commands[0].texture_key == "words/gogo_ink.png"
    player.step(1.1)
    assert len(player._effects) == 0  # expired after its own `life`


def test_active_textures_includes_sheet_and_words(spec):
    player = FxPlayer(spec, seed=8)
    player.spawn("hit_kill", 0.0, 0.0, variant=0, words=True)
    keys = player.active_textures()
    assert "sheets/hit_kill_1.png" in keys
    assert "words/don_ink.png" in keys


def test_clear_removes_matching_effects(spec):
    player = FxPlayer(spec, seed=9)
    player.spawn("hit_kill", 0.0, 0.0, variant=0, words=False)
    player.spawn("ui_tap", 1.0, 1.0, variant=0, words=False)
    removed = player.clear(lambda name: name == "hit_kill")
    assert removed == ["hit_kill"]
    assert [e.name for e in player._effects] == ["ui_tap"]


def test_frame_index_matches_floor_age_times_fps(spec):
    player = FxPlayer(spec, seed=10)
    player.spawn("ui_tap", 0.0, 0.0, variant=0, words=False)
    dt = 1.0 / spec.fps
    player.step(dt * 2.6)
    expected_index = math.floor(2.6)
    frame = spec.effects["ui_tap"]["variants"][0]["frames"][expected_index]
    commands = player.draw_list()
    assert len(commands) == 1
    rx, ry, rw, rh = frame["rect"]
    sheet_w, sheet_h = spec.sheet_sizes[spec.effects["ui_tap"]["variants"][0]["sheet"]]
    expected_u0 = rx / sheet_w
    assert abs(commands[0].uv[0] - expected_u0) < 1e-9
