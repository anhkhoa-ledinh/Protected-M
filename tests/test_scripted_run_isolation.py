"""TASK_perf_textures_1119 Fix 8 / TASK_account_persist_1121: a scripted run
(``--ui-shots``, or any ``HcpdApplication.HOOK_ENV_VARS`` hook) must NEVER
write the user's real ``assets/config/*.json`` files.

Found via a real violation while running TASK_perf_textures_1119: ``_shutdown()``
saved ``user_prefs.json`` on every exit, no matter the mode, clobbering
the real ``window_rect``/``display`` with a hidden ``--ui-shots`` window's
geometry.

TASK_account_persist_1121: ``is_scripted_run`` used to also count ``--demo``
as scripted. That was the wrong requirement — ``run_game.bat`` ran
``--demo --album``, so EVERY real game launch was classified "scripted",
redirected its account/session store to a fresh temp dir, and skipped the
exit-time saves. Accounts, "remember me" sessions and window prefs never
reached the real files. Demo and album are both real, interactive modes;
neither makes a run scripted on its own. Only the shot/automatch sequencers
and the bench/evidence env-var hooks do.

Two layers of test:
- pure-logic: ``is_scripted_run``/``HcpdApplication._is_scripted_run``
  correctly classify every mode this project actually has a flag/env var
  for;
- end-to-end: a real ``--ui-shots`` subprocess run, with every
  ``assets/config/*.json`` file hashed before and after -- the ONLY way
  to prove the real entry point (``main.py``) never touches them is to
  run it for real and check; this is NOT one of the "use a temp config
  dir" hooks CLAUDE.local.md means (those drive ``V3UI`` directly against
  an isolated tmp_path) -- it is the regression test that would have
  caught this session's own real-file writes before they happened.
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

from src.engine.app import HOOK_ENV_VARS, HcpdApplication, is_scripted_run
from src.ui_album.options import AlbumOptions

ROOT = Path(__file__).resolve().parents[1]


# ---- pure logic: is_scripted_run / _is_scripted_run -----------------------

def test_demo_alone_is_not_scripted(monkeypatch):
    """TASK_account_persist_1121: the root cause of the vanishing-accounts
    bug -- ``--demo`` must not make a run scripted any more."""
    for var in HOOK_ENV_VARS:
        monkeypatch.delenv(var, raising=False)
    assert not is_scripted_run(shots=False, ui_shots=False)


def test_album_shots_is_scripted():
    assert is_scripted_run(shots=True, ui_shots=False)


def test_ui_shots_is_scripted():
    assert is_scripted_run(shots=False, ui_shots=True)


def test_plain_interactive_run_is_not_scripted(monkeypatch):
    for var in HOOK_ENV_VARS:
        monkeypatch.delenv(var, raising=False)
    assert not is_scripted_run(shots=False, ui_shots=False)


@pytest.mark.parametrize("var", HOOK_ENV_VARS)
def test_every_hook_env_var_makes_a_run_scripted(monkeypatch, var):
    for other in HOOK_ENV_VARS:
        monkeypatch.delenv(other, raising=False)
    monkeypatch.setenv(var, "1")
    assert is_scripted_run(shots=False, ui_shots=False)


def test_is_scripted_run_method_ignores_demo_and_album(monkeypatch):
    """TASK_account_persist_1121: ``--demo --album`` (what ``run_game.bat``
    used to run) must NOT be scripted -- this is exactly the combination
    that redirected accounts/session to a temp dir on every real launch."""
    for var in HOOK_ENV_VARS:
        monkeypatch.delenv(var, raising=False)
    app = HcpdApplication.__new__(HcpdApplication)
    app._demo = True
    app._album_options = AlbumOptions(enabled=True, ui="v3")
    assert app._is_scripted_run() is False
    app._demo = False
    assert app._is_scripted_run() is False
    monkeypatch.setenv("HCPD_FX_BENCH", "1")
    assert app._is_scripted_run() is True


def test_is_scripted_run_method_still_catches_shots_and_ui_shots(monkeypatch):
    for var in HOOK_ENV_VARS:
        monkeypatch.delenv(var, raising=False)
    app = HcpdApplication.__new__(HcpdApplication)
    app._demo = False
    app._album_options = AlbumOptions(enabled=True, ui="v3", ui_shots=True)
    assert app._is_scripted_run() is True
    app._album_options = AlbumOptions(enabled=True, ui="v3", shots=True)
    assert app._is_scripted_run() is True


# ---- end-to-end: real subprocess, real config dir, NEVER the real one ----

def test_ui_shots_subprocess_leaves_config_byte_identical():
    """Runs the REAL ``main.py`` from the REAL repo root -- every
    ``assets/config/*.json`` file's bytes are read before and compared
    after, so this fails loudly if the subprocess touched any of them."""
    config_dir = ROOT / "assets" / "config"
    before = {}
    for path in sorted(config_dir.glob("*.json")):
        before[path.name] = path.read_bytes()

    env = dict(os.environ)
    env.pop("HCPD_NO_FX", None)
    result = subprocess.run(
        [sys.executable, "main.py", "--album", "--ui", "v3", "--ui-shots",
         "--no-webcam"],
        cwd=ROOT, env=env, capture_output=True, text=True, timeout=150)

    after = {}
    for path in sorted(config_dir.glob("*.json")):
        after[path.name] = path.read_bytes()

    changed = [name for name in before if before.get(name) != after.get(name)]
    new_files = [name for name in after if name not in before]
    assert not changed, (f"these config files changed: {changed}\n"
                         f"stdout tail: {result.stdout[-2000:]}\n"
                         f"stderr tail: {result.stderr[-2000:]}")
    assert not new_files, f"new config files appeared: {new_files}"
