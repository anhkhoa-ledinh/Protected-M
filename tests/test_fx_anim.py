"""
TASK_ui_sakuga_1117 Phase 7: ``src/fx/anim.py`` is a byte-exact port of
``tools/sakuga_bake/fxref.py`` — these tests run BOTH side by side over a
realistic word lifetime and assert the outputs never diverge.
"""

import importlib.util
import sys
from pathlib import Path

from src.fx.anim import Spring, WordAnim

ROOT = Path(__file__).resolve().parents[1]


def _load_fxref():
    path = ROOT / "tools" / "sakuga_bake" / "fxref.py"
    spec = importlib.util.spec_from_file_location("fxref_reference", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules["fxref_reference"] = module
    spec.loader.exec_module(module)
    return module


def _run_both(dt, seconds=2.0):
    fxref = _load_fxref()
    ours = WordAnim(100.0, 10.0, 20.0, 150.0, -0.08, 1.4)
    theirs = fxref.WordAnim(100.0, 10.0, 20.0, 150.0, -0.08, 1.4)
    max_diff = 0.0
    t = 0.0
    while t < seconds:
        a = ours.step(dt)
        b = theirs.step(dt)
        assert (a is None) == (b is None), (t, a, b)
        if a is not None:
            for av, bv in zip(a, b):
                max_diff = max(max_diff, abs(av - bv))
        t += dt
    return max_diff


def test_word_anim_matches_reference_at_60fps():
    assert _run_both(1.0 / 60.0) < 1e-9


def test_word_anim_matches_reference_at_30fps():
    assert _run_both(1.0 / 30.0) < 1e-9


def test_spring_converges_toward_target():
    spring = Spring(0.0, k=420.0, d=26.0)
    spring.to = 1.0
    for _ in range(300):
        value = spring.step(1.0 / 60.0)
    assert abs(value - 1.0) < 0.01


def test_spring_step_zero_dt_is_noop():
    spring = Spring(5.0, k=420.0, d=26.0)
    assert spring.step(0.0) == 5.0
    assert spring.step(-1.0) == 5.0
