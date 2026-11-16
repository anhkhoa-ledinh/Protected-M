"""TASK_ci_1116: proves the three resource fixtures in conftest.py actually
work — both the happy path (this dev machine has GL, a webcam and the
models, so these must PASS, not skip) and the forced/missing path (what a
resource-less CI runner sees)."""

from __future__ import annotations

import pytest

from tests.conftest import (
    DEPTH_ANYTHING_MODEL_ID,
    LAMA_ONNX_PATH,
    ModelFiles,
    _parse_gl_version,
)


# ---- gl_context -------------------------------------------------------

def test_gl_context_creates_a_current_opengl_context(gl_context):
    import glfw
    from OpenGL import GL

    assert glfw.get_current_context() is not None
    version = GL.glGetString(GL.GL_VERSION)
    assert version, "glGetString(GL_VERSION) returned nothing"


def test_gl_context_is_skipped_when_forced(monkeypatch, request):
    monkeypatch.setenv("PM_FORCE_NO_GL", "1")
    with pytest.raises(pytest.skip.Exception, match="PM_FORCE_NO_GL"):
        request.getfixturevalue("gl_context")


def test_gl_context_skips_when_glgetstring_raises(monkeypatch, request):
    """TASK_ci_1116 P3 round 1 CI failure: the runner creates a non-None
    window (GLFW only warns, it doesn't fail the call) over a context that
    isn't actually usable — glGetString(GL_VERSION) raised GLError 1282
    there. The fixture must probe this itself and skip, not let the error
    propagate into whatever test requested the fixture."""
    # Clears any ambient PM_FORCE_NO_GL=1 (e.g. from a G2-style simulation
    # run) first — this test's own glGetString failure must be what
    # triggers the skip, not an unrelated global simulation flag that
    # would skip even earlier, before glGetString is ever called.
    monkeypatch.delenv("PM_FORCE_NO_GL", raising=False)
    import OpenGL.GL as GL

    def boom(name):
        raise RuntimeError("simulated: GLError 1282 (invalid operation)")

    monkeypatch.setattr(GL, "glGetString", boom)
    with pytest.raises(pytest.skip.Exception, match="no real OpenGL driver"):
        request.getfixturevalue("gl_context")


def test_gl_context_skips_when_version_is_below_3_3(monkeypatch, request):
    monkeypatch.delenv("PM_FORCE_NO_GL", raising=False)
    import OpenGL.GL as GL

    monkeypatch.setattr(GL, "glGetString", lambda name: b"3.1.0 Mesa 20.0")
    with pytest.raises(pytest.skip.Exception, match=r"3\.1\.0 Mesa 20\.0"):
        request.getfixturevalue("gl_context")


@pytest.mark.parametrize("raw,expected", [
    (b"4.6.0 NVIDIA 471.11", (4, 6)),
    (b"3.3.0 Mesa 21.0", (3, 3)),
    (b"3.1.0 Mesa 20.0", (3, 1)),
    ("4.1 INTEL-14.4.21", (4, 1)),
    (b"OpenGL ES 3.2 Mesa 22.0", (3, 2)),
    (b"garbage, no version here", None),
    (None, None),
])
def test_parse_gl_version(raw, expected):
    assert _parse_gl_version(raw) == expected


# ---- webcam -------------------------------------------------------------

def test_webcam_opens_and_reads_a_frame(webcam):
    assert webcam.isOpened()
    ok, frame = webcam.read()
    assert ok and frame is not None


def test_webcam_is_skipped_when_forced(monkeypatch, request):
    monkeypatch.setenv("PM_FORCE_NO_WEBCAM", "1")
    with pytest.raises(pytest.skip.Exception, match="PM_FORCE_NO_WEBCAM"):
        request.getfixturevalue("webcam")


# ---- model_files ----------------------------------------------------------

def test_model_files_reports_lama_matching_real_file_presence(monkeypatch, request):
    """Must reflect whatever is ACTUALLY on disk, not assume it's present —
    lama_fp32.onnx is gitignored, so it exists on this dev machine but not
    on CI (TASK_ci_1116 P3 round 1 CI failure: this hard-asserted presence
    and failed there). Clears any ambient PM_FORCE_NO_MODELS=1 first, same
    reason as the gl_context tests above — this checks the genuine,
    non-forced path."""
    monkeypatch.delenv("PM_FORCE_NO_MODELS", raising=False)
    model_files = request.getfixturevalue("model_files")
    assert model_files.lama_onnx == (
        LAMA_ONNX_PATH if LAMA_ONNX_PATH.exists() else None)


def test_model_files_forced_missing_reports_both_absent(monkeypatch):
    monkeypatch.setenv("PM_FORCE_NO_MODELS", "1")
    files = ModelFiles()
    assert files.lama_onnx is None
    assert files.depth_anything_cached is False
    with pytest.raises(pytest.skip.Exception, match="LaMa"):
        files.require_lama()
    with pytest.raises(pytest.skip.Exception, match=DEPTH_ANYTHING_MODEL_ID):
        files.require_depth_anything()


def test_model_files_depth_anything_flag_matches_the_real_cache_state():
    """``depth_anything_cached`` must reflect reality, whatever it is: True
    normally on this machine (G1), False under P1's HF_HUB_OFFLINE=1 +
    empty HF_HOME simulation (G2) — and ``require_depth_anything()`` must
    agree with its own flag either way. Deliberately NOT a hardcoded
    ``is True``, so this test passes under both G1 and G2 instead of
    failing the moment the environment is the thing being tested."""
    files = ModelFiles()
    if files.depth_anything_cached:
        files.require_depth_anything()  # must not skip
    else:
        with pytest.raises(pytest.skip.Exception, match=DEPTH_ANYTHING_MODEL_ID):
            files.require_depth_anything()


def test_require_depth_anything_skips_with_a_clear_reason_when_missing(model_files):
    model_files.depth_anything_cached = False
    with pytest.raises(pytest.skip.Exception, match=DEPTH_ANYTHING_MODEL_ID):
        model_files.require_depth_anything()


def test_require_lama_skips_with_a_clear_reason_when_missing(model_files):
    model_files.lama_onnx = None
    with pytest.raises(pytest.skip.Exception, match="LaMa"):
        model_files.require_lama()


def test_require_methods_are_silent_when_the_resource_is_present():
    files = ModelFiles()
    files.depth_anything_cached = True
    files.lama_onnx = LAMA_ONNX_PATH
    files.require_depth_anything()  # must not raise/skip
    files.require_lama()            # must not raise/skip
