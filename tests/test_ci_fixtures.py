"""TASK_ci_1116: proves the three resource fixtures in conftest.py actually
work — both the happy path (this dev machine has GL, a webcam and the
models, so these must PASS, not skip) and the forced/missing path (what a
resource-less CI runner sees)."""

from __future__ import annotations

import pytest

from tests.conftest import DEPTH_ANYTHING_MODEL_ID, LAMA_ONNX_PATH, ModelFiles


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

def test_model_files_reports_lama_present_on_this_machine(model_files):
    # Filesystem-only check, unaffected by TASK_ci_1116 P1's HF_HOME
    # simulation — true on this dev machine (G1's "0 new skips").
    assert model_files.lama_onnx == LAMA_ONNX_PATH


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
