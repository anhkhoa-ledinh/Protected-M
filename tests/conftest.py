"""
TASK_ci_1116: fixtures for resources GitHub Actions' ``windows-latest``
runner does not have — no GPU, no webcam, no interactive desktop, and no
local cache of the Depth Anything / LaMa weights (only
``assets/models/face_landmarker.task`` is committed to git).

Each fixture *tries* the real resource and ``pytest.skip``\\ s with a clear
reason when it is missing, instead of letting the test fail or error. A
test that genuinely needs the resource should request the matching
fixture; nothing here changes behaviour for a test that does not.

``PM_FORCE_NO_GL=1`` / ``PM_FORCE_NO_WEBCAM=1`` / ``PM_FORCE_NO_MODELS=1``
force the ``gl_context`` / ``webcam`` / ``model_files`` fixtures to skip
even on a machine that actually has the resource — used to simulate the
CI runner locally (see README/TASK_ci_1116 P1) without needing a second,
resource-less machine to prove the suite survives their absence.
"""

from __future__ import annotations

import hashlib
import os
import re
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def _config_md5s() -> dict[str, str]:
    cfg = ROOT / "assets" / "config"
    return {f.name: hashlib.md5(f.read_bytes()).hexdigest()
            for f in sorted(cfg.glob("*.json"))}


@pytest.fixture(autouse=True, scope="session")
def _real_config_untouched():
    """Fail the session if any test changed a file in ``assets/config``
    (the user's real accounts, prefs, ...). Tests must use a temp root."""
    before = _config_md5s()
    yield
    after = _config_md5s()
    changed = sorted(n for n in before.keys() | after.keys()
                     if before.get(n) != after.get(n))
    if changed:
        pytest.fail("tests touched the real config: " + ", ".join(changed),
                    pytrace=False)


@pytest.fixture(autouse=True)
def _photo_cache_in_tmp(tmp_path_factory, monkeypatch):
    """The photo gate writes its clean copies to ``HCPD_PHOTO_CACHE``: keep
    them out of the real ``assets/cache`` while the tests run."""
    monkeypatch.setenv("HCPD_PHOTO_CACHE",
                       str(tmp_path_factory.getbasetemp() / "photo_cache"))

#: Same path src/content/lama_fill.py:MODEL_PATH uses — gitignored, 208 MB,
#: never committed (see .gitignore's own comment on this file).
LAMA_ONNX_PATH = ROOT / "assets" / "models" / "lama_fp32.onnx"

#: The HF model id src/content/depth_estimator.py uses for BOTH the torch
#: and the onnxruntime backend (the latter still needs this for its image
#: processor — see _OnnxRuntimeBackend's docstring).
DEPTH_ANYTHING_MODEL_ID = "LiheYoung/depth-anything-small-hf"


#: Matches the "MAJOR.MINOR" at the start of a GL_VERSION string, e.g.
#: b"4.6.0 NVIDIA 471.11" or b"3.3.0 Mesa 21.0" or b"OpenGL ES 3.2 ...".
_GL_VERSION_RE = re.compile(rb"(\d+)\.(\d+)")


def _parse_gl_version(raw) -> tuple[int, int] | None:
    data = raw if isinstance(raw, bytes) else str(raw).encode("utf-8", "replace")
    match = _GL_VERSION_RE.search(data)
    return (int(match.group(1)), int(match.group(2))) if match else None


@pytest.fixture
def gl_context():
    """
    A hidden GLFW window with the same context hints
    ``src/renderer/window.py:GlfwWindow.create()`` requests (OpenGL 3.3,
    core profile) — skip if the runner can't create a REAL, usable one.

    Creating the window is not enough: a runner with no real GPU/driver
    (observed on GitHub Actions' ``windows-latest``) can still return a
    non-``None`` window from ``glfw.create_window()`` — the hints are a
    *request*, GLFW only warns if the driver can't honour it, it doesn't
    fail the call — while the context underneath is unusable, and
    ``glGetString(GL_VERSION)`` raises ``GLError`` 1282 ("invalid
    operation"). So this fixture also makes the probed context current and
    calls ``glGetString(GL_VERSION)`` itself, and skips if that raises,
    returns nothing, or reports a version below 3.3.
    """
    if os.environ.get("PM_FORCE_NO_GL") == "1":
        pytest.skip("PM_FORCE_NO_GL=1 (forced — simulating a runner with no "
                    "OpenGL context, e.g. GitHub Actions windows-latest)")
    import glfw

    if not glfw.init():
        pytest.skip("glfw.init() failed — no display/driver available")
    glfw.window_hint(glfw.VISIBLE, glfw.FALSE)
    glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3)
    glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
    glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)
    window = glfw.create_window(64, 64, "pytest-gl-probe", None, None)
    if window is None:
        glfw.terminate()
        pytest.skip("Could not create an OpenGL 3.3 core-profile context "
                    "(no GPU/driver support on this runner)")
    glfw.make_context_current(window)

    skip_reason = None
    try:
        from OpenGL import GL

        version_raw = GL.glGetString(GL.GL_VERSION)
    except Exception as exc:
        version_raw = None
        skip_reason = (f"glGetString(GL_VERSION) raised {exc!r} — no real "
                       "OpenGL driver on this runner")

    if skip_reason is None and version_raw is None:
        skip_reason = ("glGetString(GL_VERSION) returned nothing — no "
                       "real OpenGL driver on this runner")

    if skip_reason is None:
        parsed = _parse_gl_version(version_raw)
        if parsed is None or parsed < (3, 3):
            version_text = (version_raw.decode("utf-8", "replace")
                            if isinstance(version_raw, bytes) else str(version_raw))
            skip_reason = f"OpenGL version {version_text!r} is below the required 3.3"

    if skip_reason is not None:
        glfw.destroy_window(window)
        glfw.terminate()
        pytest.skip(skip_reason)

    try:
        yield window
    finally:
        glfw.destroy_window(window)
        glfw.terminate()


@pytest.fixture
def webcam():
    """
    Device 0, opened the same way ``src/camera/capture.py:CameraCapture.open()``
    does — skip if the runner has no camera.
    """
    if os.environ.get("PM_FORCE_NO_WEBCAM") == "1":
        pytest.skip("PM_FORCE_NO_WEBCAM=1 (forced — simulating a runner with "
                    "no webcam, e.g. GitHub Actions windows-latest)")
    import cv2

    backend = cv2.CAP_DSHOW if sys.platform == "win32" else cv2.CAP_ANY
    cap = cv2.VideoCapture(0, backend)
    if not cap.isOpened():
        cap.release()
        cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        cap.release()
        pytest.skip("Could not open webcam device 0 (no camera on this runner)")
    try:
        yield cap
    finally:
        cap.release()


def _depth_anything_cached(model_id: str = DEPTH_ANYTHING_MODEL_ID) -> bool:
    """
    True if the HF snapshot for ``model_id`` is already on disk.

    ``local_files_only=True`` means this NEVER touches the network, even
    when ``HF_HUB_OFFLINE`` is unset — so calling this is always safe, on
    CI or anywhere else.
    """
    try:
        from transformers import AutoConfig

        AutoConfig.from_pretrained(model_id, local_files_only=True)
        return True
    except Exception:
        return False


class ModelFiles:
    """
    What real, not-committed model weight is actually on this machine
    right now. ``face_landmarker.task`` is committed to git and always
    present, so it is out of scope here.

    A single ``model_files`` fixture covers two independent resources
    (Depth Anything's HF cache, LaMa's ONNX export) — call the
    ``require_*`` method for whichever one a test actually needs; it
    skips with a clear reason if that one is missing, leaving the other
    resource's presence irrelevant to that test.

    ``PM_FORCE_NO_MODELS=1`` reports both as missing even on a machine
    that has them, to simulate the CI runner's empty model cache locally.
    """

    def __init__(self) -> None:
        if os.environ.get("PM_FORCE_NO_MODELS") == "1":
            self.lama_onnx = None
            self.depth_anything_cached = False
        else:
            self.lama_onnx = LAMA_ONNX_PATH if LAMA_ONNX_PATH.exists() else None
            self.depth_anything_cached = _depth_anything_cached()

    def require_depth_anything(self) -> None:
        if not self.depth_anything_cached:
            pytest.skip(
                f"Depth Anything weights ({DEPTH_ANYTHING_MODEL_ID}) are not "
                "cached locally, and CI must never download them")

    def require_lama(self) -> None:
        if self.lama_onnx is None:
            pytest.skip(f"LaMa weights not found at {LAMA_ONNX_PATH} "
                        "(gitignored, 208 MB, never committed)")


@pytest.fixture
def model_files() -> ModelFiles:
    return ModelFiles()
