"""TASK_perf_textures_1119 Fix 3: ``AsyncTextureLoader`` (pure threading, no
GL) — request/poll, priority ordering, dedup, and that it never decodes
the SAME key twice while a job for it is in flight.
"""
from __future__ import annotations

import time

import numpy as np
import pytest

from src.ui_album.texture_loader import (PRIORITY_CURRENT, PRIORITY_PREFETCH,
                                         AsyncTextureLoader)


@pytest.fixture
def loader():
    instance = AsyncTextureLoader()
    yield instance
    instance.stop()


def _save_png(path, size=(9, 9)):
    from PIL import Image
    array = np.zeros((size[1], size[0], 4), dtype=np.uint8)
    array[0, 0] = [255, 0, 0, 255]
    Image.fromarray(array, "RGBA").save(path)


def test_request_then_poll_returns_the_decoded_array(loader, tmp_path):
    path = tmp_path / "a.png"
    _save_png(path)
    loader.request(("a",), path, 0)
    deadline = time.time() + 5.0
    results = []
    while not results and time.time() < deadline:
        results = loader.poll()
        time.sleep(0.005)
    assert len(results) == 1
    key, array = results[0]
    assert key == ("a",)
    assert array is not None
    assert array.shape == (9, 9, 4)


def test_poll_is_empty_before_any_request(loader):
    assert loader.poll() == []


def test_duplicate_request_for_the_same_key_is_a_no_op_while_pending(loader, tmp_path, monkeypatch):
    path = tmp_path / "a.png"
    _save_png(path)
    calls = []
    import src.ui_album.texture_loader as mod
    real_load = mod.load_rgba

    def counting_load(p, max_width=0):
        calls.append(p)
        time.sleep(0.05)      # keep it "in flight" long enough to race the dup request
        return real_load(p, max_width=max_width)

    monkeypatch.setattr(mod, "load_rgba", counting_load)
    loader.request(("a",), path, 0)
    assert loader.is_pending(("a",))
    loader.request(("a",), path, 0)       # duplicate, still pending -> no-op
    deadline = time.time() + 5.0
    results = []
    while not results and time.time() < deadline:
        results = loader.poll()
        time.sleep(0.005)
    assert len(calls) == 1
    assert len(results) == 1


def test_is_pending_false_once_the_job_is_delivered(loader, tmp_path):
    path = tmp_path / "a.png"
    _save_png(path)
    loader.request(("a",), path, 0)
    deadline = time.time() + 5.0
    results = []
    while not results and time.time() < deadline:
        results = loader.poll()
        time.sleep(0.005)
    assert not loader.is_pending(("a",))


def test_missing_file_delivers_none_not_an_exception(loader, tmp_path):
    loader.request(("missing",), tmp_path / "nope.png", 0)
    deadline = time.time() + 5.0
    results = []
    while not results and time.time() < deadline:
        results = loader.poll()
        time.sleep(0.005)
    assert results == [(("missing",), None)]


def test_priority_constants_are_ordered_current_before_prefetch():
    assert PRIORITY_CURRENT < PRIORITY_PREFETCH
