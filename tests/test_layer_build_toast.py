"""
PB-076: dựng lớp hoãn ĐÚNG MỘT khung để toast kịp lên màn.

Dựng lớp chạy trên luồng GL nên cửa sổ đứng hình 9–30 s; toast đặt ngay trước
đó chỉ hiện ra *sau* khi xong. Hoãn một khung thì khung mang toast được vẽ và
swap trước, nên lời nhắn nằm trên màn suốt lúc đóng băng.

Hai điều phải giữ: khung hoãn **không** dựng gì (test này thất bại trên code
cũ), và khung sau dựng với **đúng** bản đồ depth cũ — không bản sao, không
đổi byte nào, nên bộ lớp y hệt trước khi hoãn.

Dựng instance bằng ``object.__new__`` như ``tests/test_depth_sweep.py``: cả
``_poll_depth_mesh`` chỉ đụng vài thuộc tính.
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
from types import SimpleNamespace

from src.config.schema import ContentSettings
from src.content.depth_worker import DepthWorker
from src.renderer.content_plane import LAYER_BUILD_TOAST, ContentPlaneRenderer

DEPTH = np.linspace(0.0, 1.0, 12, dtype=np.float32).reshape(3, 4)
_BAKE = SimpleNamespace(depth=DEPTH, k=3, layers=[])


class _Worker:
    """Worker tối thiểu: giao kết quả đúng một lần."""

    failed = False

    def __init__(self, payload) -> None:
        self._payload = payload
        self.pops = 0

    def poll(self) -> None:
        pass

    def pop_ready(self, _key):
        self.pops += 1
        payload, self._payload = self._payload, None
        return payload


def _plane(tmp_path, clustering: bool = True):
    """Content plane tối thiểu cho ``_poll_depth_mesh``, ghi lại lượt dựng."""
    image = tmp_path / "t.png"
    image.write_bytes(b"not-a-real-png")
    plane = object.__new__(ContentPlaneRenderer)
    content = ContentSettings()
    content.layer_clustering_enabled = clustering
    content.depth_enabled = True
    plane._content = content
    plane._current_path = image
    # L1: chế độ lớp đi qua ``BakeWorker`` (tiến trình con), chế độ mesh vẫn
    # qua ``DepthWorker``. Hai đường, hai worker — đúng như app dựng chúng.
    worker = _Worker(_BAKE if clustering else DEPTH)
    plane._bake_worker = worker if clustering else None
    plane._depth_worker = None if clustering else worker
    plane._depth_mesh_key = None
    plane._deferred_layers = None
    plane._build_notice = None
    plane.built = []
    plane._apply_bake = lambda result, key: plane.built.append(
        ("layers", result.depth, key))
    plane._apply_depth_mesh = lambda depth, key: plane.built.append(("mesh", depth, key))
    return plane, image


class DeferOneFrameTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.addCleanup(self.tmp.cleanup)

    def test_first_poll_builds_nothing_and_posts_the_toast(self) -> None:
        plane, _ = _plane(self.root)
        plane._poll_depth_mesh()
        self.assertEqual(plane.built, [])            # thất bại trên code cũ
        self.assertEqual(plane.take_build_notice(), LAYER_BUILD_TOAST)

    def test_second_poll_builds_with_the_same_depth_bytes(self) -> None:
        plane, image = _plane(self.root)
        plane._poll_depth_mesh()
        plane.take_build_notice()    # khung hoãn: renderer đã lấy để hiện toast
        plane._poll_depth_mesh()
        self.assertEqual([row[0] for row in plane.built], ["layers"])
        _kind, depth, key = plane.built[0]
        self.assertEqual(depth.tobytes(), DEPTH.tobytes())
        self.assertEqual(key, DepthWorker.media_key(image))
        self.assertIsNone(plane.take_build_notice())  # toast chỉ một lần
        self.assertEqual(plane._bake_worker.pops, 1)  # không lấy lại từ worker

    def test_mesh_mode_is_not_deferred(self) -> None:
        # Không tách lớp thì không có cửa sổ đóng băng dài; giữ như cũ.
        plane, _ = _plane(self.root, clustering=False)
        plane._poll_depth_mesh()
        self.assertEqual([row[0] for row in plane.built], ["mesh"])
        self.assertIsNone(plane.take_build_notice())

    def test_toast_is_ascii_only(self) -> None:
        # Toast vẽ bằng FONT_HERSHEY_SIMPLEX: ký tự ngoài ASCII thành ô vuông.
        self.assertTrue(LAYER_BUILD_TOAST.isascii())


if __name__ == "__main__":
    unittest.main()
