"""
TASK_game_1110 L1 — nướng lớp ngoài tiến trình, có cache.

Ba điều phải giữ, và cả ba đều là lời hứa về **kiến trúc** chứ không phải về
một con số:

1. Ở chế độ lớp, app dựng ``BakeWorker`` chứ **không** dựng ``DepthWorker``:
   không luồng nào của app còn đụng onnxruntime. Đúng cái đã giết hai lượt
   chơi ngày 07/11 (mã thoát −805306369, không traceback).
2. Khoá cache đổi theo đúng những khoá có ảnh hưởng tới bước nướng, và
   **không** đổi theo khoá chỉ dùng lúc vẽ (``layer_z_far_m``, overscan…).
   Nhờ vế sau mà lượt quét z của L4 không phải nướng lại bốn lần.
3. Ghi rồi đọc lại cho đúng bộ lớp cũ, từng byte alpha.
"""

from __future__ import annotations

import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

import numpy as np

from src.config.schema import ContentSettings
from src.content import bake as B

ROOT = Path(__file__).resolve().parents[1]


def _content(**over) -> ContentSettings:
    data = json.loads((ROOT / "assets" / "config" / "content.demo.json")
                      .read_text(encoding="utf-8"))
    data.update(over)
    return ContentSettings.from_dict(data)


class SettingsHashTest(unittest.TestCase):
    def test_draw_time_keys_do_not_change_the_hash(self) -> None:
        base = _content()
        for key, value in (("layer_z_far_m", -0.22),
                           ("overscan_e_max_mm", 0.0),
                           ("relief_enabled", False),
                           ("mist_enabled", True),
                           ("depth_mesh_resolution", 32)):
            with self.subTest(key=key):
                self.assertEqual(B.settings_hash(base),
                                 B.settings_hash(replace(base, **{key: value})))

    def test_bake_time_keys_do_change_the_hash(self) -> None:
        base = _content()
        for key, value in (("fill_method", "telea"),
                           ("plate_fill", "lama"),
                           ("edge_style", "soft"),
                           ("far_fog_mode", "atmo"),
                           ("far_desaturation", 0.3),
                           ("adaptive_k_max", 5),
                           ("layer_seed", 7)):
            with self.subTest(key=key):
                self.assertNotEqual(B.settings_hash(base),
                                    B.settings_hash(replace(base, **{key: value})))

    def test_every_bake_key_exists_on_content_settings(self) -> None:
        # Gõ sai tên một khoá thì ``getattr`` trả None lặng lẽ và khoá cache
        # mất đi một chiều — không bao giờ thấy, trừ khi chốt ở đây.
        known = set(ContentSettings().__dict__)
        missing = [k for k in B.BAKE_KEYS if k not in known]
        self.assertEqual(missing, [], f"khoa la trong BAKE_KEYS: {missing}")


class CacheRoundTripTest(unittest.TestCase):
    """Ghi rồi đọc lại phải ra đúng bộ lớp cũ."""

    def test_write_then_load_keeps_every_alpha_byte(self) -> None:
        from src.content.layer_clustering import cluster_into_layers

        rng = np.random.default_rng(3)
        rgb = rng.integers(0, 255, (48, 64, 3), dtype=np.uint8)
        depth = np.tile(np.linspace(0.0, 1.0, 64, dtype=np.float32), (48, 1))
        cluster = cluster_into_layers(rgb, depth, n_clusters=3,
                                      disocclusion=None)
        result = B.result_from_cluster(cluster, depth, 3)
        meta = {"format": B.CACHE_FORMAT, "k": 3, "bake_s": 1.5,
                "mist_color": list(cluster.mist_color), "timing_ms": {},
                "layers": [{"index": i, "mean_depth": l.mean_depth,
                            "depth_lo": l.depth_lo, "depth_hi": l.depth_hi,
                            "coverage": l.coverage, "depth_norm": l.depth_norm,
                            "hole_pct": l.hole_pct, "grade": None}
                           for i, l in enumerate(result.layers)]}
        with tempfile.TemporaryDirectory() as tmp:
            cache_dir = Path(tmp) / "maps" / "abc"
            B._write_cache(cache_dir, result, depth, meta)
            back = B.load_from_cache(cache_dir)
        self.assertIsNotNone(back)
        self.assertEqual(back.k, 3)
        self.assertTrue(back.from_cache)
        self.assertEqual(len(back.layers), len(result.layers))
        for want, got in zip(result.layers, back.layers):
            self.assertEqual(want.rgba.tobytes(), got.rgba.tobytes())
            self.assertEqual(want.mask.tobytes(), got.mask.tobytes())
        self.assertTrue(np.array_equal(depth, back.depth))

    def test_a_missing_directory_is_a_miss_not_a_crash(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            self.assertIsNone(B.load_from_cache(Path(tmp) / "khong-co"))

    def test_a_newer_format_number_invalidates_old_entries(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            cache_dir = Path(tmp) / "abc"
            cache_dir.mkdir()
            (cache_dir / "meta.json").write_text(
                json.dumps({"format": B.CACHE_FORMAT - 1}), encoding="utf-8")
            self.assertIsNone(B.load_from_cache(cache_dir))


class NoOnnxInTheGlThreadTest(unittest.TestCase):
    """Chế độ lớp dựng ``BakeWorker``; chế độ mesh vẫn dùng ``DepthWorker``."""

    def _plane(self, clustering: bool):
        from src.renderer import content_plane as cp

        plane = cp.ContentPlaneRenderer.__new__(cp.ContentPlaneRenderer)
        content = _content(layer_clustering_enabled=clustering)
        plane._content = content
        plane._bake_worker = None
        plane._last_bake = None
        plane._depth_worker = None
        if content.depth_enabled and content.layer_clustering_enabled:
            plane._bake_worker = B.BakeWorker(ROOT)
        elif content.depth_enabled:
            from src.content.depth_worker import DepthWorker

            plane._depth_worker = DepthWorker(content.depth_model,
                                              content.depth_backend)
        return plane

    def test_layer_mode_uses_the_child_process_worker(self) -> None:
        plane = self._plane(True)
        self.assertIsInstance(plane._bake_worker, B.BakeWorker)
        self.assertIsNone(plane._depth_worker)

    def test_mesh_mode_keeps_the_old_thread_worker(self) -> None:
        plane = self._plane(False)
        self.assertIsNone(plane._bake_worker)
        self.assertIsNotNone(plane._depth_worker)

    def test_the_child_entry_point_is_a_module_level_function(self) -> None:
        # ``spawn`` phải pickle được target; một closure hay method sẽ nổ ở
        # Windows với "Can't pickle local object" đúng lúc mở ảnh đầu tiên.
        import pickle

        self.assertEqual(pickle.loads(pickle.dumps(B._child_main)),
                         B._child_main)


if __name__ == "__main__":
    unittest.main()
