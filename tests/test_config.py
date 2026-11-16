"""Unit tests for configuration loading and persistence."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from src.config import AppConfig, load_config, merge_user_config, save_config


class TestConfigSystem(unittest.TestCase):
    """JSON config round-trip and merge behavior."""

    def test_default_json_loads(self) -> None:
        path = Path(__file__).resolve().parents[1] / "assets" / "config" / "default.json"
        config = load_config(path)
        self.assertEqual(config.camera.width, 1280)
        self.assertTrue(config.filter.prediction_enabled)
        self.assertTrue(config.adaptive_quality.enabled)

    def test_json_round_trip(self) -> None:
        config = AppConfig()
        config.screen.width_mm = 510.0
        config.filter.prediction_frames = 2.0

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "test.json"
            save_config(config, path)
            loaded = load_config(path)

        self.assertEqual(loaded.screen.width_mm, 510.0)
        self.assertEqual(loaded.filter.prediction_frames, 2.0)

    def test_merge_user_config(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp) / "base.json"
            user = Path(tmp) / "user.json"
            save_config(AppConfig(), base)
            save_config(
                AppConfig.from_dict({"camera": {"device_id": 2}}),
                user,
            )
            merged = merge_user_config(base, user)
            self.assertEqual(merged.camera.device_id, 2)
            self.assertEqual(merged.camera.width, 1280)

    def test_user_config_only_overrides_keys_it_actually_sets(self) -> None:
        """
        ``user.json`` chỉ được đè những khoá nó thật sự ghi.

        Dạng cũ nạp ``user.json`` thành một ``AppConfig`` đầy đủ rồi
        ``to_dict()``, nên mọi trường người dùng không ghi quay về **mặc định
        dataclass** và phủ đè lên ``default.json``. Đo 2026-10-12 trên cấu hình
        thật: ``"fullscreen": true`` trong ``default.json`` về lại ``False``,
        app không bao giờ chạy toàn màn hình dù đã bật cờ. Lỗi ngủ yên vì mọi
        khoá còn lại tình cờ trùng mặc định.
        """
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp) / "base.json"
            user = Path(tmp) / "user.json"
            # Hai giá trị KHÁC mặc định dataclass (False / 4).
            base.write_text(
                json.dumps({"renderer": {"fullscreen": True, "msaa_samples": 8}}),
                encoding="utf-8",
            )
            # user.json không nhắc gì tới renderer.
            user.write_text(
                json.dumps({"user": {"ipd_mm": 61.0}}), encoding="utf-8"
            )

            merged = merge_user_config(base, user)

            self.assertTrue(
                merged.renderer.fullscreen,
                "user.json không nhắc tới fullscreen thì không được xoá nó",
            )
            self.assertEqual(merged.renderer.msaa_samples, 8)
            self.assertEqual(merged.user.ipd_mm, 61.0, "khoá user.json vẫn phải thắng")


if __name__ == "__main__":
    unittest.main()
