r"""
TASK_game_1110 L2 — ``content.game.json``: hồ sơ riêng cho ván chơi.

``content.demo.json`` (bản trình chiếu ảnh) phải đứng yên — không một khoá
nào trong nó được đổi vì có ``content.game.json``. Test này chốt cả hai chiều:
giá trị khởi đầu của L2 đúng như kịch bản, và ``content.demo.json`` chưa bị
đụng tới.
"""

from __future__ import annotations

import json
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from src.config.schema import ContentSettings

ROOT = Path(__file__).resolve().parents[1]
GAME_PATH = ROOT / "assets" / "config" / "content.game.json"
DEMO_PATH = ROOT / "assets" / "config" / "content.demo.json"


def _game_payload() -> dict:
    return json.loads(GAME_PATH.read_text(encoding="utf-8"))


class GameProfileValuesTest(unittest.TestCase):
    def test_the_file_exists_and_parses(self) -> None:
        self.assertTrue(GAME_PATH.exists())
        settings = ContentSettings.from_dict(_game_payload())
        self.assertTrue(settings.layer_clustering_enabled)

    def test_l2_starting_values(self) -> None:
        data = _game_payload()
        self.assertEqual(data["edge_style"], "soft")
        self.assertFalse(data["relief_enabled"])
        self.assertEqual(data["far_fog_mode"], "atmo")
        self.assertLessEqual(data["far_desaturation"], 0.3)
        self.assertLessEqual(data["atmosphere_blend_max"], 0.1)
        self.assertEqual(data["fill_method"], "lama")
        self.assertEqual(data["plate_fill"], "lama")

    def test_edge_alpha_is_refined_by_the_guided_filter(self) -> None:
        data = _game_payload()
        self.assertTrue(data.get("edge_guided_refine"))
        self.assertIn(data.get("edge_guided_feather_px"), (2, 3))

    def test_l3_back_layers_dilate_the_hole_against_fringe(self) -> None:
        data = _game_payload()
        self.assertEqual(data.get("disocclusion_fringe_dilate_px"), 6)

    def test_l4_starts_at_the_interim_parallax_budget(self) -> None:
        # L4: cho tới khi người dùng chọn, dùng -0.22 (kịch bản).
        data = _game_payload()
        self.assertEqual(data["layer_z_far_m"], -0.22)


class DemoProfileUntouchedTest(unittest.TestCase):
    """``content.demo.json`` không đổi vì L2 — đúng lời hứa của kịch bản."""

    def test_demo_config_keeps_its_own_decisions(self) -> None:
        demo = json.loads(DEMO_PATH.read_text(encoding="utf-8"))
        self.assertEqual(demo["edge_style"], "diecut")
        self.assertTrue(demo["relief_enabled"])
        self.assertEqual(demo["far_fog_mode"], "complement")
        self.assertEqual(demo["layer_z_far_m"], -0.6)
        self.assertNotIn("edge_guided_refine", demo)
        self.assertNotIn("disocclusion_fringe_dilate_px", demo)


class LoadContentPicksTheGameProfileTest(unittest.TestCase):
    """``--album`` đọc ``content.game.json``, thắng cả khi ``--demo`` cũng bật."""

    @staticmethod
    def _app(root: Path, *, demo: bool, album_enabled: bool):
        from src.engine.app import HcpdApplication
        from src.ui_album.options import AlbumOptions

        app = HcpdApplication.__new__(HcpdApplication)
        app._demo = demo
        app._album_options = AlbumOptions(enabled=album_enabled)
        return app._load_content(root)

    def _tmp_configs(self, tmp: Path) -> Path:
        cfg = tmp / "assets" / "config"
        cfg.mkdir(parents=True)
        for name, far in (("content.json", -0.1), ("content.demo.json", -0.2),
                          ("content.game.json", -0.3)):
            (cfg / name).write_text(json.dumps({"layer_z_far_m": far}),
                                    encoding="utf-8")
        return tmp

    def test_album_wins_over_demo(self) -> None:
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            root = self._tmp_configs(Path(tmp))
            content = self._app(root, demo=True, album_enabled=True)
            self.assertEqual(content.layer_z_far_m, -0.3)

    def test_album_alone_also_reads_the_game_profile(self) -> None:
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            root = self._tmp_configs(Path(tmp))
            content = self._app(root, demo=False, album_enabled=True)
            self.assertEqual(content.layer_z_far_m, -0.3)

    def test_demo_alone_is_unchanged(self) -> None:
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            root = self._tmp_configs(Path(tmp))
            content = self._app(root, demo=True, album_enabled=False)
            self.assertEqual(content.layer_z_far_m, -0.2)

    def test_neither_flag_reads_the_plain_config(self) -> None:
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            root = self._tmp_configs(Path(tmp))
            content = self._app(root, demo=False, album_enabled=False)
            self.assertEqual(content.layer_z_far_m, -0.1)


if __name__ == "__main__":
    unittest.main()
