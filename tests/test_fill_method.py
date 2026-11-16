"""PB-074: key ``fill_method`` cho bước lấp khe (telea mặc định | fsr)."""

from __future__ import annotations

import hashlib
import json
import time
import unittest
from dataclasses import replace
from pathlib import Path

import cv2
import numpy as np

from src.config.schema import ContentSettings
from src.content.disocclusion import DisocclusionParams, inpaint_hole
from src.content.layer_clustering import (LayerEdgeParams, LayerStagingParams,
                                          cluster_into_layers)

ROOT = Path(__file__).resolve().parents[1]

#: Vân tay của ĐẦU RA HÔM NAY, chụp ở `d436f57` TRƯỚC khi thêm ``fill_method``.
GOLDEN_TELEA = ("54304d378dc40de9e0ec3480eacaa8fc",
                "01d6ec895ad8aba6bb45b0ea9db4e19c",
                "8d62f1dfea18af8658e4b485a86550c9")


def _scene():
    rng = np.random.default_rng(1)
    rgb = rng.integers(0, 256, (90, 120, 3)).astype(np.uint8)
    ramp = np.linspace(0, 1, 90)[:, None] * np.ones((1, 120))
    depth = (ramp * 0.8 + rng.random((90, 120)) * 0.2).astype(np.float32)
    return rgb, depth


def _layers(method):
    rgb, depth = _scene()
    return cluster_into_layers(
        rgb, depth, n_clusters=3, seed=0, staging=LayerStagingParams(),
        edge=LayerEdgeParams(),
        disocclusion=DisocclusionParams(fill_method=method)).layers


def _hole_scene():
    """Ảnh có vân rõ + một lỗ vuông, để kiểm hướng của mask."""
    rng = np.random.default_rng(7)
    rgb = rng.integers(0, 256, (64, 64, 3)).astype(np.uint8)
    hole = np.zeros((64, 64), dtype=np.uint8)
    hole[24:40, 24:40] = 255
    return rgb, hole


class TestTeleaIsByteIdentical(unittest.TestCase):
    """Mặc định ``"telea"`` không được đổi một byte nào."""

    def test_default_is_telea(self) -> None:
        self.assertEqual(DisocclusionParams().fill_method, "telea")
        self.assertEqual(ContentSettings().fill_method, "telea")
        self.assertEqual(ContentSettings.from_dict({}).fill_method, "telea")

    def test_telea_matches_golden(self) -> None:
        for layer, want in zip(_layers("telea"), GOLDEN_TELEA):
            got = hashlib.sha256(layer.rgba.tobytes()).hexdigest()[:32]
            self.assertEqual(got, want, f"lop {layer.index}")

    def test_committed_demo_config_matches_the_user_decision(self) -> None:
        """
        Demo config để ``"lama"`` vì **người dùng quyết** (31/10 23:26): chấp
        nhận chờ lâu ở ảnh mới để đổi lấy chất lượng 4/6 của PB-074. Tiêu chí
        thời gian của PB-074 bị **miễn**, không phải được chỉnh lại.

        Mặc định của CODE vẫn là ``"telea"`` — test trên đã giữ phần đó. Trước
        quyết định này, test đây chốt demo config phải là ``"telea"`` để không
        ai lặng lẽ đổi thứ người dùng nhìn thấy.
        """
        data = json.loads((ROOT / "assets/config/content.demo.json")
                          .read_text(encoding="utf-8"))
        self.assertEqual(data.get("fill_method", "telea"), "lama")
        self.assertEqual(data.get("plate_fill", "telea"), "telea")

    def test_unknown_method_falls_back_to_telea(self) -> None:
        rgb, hole = _hole_scene()
        np.testing.assert_array_equal(inpaint_hole(rgb, hole, 5, "khong-biet"),
                                      inpaint_hole(rgb, hole, 5, "telea"))


class TestFsr(unittest.TestCase):
    """FSR nhận mask NGƯỢC với ``cv2.inpaint``; bọc phải đảo giúp."""

    def test_known_pixels_move_by_at_most_one_level(self) -> None:
        """
        FSR dựng lại theo khối nên nó **cũng đụng** điểm đã biết — nhưng chỉ
        ở mức làm tròn: đo được tối đa **1/255**, trung bình 0,373. Ghi lại
        bằng test để nếu bản OpenCV sau đổi hành vi thì biết ngay; chỗ gọi
        thật (``fill_layer_disocclusion``) vẫn chỉ chép đúng vùng lỗ.
        """
        rgb, hole = _hole_scene()
        out = inpaint_hole(rgb, hole, 5, "fsr")
        outside = hole == 0
        delta = np.abs(out.astype(int) - rgb.astype(int))[outside]
        self.assertLessEqual(int(delta.max()), 1)

    def test_the_hole_is_actually_filled(self) -> None:
        """Đảo mask sai thì vùng lỗ ra đen/nguyên si — test này bắt được."""
        rgb, hole = _hole_scene()
        out = inpaint_hole(rgb, hole, 5, "fsr")
        inside = hole > 0
        self.assertGreater(float(out[inside].mean()), 1.0)
        self.assertFalse(np.array_equal(out[inside], rgb[inside]))

    def test_fsr_differs_from_telea_in_the_pipeline(self) -> None:
        got = [hashlib.sha256(l.rgba.tobytes()).hexdigest()[:32]
               for l in _layers("fsr")]
        self.assertNotEqual(got, list(GOLDEN_TELEA))


if __name__ == "__main__":
    unittest.main()


class TestLama(unittest.TestCase):
    """
    ``"lama"``: big-lama chạy ONNX, MỘT lượt suy luận mỗi lớp (cả ảnh ở cỡ làm
    việc thu vừa ô 512), cache ra đĩa. Test ở đây **không chạy model** —
    model 208 MB và mỗi lượt ~5,5 s trên máy này, không thuộc về test đơn vị.
    """

    def test_falls_back_to_telea_when_the_model_is_missing(self) -> None:
        """Thiếu model thì lấp vẫn phải ra ảnh, không được ném lỗi."""
        from unittest import mock

        rgb = np.full((40, 60, 3), 120, np.uint8)
        hole = np.zeros((40, 60), np.uint8)
        hole[10:20, 20:35] = 255
        with mock.patch("src.content.disocclusion.inpaint_lama",
                        return_value=None):
            got = inpaint_hole(rgb, hole, 5, "lama")
        want = inpaint_hole(rgb, hole, 5, "telea")
        np.testing.assert_array_equal(got, want)

    def test_tile_keeps_aspect_and_pads_to_512(self) -> None:
        from src.content.lama_fill import TILE, _fit_into_tile

        rgb = np.full((360, 640, 3), 200, np.uint8)
        hole = np.zeros((360, 640), np.uint8)
        hole[100:150, 200:260] = 255
        tile_rgb, tile_hole, (new_w, new_h) = _fit_into_tile(rgb, hole)
        self.assertEqual(tile_rgb.shape[:2], (TILE, TILE))
        self.assertEqual(tile_hole.shape[:2], (TILE, TILE))
        self.assertEqual((new_w, new_h), (TILE, round(TILE * 360 / 640)))
        # Tỉ lệ giữ nguyên trong sai số một pixel làm tròn.
        self.assertAlmostEqual(new_w / new_h, 640 / 360, places=1)
        # Phần đệm KHÔNG được coi là lỗ, nếu không model đi dựng lại cả viền.
        self.assertEqual(int(tile_hole[new_h:, :].sum()), 0)

    def test_default_is_still_telea(self) -> None:
        self.assertEqual(ContentSettings().fill_method, "telea")
        self.assertEqual(DisocclusionParams().fill_method, "telea")


class TestLamaSessionLifecycle(unittest.TestCase):
    """
    PB-076: ảnh nào **trúng cache đủ mọi lớp** thì KHÔNG được dựng phiên ORT.

    Dựng phiên tốn ~42 s cho model 208 MB. Nếu nó bị gọi cả khi có cache thì
    mọi lần nạp ảnh quen đều trả giá đó một cách vô ích.
    """

    def test_cache_hit_never_creates_the_session(self) -> None:
        from unittest import mock

        from src.content import lama_fill

        rgb = np.full((40, 60, 3), 70, np.uint8)
        hole = np.zeros((40, 60), np.uint8)
        hole[10:20, 20:35] = 255
        key = lama_fill._cache_key(rgb, hole)
        cached = lama_fill.CACHE_DIR / f"{key}.png"
        cached.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(cached), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
        try:
            with mock.patch.object(lama_fill, "_get_session") as session:
                out = lama_fill.inpaint_lama(rgb, hole)
            session.assert_not_called()
            self.assertIsNotNone(out)
            self.assertEqual(out.shape, rgb.shape)
        finally:
            cached.unlink(missing_ok=True)

    def test_preload_skips_when_the_session_already_failed(self) -> None:
        from src.content import lama_fill

        before = lama_fill._session_failed
        lama_fill._session_failed = True
        try:
            self.assertFalse(lama_fill.preload_session_async())
        finally:
            lama_fill._session_failed = before

    def test_the_app_process_never_preloads_a_lama_session(self) -> None:
        """
        TASK_game_1110 L1: LaMa nay CHỈ chạy trong tiến trình nướng.

        Trước L1, app gọi ``preload_session_async`` ở luồng nền sau khung đầu,
        và chính cú va chạm giữa luồng ấy với luồng GL đã giết hai lượt chơi
        (07/11). Giờ không còn chỗ nào trong app gọi nó nữa — chốt bằng mã
        nguồn, vì đây là một lời hứa về KIẾN TRÚC, không phải về giá trị.
        """
        source = (ROOT / "src" / "engine" / "app.py").read_text(encoding="utf-8")
        self.assertNotIn("preload_session_async", source)
        plane = (ROOT / "src" / "renderer" / "content_plane.py").read_text(
            encoding="utf-8")
        self.assertNotIn("lama", plane.lower())


class TestSessionIsBuiltOnce(unittest.TestCase):
    """
    PB-076: nhiều luồng cùng hỏi thì vẫn **chỉ dựng một phiên**.

    Thiếu khoá thì luồng nạp nền và luồng GL cùng qua cổng kiểm rồi dựng hai
    phiên: đo ngày 01/11 trong `D:\hcpd_demo` thấy đúng hai dòng "LaMa ONNX
    loaded" (80,0 s + 63,3 s) và lượt nạp ảnh kéo lên 118 s.
    """

    def test_concurrent_callers_build_one_session(self) -> None:
        import threading
        from unittest import mock

        from src.content import lama_fill

        calls = []

        def slow_build():
            calls.append(1)
            time.sleep(0.2)
            lama_fill._session = object()
            return lama_fill._session

        before = (lama_fill._session, lama_fill._session_failed)
        lama_fill._session, lama_fill._session_failed = None, False
        try:
            with mock.patch.object(lama_fill, "_load_session_locked",
                                   side_effect=slow_build):
                threads = [threading.Thread(target=lama_fill._get_session)
                           for _ in range(4)]
                for thread in threads:
                    thread.start()
                for thread in threads:
                    thread.join()
            self.assertEqual(len(calls), 1, "dung nhieu hon mot phien")
        finally:
            lama_fill._session, lama_fill._session_failed = before
