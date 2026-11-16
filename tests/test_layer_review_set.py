r"""
TASK_game_1110 — bộ ảnh review của ``scripts/layer_review.py``.

Thuần Python, không GL: chấm đúng phần có thể chấm mà không mở ngữ cảnh GL —
tên tám ảnh chính thức, và việc dò đường dẫn NFC/NFD không ném ngoại lệ.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import layer_review as LR  # noqa: E402


class EightSetTest(unittest.TestCase):
    def test_eight_unique_names(self) -> None:
        self.assertEqual(len(LR.EIGHT_SET), 8)
        self.assertEqual(len(set(LR.EIGHT_SET)), 8)

    def test_excludes_the_known_duplicate_t_images(self) -> None:
        # t11 == u2_cat, t12 == u1_anime_room, t17 == u3_van_tran_pha (cùng
        # byte, tìm ra 10/11) — bộ tám không được đếm trùng.
        for dup in ("t11", "t12", "t17"):
            self.assertNotIn(dup, LR.EIGHT_SET)

    def test_three_are_user_photos_five_are_t_images(self) -> None:
        user = [n for n in LR.EIGHT_SET if n.startswith("u")]
        t = [n for n in LR.EIGHT_SET if n.startswith("t")]
        self.assertEqual(len(user), 3)
        self.assertEqual(len(t), 5)


class NfcNfdResolutionTest(unittest.TestCase):
    """USER_DIR/USER_IMAGES phải dò ra file thật dù OneDrive ghi tên ở NFD."""

    def test_resolve_by_nfc_never_raises_on_a_missing_parent(self) -> None:
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            missing = Path(tmp) / "khong-ton-tai"
            result = LR._resolve_by_nfc(missing, "Hình ảnh")
            self.assertEqual(result, missing / "Hình ảnh")

    def test_resolve_by_nfc_finds_an_nfd_entry_by_nfc_name(self) -> None:
        import tempfile
        import unicodedata

        with tempfile.TemporaryDirectory() as tmp:
            parent = Path(tmp)
            nfd_name = unicodedata.normalize("NFD", "Hình ảnh")
            (parent / nfd_name).mkdir()
            found = LR._resolve_by_nfc(parent, "Hình ảnh")
            self.assertTrue(found.is_dir())
            self.assertEqual(unicodedata.normalize("NFC", found.name),
                             "Hình ảnh")


if __name__ == "__main__":
    unittest.main()
