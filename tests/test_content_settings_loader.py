"""Tests for the dataclass-driven ContentSettings loader."""

from __future__ import annotations

import json
import unittest
from dataclasses import fields
from pathlib import Path

from src.config.schema import CONTENT_KEY_ALIASES, ContentSettings

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONTENT_JSON = PROJECT_ROOT / "assets" / "config" / "content.json"


def _perturb(default):
    """Return a value that differs from ``default`` but shares its type."""
    if isinstance(default, bool):
        return not default
    if isinstance(default, int):
        return default + 7
    if isinstance(default, float):
        return default + 1.5
    if isinstance(default, str):
        return "changed_" + default
    if isinstance(default, tuple):
        return tuple(v + 1.0 for v in default)
    raise AssertionError(f"unhandled default type {type(default).__name__}")


class EveryFieldReachableTest(unittest.TestCase):
    """
    The regression guard for the original bug.

    The old loader parsed ~190 lines of payload.get() by hand, so any field
    added to the dataclass without editing that list was silently ignored.
    This test fails the moment a field stops being readable from content.json.
    """

    def test_every_field_is_reachable_from_json(self) -> None:
        defaults = ContentSettings()
        unreachable = []
        for field_info in fields(ContentSettings):
            name = field_info.name
            default = getattr(defaults, name)
            wanted = _perturb(default)
            loaded = ContentSettings.from_dict({name: wanted})
            if getattr(loaded, name) != wanted:
                unreachable.append(name)
        self.assertEqual(
            unreachable, [], f"fields not readable from content.json: {unreachable}"
        )

    def test_one_key_does_not_disturb_the_others(self) -> None:
        defaults = ContentSettings()
        loaded = ContentSettings.from_dict({"layer_count": 9})
        self.assertEqual(loaded.layer_count, 9)
        for field_info in fields(ContentSettings):
            if field_info.name == "layer_count":
                continue
            self.assertEqual(
                getattr(loaded, field_info.name),
                getattr(defaults, field_info.name),
                f"{field_info.name} changed unexpectedly",
            )


class RealConfigTest(unittest.TestCase):
    """The shipped content.json must load cleanly and completely."""

    def test_every_shipped_key_lands_on_the_dataclass(self) -> None:
        self.assertTrue(CONTENT_JSON.exists(), f"missing {CONTENT_JSON}")
        payload = json.loads(CONTENT_JSON.read_text(encoding="utf-8"))
        settings = ContentSettings.from_dict(payload)
        names = {f.name for f in fields(ContentSettings)}

        for key, raw in payload.items():
            if key not in names:
                continue
            got = getattr(settings, key)
            expected = (
                tuple(float(v) for v in raw) if isinstance(got, tuple) else type(got)(raw)
            )
            self.assertEqual(got, expected, f"{key} mismatch")

    def test_shipped_config_has_no_unrecognised_keys(self) -> None:
        payload = json.loads(CONTENT_JSON.read_text(encoding="utf-8"))
        known = {f.name for f in fields(ContentSettings)}
        known.update(CONTENT_KEY_ALIASES.values())
        unknown = sorted(
            k for k in payload if k not in known and not k.startswith("_")
        )
        self.assertEqual(unknown, [], f"stale keys in content.json: {unknown}")


class CoercionTest(unittest.TestCase):
    """JSON is loosely typed; the dataclass is not."""

    def test_int_json_value_becomes_float_field(self) -> None:
        loaded = ContentSettings.from_dict({"gvf_threshold": 1})
        self.assertIsInstance(loaded.gvf_threshold, float)
        self.assertEqual(loaded.gvf_threshold, 1.0)

    def test_float_json_value_becomes_int_field(self) -> None:
        loaded = ContentSettings.from_dict({"layer_count": 4.0})
        self.assertIsInstance(loaded.layer_count, int)
        self.assertEqual(loaded.layer_count, 4)

    def test_tuple_field_coerces_element_types(self) -> None:
        loaded = ContentSettings.from_dict({"plane_position_m": [0, 1, -2]})
        self.assertEqual(loaded.plane_position_m, (0.0, 1.0, -2.0))
        self.assertTrue(all(isinstance(v, float) for v in loaded.plane_position_m))

    def test_zero_and_false_are_not_treated_as_missing(self) -> None:
        loaded = ContentSettings.from_dict(
            {"layer_count": 0, "mist_enabled": False, "layer_z_near_m": 0.0}
        )
        self.assertEqual(loaded.layer_count, 0)
        self.assertFalse(loaded.mist_enabled)
        self.assertEqual(loaded.layer_z_near_m, 0.0)


class AliasTest(unittest.TestCase):
    """rim_light_intensity falls back to the older layer_rim_intensity key."""

    def test_alias_used_when_primary_absent(self) -> None:
        loaded = ContentSettings.from_dict({"layer_rim_intensity": 0.77})
        self.assertAlmostEqual(loaded.rim_light_intensity, 0.77)

    def test_primary_wins_over_alias(self) -> None:
        loaded = ContentSettings.from_dict(
            {"rim_light_intensity": 0.11, "layer_rim_intensity": 0.99}
        )
        self.assertAlmostEqual(loaded.rim_light_intensity, 0.11)
        self.assertAlmostEqual(loaded.layer_rim_intensity, 0.99)


class BadInputTest(unittest.TestCase):
    """Malformed config must degrade to defaults, never kill startup."""

    def test_empty_payload_gives_defaults(self) -> None:
        self.assertEqual(ContentSettings.from_dict({}), ContentSettings())

    def test_unparsable_value_falls_back_to_default(self) -> None:
        defaults = ContentSettings()
        with self.assertLogs("src.config.schema", level="WARNING"):
            loaded = ContentSettings.from_dict({"layer_count": "not-a-number"})
        self.assertEqual(loaded.layer_count, defaults.layer_count)

    def test_wrong_length_tuple_falls_back_to_default(self) -> None:
        defaults = ContentSettings()
        with self.assertLogs("src.config.schema", level="WARNING"):
            loaded = ContentSettings.from_dict({"frame_color": [0.5, 0.5]})
        self.assertEqual(loaded.frame_color, defaults.frame_color)

    def test_unknown_key_is_reported_and_ignored(self) -> None:
        with self.assertLogs("src.config.schema", level="WARNING") as captured:
            loaded = ContentSettings.from_dict({"totally_made_up": 1})
        self.assertIn("totally_made_up", "\n".join(captured.output))
        self.assertEqual(loaded, ContentSettings())

    def test_underscore_comment_keys_are_not_reported(self) -> None:
        # JSON has no comments; "_note" style keys are used for documentation.
        with self.assertRaises(AssertionError):
            with self.assertLogs("src.config.schema", level="WARNING"):
                ContentSettings.from_dict({"_note": "a comment"})


if __name__ == "__main__":
    unittest.main()
