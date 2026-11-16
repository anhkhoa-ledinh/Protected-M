"""Tests for per-user face-scale calibration (PHASE B)."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from src.pose.calibration import (
    FACE_SCALE_MAX,
    FACE_SCALE_MIN,
    CameraIntrinsics,
    FaceCalibration,
    FaceCalibrationError,
    compute_face_scale,
    load_camera_intrinsics,
    load_face_calibration,
    resolve_face_scale,
    resolve_intrinsics,
    save_face_calibration,
)
from src.pose.head_pose import HeadPoseEstimator
from src.tracking.landmarks import (
    FACE_MODEL_MM,
    FACE_MODEL_RAW,
    FACE_MODEL_UNITS_PER_MM,
    LandmarkSet,
)

import cv2


def _landmarks_at(distance_mm: float, intrinsics: CameraIntrinsics, model) -> LandmarkSet:
    """Project a face model at a true distance to synthesize 2D landmarks."""
    points, _ = cv2.projectPoints(
        np.asarray(model, dtype=np.float64),
        np.zeros((3, 1)),
        np.array([[0.0], [0.0], [float(distance_mm)]]),
        intrinsics.matrix,
        intrinsics.dist_coeffs,
    )
    points = points.reshape(-1, 2)
    return LandmarkSet(
        image_points=points,
        left_iris=points[2],
        right_iris=points[3],
    )


class FaceModelUnitsTest(unittest.TestCase):
    """FACE_MODEL_MM must be in true millimetres after the rescale."""

    def test_model_is_raw_divided_by_units_per_mm(self) -> None:
        np.testing.assert_allclose(
            FACE_MODEL_MM, FACE_MODEL_RAW / FACE_MODEL_UNITS_PER_MM
        )

    def test_model_dimensions_are_anthropometric(self) -> None:
        eye_span = float(np.linalg.norm(FACE_MODEL_MM[3] - FACE_MODEL_MM[2]))
        nose_chin = float(np.linalg.norm(FACE_MODEL_MM[1] - FACE_MODEL_MM[0]))
        self.assertAlmostEqual(eye_span, 90.0, places=6)
        self.assertTrue(55.0 <= nose_chin <= 80.0, f"nose->chin {nose_chin:.1f} mm")


class ComputeFaceScaleTest(unittest.TestCase):
    """face_scale = D / median(|tvec.z|), with a hard sanity gate."""

    def test_scale_is_distance_over_median(self) -> None:
        samples = [480.0, 500.0, 520.0]  # median 500
        result = compute_face_scale(samples, 600.0, min_valid_frames=3)
        self.assertAlmostEqual(result.face_scale, 600.0 / 500.0, places=9)
        self.assertAlmostEqual(result.tz_median_mm, 500.0, places=9)
        self.assertEqual(result.valid_frames, 3)
        self.assertEqual(result.distance_mm, 600.0)

    def test_median_rejects_outliers(self) -> None:
        clean = compute_face_scale([600.0] * 9, 600.0, min_valid_frames=3)
        spiked = compute_face_scale(
            [600.0] * 9 + [9000.0], 600.0, min_valid_frames=3
        )
        self.assertAlmostEqual(clean.face_scale, spiked.face_scale, places=9)

    def test_absolute_value_applied_to_negative_tz(self) -> None:
        result = compute_face_scale([-500.0] * 5, 600.0, min_valid_frames=3)
        self.assertAlmostEqual(result.face_scale, 1.2, places=9)

    def test_too_few_valid_frames_raises(self) -> None:
        with self.assertRaises(FaceCalibrationError):
            compute_face_scale([600.0] * 29, 600.0, min_valid_frames=30)

    def test_non_positive_distance_raises(self) -> None:
        with self.assertRaises(FaceCalibrationError):
            compute_face_scale([600.0] * 40, 0.0)


class FaceScaleSanityGateTest(unittest.TestCase):
    """Outside [0.6, 1.6] calibration must fail, never fall back."""

    def test_scale_below_gate_raises(self) -> None:
        # This is the real pre-rescale case: tz_median 2824 at D=600 -> 0.21.
        with self.assertRaises(FaceCalibrationError):
            compute_face_scale([2824.23] * 40, 600.0)

    def test_scale_above_gate_raises(self) -> None:
        with self.assertRaises(FaceCalibrationError):
            compute_face_scale([300.0] * 40, 600.0)

    def test_gate_boundaries_are_inclusive(self) -> None:
        low = compute_face_scale([600.0 / FACE_SCALE_MIN] * 40, 600.0)
        high = compute_face_scale([600.0 / FACE_SCALE_MAX] * 40, 600.0)
        self.assertAlmostEqual(low.face_scale, FACE_SCALE_MIN, places=6)
        self.assertAlmostEqual(high.face_scale, FACE_SCALE_MAX, places=6)

    def test_stored_out_of_gate_value_is_ignored(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "user.json"
            path.write_text(
                json.dumps({"face_calibration": {"face_scale": 0.21}}),
                encoding="utf-8",
            )
            self.assertIsNone(load_face_calibration(path))


class FaceCalibrationPersistenceTest(unittest.TestCase):
    """user.json round-trip must preserve unrelated config keys."""

    def _calibration(self) -> FaceCalibration:
        return compute_face_scale([565.0] * 40, 600.0)

    def test_round_trip(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "user.json"
            original = self._calibration()
            save_face_calibration(original, path)
            loaded = load_face_calibration(path)
            self.assertIsNotNone(loaded)
            assert loaded is not None
            self.assertAlmostEqual(loaded.face_scale, original.face_scale, places=9)
            self.assertEqual(loaded.valid_frames, original.valid_frames)

    def test_save_merges_and_preserves_other_keys(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "user.json"
            path.write_text(
                json.dumps({"screen": {"width_mm": 344.0}}), encoding="utf-8"
            )
            save_face_calibration(self._calibration(), path)
            payload = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(payload["screen"]["width_mm"], 344.0)
            self.assertIn("face_calibration", payload)

    def test_missing_file_returns_none(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            self.assertIsNone(load_face_calibration(Path(tmp) / "absent.json"))

    def test_file_without_block_returns_none(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "user.json"
            path.write_text(json.dumps({"screen": {}}), encoding="utf-8")
            self.assertIsNone(load_face_calibration(path))


class CalibratedZPathTest(unittest.TestCase):
    """Calibrated path reports true millimetres; legacy path is unchanged."""

    def setUp(self) -> None:
        self.intrinsics = CameraIntrinsics.from_image_size(1280, 720)
        self.offset = np.zeros(3, dtype=np.float64)

    def test_calibrated_z_tracks_true_distance(self) -> None:
        # A user whose face is 1.1x the generic model.
        true_scale = 1.1
        model = FACE_MODEL_MM * true_scale
        estimator = HeadPoseEstimator(
            self.intrinsics,
            webcam_offset_mm=self.offset,
            face_scale=true_scale,
        )
        for distance in (400.0, 600.0, 800.0):
            result = estimator.estimate(
                _landmarks_at(distance, self.intrinsics, model), timestamp=0.0
            )
            self.assertAlmostEqual(
                float(result.translation_mm[2]), distance, delta=1.0
            )

    def test_calibrated_path_ignores_first_frame_anchor(self) -> None:
        """No baseline is captured when calibrated — order must not matter."""
        model = FACE_MODEL_MM.copy()
        estimator = HeadPoseEstimator(
            self.intrinsics, webcam_offset_mm=self.offset, face_scale=1.0
        )
        estimator.estimate(_landmarks_at(800.0, self.intrinsics, model), timestamp=0.0)
        result = estimator.estimate(
            _landmarks_at(400.0, self.intrinsics, model), timestamp=1.0
        )
        self.assertAlmostEqual(float(result.translation_mm[2]), 400.0, delta=1.0)
        self.assertIsNone(estimator._reference_depth_raw)

    def test_uncalibrated_path_is_invariant_to_the_model_rescale(self) -> None:
        """Legacy behaviour must be byte-for-byte unchanged by /5 rescale."""
        distances = (600.0, 500.0, 700.0, 800.0, 400.0)

        def run(model) -> list[float]:
            estimator = HeadPoseEstimator(
                self.intrinsics, webcam_offset_mm=self.offset, face_scale=None
            )
            return [
                float(
                    estimator.estimate(
                        _landmarks_at(d, self.intrinsics, model), timestamp=0.0
                    ).translation_mm[2]
                )
                for d in distances
            ]

        # Landmarks are identical up to scale, so the legacy ratio path must
        # produce identical Z whether the model is raw units or millimetres.
        np.testing.assert_allclose(
            run(FACE_MODEL_MM), run(FACE_MODEL_RAW), rtol=1e-9, atol=1e-6
        )

    def test_uncalibrated_first_frame_anchors_to_reference_distance(self) -> None:
        estimator = HeadPoseEstimator(
            self.intrinsics,
            reference_distance_mm=600.0,
            webcam_offset_mm=self.offset,
            face_scale=None,
        )
        first = estimator.estimate(
            _landmarks_at(750.0, self.intrinsics, FACE_MODEL_MM), timestamp=0.0
        )
        self.assertAlmostEqual(float(first.translation_mm[2]), 600.0, delta=1.0)
        self.assertIsNotNone(estimator._reference_depth_raw)


if __name__ == "__main__":
    unittest.main()


class MeasuredIntrinsicsTest(unittest.TestCase):
    """
    Guards on loading a MEASURED intrinsics block (audit #10).

    fx sets the recovered angular direction ``(u - cx) / fx``, so it scales
    lateral parallax directly. The block shipped in the repo was never read by
    any code path; these tests pin both halves of the fix — that a block without
    provenance is refused, and that a block with provenance is scaled correctly.
    """

    @staticmethod
    def _write(block: dict) -> Path:
        handle = tempfile.NamedTemporaryFile(
            "w", suffix=".json", delete=False, encoding="utf-8"
        )
        json.dump({"camera_intrinsics": block}, handle)
        handle.close()
        return Path(handle.name)

    def test_block_without_declared_resolution_is_refused(self) -> None:
        """fx/fy/cx/cy are pixels; without their resolution they mean nothing."""
        path = self._write(
            {"fx": 960.0, "fy": 960.0, "cx": 640.0, "cy": 360.0}
        )
        try:
            self.assertIsNone(load_camera_intrinsics(1280, 720, path))
        finally:
            path.unlink()

    def test_block_is_scaled_to_the_running_resolution(self) -> None:
        path = self._write(
            {
                "fx": 960.0, "fy": 960.0, "cx": 640.0, "cy": 360.0,
                "width": 1280, "height": 720,
            }
        )
        try:
            half = load_camera_intrinsics(640, 360, path)
            self.assertIsNotNone(half)
            self.assertAlmostEqual(half.fx, 480.0, places=6)
            self.assertAlmostEqual(half.cx, 320.0, places=6)
            self.assertAlmostEqual(half.cy, 180.0, places=6)
        finally:
            path.unlink()

    def test_aspect_mismatch_is_refused_not_stretched(self) -> None:
        """A different aspect is a crop or another sensor mode, not a resize."""
        path = self._write(
            {
                "fx": 960.0, "fy": 960.0, "cx": 640.0, "cy": 360.0,
                "width": 1280, "height": 720,
            }
        )
        try:
            self.assertIsNone(load_camera_intrinsics(640, 480, path))
        finally:
            path.unlink()

    def test_from_dict_no_longer_defaults_fx_to_width(self) -> None:
        """
        The old default was fx = width, i.e. 53.1 deg FOV, against
        from_image_size's 63 deg — a 23% disagreement inside one class.
        """
        with self.assertRaises(ValueError):
            CameraIntrinsics.from_dict(
                {"cx": 640.0, "cy": 360.0, "width": 1280, "height": 720},
                1280,
                720,
            )

    def test_resolver_falls_back_to_assumed_fov(self) -> None:
        missing = Path(tempfile.gettempdir()) / "hcpd-no-such-calibration.json"
        self.assertFalse(missing.exists())
        resolved = resolve_intrinsics(1280, 720, missing)
        self.assertAlmostEqual(
            resolved.fx, CameraIntrinsics.from_image_size(1280, 720).fx, places=9
        )


class FaceScaleIntrinsicsCouplingTest(unittest.TestCase):
    """
    ``face_scale`` is only valid for the fx it was measured with (audit #10).

    ``face_scale = D / median(|tvec.z|)`` and tvec.z scales with fx, so the scale
    silently absorbs whatever fx was in force. The [0.6, 1.6] sanity gate is far
    too wide to notice an 8% shift, so the coupling has to be checked explicitly.
    """

    def _store(self, block: dict) -> Path:
        handle = tempfile.NamedTemporaryFile(
            "w", suffix=".json", delete=False, encoding="utf-8"
        )
        json.dump({"face_calibration": block}, handle)
        handle.close()
        return Path(handle.name)

    def test_unchanged_fx_returns_the_stored_scale_untouched(self) -> None:
        intrinsics = CameraIntrinsics.from_image_size(1280, 720)
        path = self._store(
            {"face_scale": 0.9555, "intrinsics_fx": intrinsics.fx}
        )
        try:
            self.assertAlmostEqual(
                resolve_face_scale(intrinsics, path), 0.9555, places=12
            )
        finally:
            path.unlink()

    def test_legacy_block_without_fx_assumes_the_assumed_fov(self) -> None:
        """Every calibration that existed before came from from_image_size."""
        intrinsics = CameraIntrinsics.from_image_size(1280, 720)
        path = self._store({"face_scale": 0.9555})
        try:
            self.assertAlmostEqual(
                resolve_face_scale(intrinsics, path), 0.9555, places=12
            )
        finally:
            path.unlink()

    def test_changed_fx_migrates_by_the_fx_ratio(self) -> None:
        path = self._store({"face_scale": 0.9555, "intrinsics_fx": 1044.4})
        live = CameraIntrinsics(1280, 720, 960.0, 960.0, 640.0, 360.0, np.zeros(5))
        try:
            self.assertAlmostEqual(
                resolve_face_scale(live, path),
                0.9555 * 1044.4 / 960.0,
                places=9,
            )
        finally:
            path.unlink()

    def test_anisotropic_intrinsics_refuse_the_stored_scale(self) -> None:
        """
        The fx-ratio migration is blind to fy. Measured: 6% of fy anisotropy
        moves migrated depth by 3.3% (20 mm at 600 mm), so refuse instead.
        """
        path = self._store({"face_scale": 0.9555, "intrinsics_fx": 1044.4})
        live = CameraIntrinsics(1280, 720, 960.0, 900.0, 640.0, 360.0, np.zeros(5))
        try:
            self.assertIsNone(resolve_face_scale(live, path))
        finally:
            path.unlink()

    def test_migration_outside_the_sanity_gate_is_refused(self) -> None:
        path = self._store({"face_scale": 1.5, "intrinsics_fx": 2000.0})
        live = CameraIntrinsics(1280, 720, 500.0, 500.0, 640.0, 360.0, np.zeros(5))
        try:
            self.assertIsNone(resolve_face_scale(live, path))
        finally:
            path.unlink()

    def test_computed_calibration_records_its_fx(self) -> None:
        result = compute_face_scale(
            [600.0] * 40, 600.0, intrinsics_fx=1044.4
        )
        self.assertAlmostEqual(result.intrinsics_fx, 1044.4, places=9)
        self.assertAlmostEqual(
            result.to_dict()["intrinsics_fx"], 1044.4, places=9
        )
