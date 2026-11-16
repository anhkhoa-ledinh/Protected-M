"""Camera intrinsics and per-user face-scale calibration."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import numpy.typing as npt

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class CameraIntrinsics:
    """Pinhole camera intrinsic matrix and distortion coefficients."""

    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float
    dist_coeffs: npt.NDArray[np.float64]

    @property
    def matrix(self) -> npt.NDArray[np.float64]:
        """3x3 camera matrix K."""
        return np.array(
            [
                [self.fx, 0.0, self.cx],
                [0.0, self.fy, self.cy],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )

    @classmethod
    def from_image_size(
        cls,
        width: int,
        height: int,
        fov_deg: float = 63.0,
    ) -> CameraIntrinsics:
        """Estimate intrinsics from image resolution and approximate FOV."""
        fx = width / (2.0 * np.tan(np.radians(fov_deg * 0.5)))
        fy = fx
        cx = width * 0.5
        cy = height * 0.5
        dist = np.zeros(5, dtype=np.float64)
        return cls(width, height, fx, fy, cx, cy, dist)

    @classmethod
    def from_dict(cls, data: dict, width: int, height: int) -> CameraIntrinsics:
        """
        Parse a MEASURED intrinsics block and rescale it to a running resolution.

        Raises :class:`ValueError` rather than filling gaps with guesses. The
        previous version defaulted ``fx`` and ``fy`` to ``width``, which is a
        53.1 deg horizontal FOV, while :meth:`from_image_size` assumes 63 deg
        (``fx = 0.8156 * width``). Two constructors of the same class disagreeing
        by 23% on the single number that sets lateral parallax is not a default,
        it is a trap.

        ``width``/``height`` inside ``data`` are the resolution the calibration
        was MEASURED at, and they are mandatory: fx, fy, cx, cy are all in
        pixels, so a block measured at 1280x720 means nothing at 640x480 until
        it is scaled. Aspect ratio must match -- webcams change FOV between
        sensor modes rather than plain-resizing, and a crop cannot be recovered
        by scaling, so a mismatch is refused instead of approximated.
        """
        if not data:
            raise ValueError("Empty camera_intrinsics block")

        missing = [k for k in ("fx", "fy", "cx", "cy") if k not in data]
        if missing:
            raise ValueError(f"camera_intrinsics is missing {', '.join(missing)}")

        calib_w = float(data.get("width", 0.0))
        calib_h = float(data.get("height", 0.0))
        if calib_w <= 0.0 or calib_h <= 0.0:
            raise ValueError(
                "camera_intrinsics must declare the 'width'/'height' it was "
                "measured at; fx/fy/cx/cy are pixel quantities and cannot be "
                "applied to another resolution without it"
            )

        calib_aspect = calib_w / calib_h
        run_aspect = float(width) / float(max(height, 1))
        if abs(calib_aspect - run_aspect) > 0.01:
            raise ValueError(
                f"camera_intrinsics measured at {calib_w:.0f}x{calib_h:.0f} "
                f"(aspect {calib_aspect:.3f}) cannot be scaled to "
                f"{width}x{height} (aspect {run_aspect:.3f}) — different aspect "
                "means a crop or a different sensor mode, not a resize"
            )

        scale_x = float(width) / calib_w
        scale_y = float(height) / calib_h
        return cls(
            width=width,
            height=height,
            fx=float(data["fx"]) * scale_x,
            fy=float(data["fy"]) * scale_y,
            cx=float(data["cx"]) * scale_x,
            cy=float(data["cy"]) * scale_y,
            dist_coeffs=np.asarray(
                data.get("dist_coeffs", [0.0, 0.0, 0.0, 0.0, 0.0]),
                dtype=np.float64,
            ),
        )


#: Key holding the intrinsics block inside assets/config/camera_calibration.json.
CAMERA_INTRINSICS_KEY: str = "camera_intrinsics"


def default_camera_calibration_path() -> Path:
    """Path to assets/config/camera_calibration.json inside this project."""
    return (
        Path(__file__).resolve().parents[2]
        / "assets"
        / "config"
        / "camera_calibration.json"
    )


def load_camera_intrinsics(
    width: int,
    height: int,
    path: Path | None = None,
) -> CameraIntrinsics | None:
    """
    Load measured intrinsics for a running resolution, or ``None`` to fall back.

    ``None`` means "no usable measurement" and the caller must use
    :meth:`CameraIntrinsics.from_image_size`. Every reason for ``None`` is
    logged, because the failure mode here is silence: this file existed and was
    never read by anything, so nobody noticed that the live path was running on
    an assumed 63 deg FOV.

    WHY fx MATTERS, and why it is not just a Z problem: solvePnP recovers the
    angular direction of the face as ``(u - cx) / fx``. A stored ``face_scale``
    absorbs an fx error on Z (it is a scalar on depth), but nothing absorbs it
    laterally -- with Z correct, ``x_est = x_true * fx_true / fx_assumed``. At
    the shipped assumption fx = 1044.4 against the 960 written in this file,
    that is an 8% error on horizontal parallax, the one quantity this whole
    project exists to get right.

    The block shipped in the repo declares no ``width``/``height`` and has
    ``dist_coeffs`` all zero with ``cx``/``cy`` exactly at the image centre --
    the signature of a hand-written placeholder, not a ``cv2.calibrateCamera``
    run. It is therefore REFUSED, loudly, rather than adopted: replacing a
    measured-by-assumption fx with a guessed-by-someone-else fx is not progress.
    Run a real checkerboard calibration, write ``width``/``height`` into the
    block, and this function starts returning it.
    """
    config_path = path or default_camera_calibration_path()
    if not config_path.exists():
        logger.info(
            "No %s — intrinsics from assumed FOV (from_image_size)",
            config_path.name,
        )
        return None
    try:
        payload = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        logger.warning("Cannot read %s (%s) — using assumed FOV", config_path, exc)
        return None

    block = payload.get(CAMERA_INTRINSICS_KEY) if isinstance(payload, dict) else None
    if not isinstance(block, dict) or not block:
        logger.info(
            "%s has no usable '%s' block — using assumed FOV",
            config_path.name,
            CAMERA_INTRINSICS_KEY,
        )
        return None

    try:
        intrinsics = CameraIntrinsics.from_dict(block, width, height)
    except (ValueError, TypeError) as exc:
        assumed = CameraIntrinsics.from_image_size(width, height)
        logger.warning(
            "IGNORING %s: %s. Staying on the assumed-FOV estimate "
            "(fx=%.1f px). Lateral parallax scales with fx, so this remains an "
            "UNMEASURED assumption — see load_camera_intrinsics.__doc__",
            config_path.name,
            exc,
            assumed.fx,
        )
        return None

    assumed = CameraIntrinsics.from_image_size(width, height)
    logger.info(
        "Measured intrinsics loaded from %s: fx=%.1f fy=%.1f cx=%.1f cy=%.1f "
        "(assumed-FOV estimate would have been fx=%.1f, ratio %.4f)",
        config_path.name,
        intrinsics.fx,
        intrinsics.fy,
        intrinsics.cx,
        intrinsics.cy,
        assumed.fx,
        intrinsics.fx / assumed.fx,
    )
    return intrinsics


def resolve_intrinsics(
    width: int,
    height: int,
    path: Path | None = None,
) -> CameraIntrinsics:
    """
    Measured intrinsics when available, otherwise the assumed-FOV estimate.

    ONE entry point, used by the tracking worker AND the calibration wizard. They
    must agree: ``face_scale`` is measured against a specific fx, so calibrating
    with one fx and running with another reintroduces exactly the coupling error
    that :func:`resolve_face_scale` exists to catch.
    """
    measured = load_camera_intrinsics(width, height, path)
    if measured is not None:
        return measured
    return CameraIntrinsics.from_image_size(width, height)


# ---------------------------------------------------------------------------
# PHASE B — per-user face-scale calibration.
#
# cv2.solvePnP returns tvec in the units of the 3D model it is given.  If a
# real face is k times larger than FACE_MODEL_MM, tvec.z is off by the same
# factor k, so ONE scalar multiplying the model coordinates removes the error.
#
# face_scale = D / median(|tvec.z|)   measured at a known distance D.
# ---------------------------------------------------------------------------

#: Sanity gate on the computed scale. Outside this range calibration FAILS —
#: it never silently falls back to a default.
FACE_SCALE_MIN: float = 0.6
FACE_SCALE_MAX: float = 1.6

#: Capture protocol.
CALIBRATION_FRAMES: int = 60
CALIBRATION_MIN_VALID_FRAMES: int = 30
CALIBRATION_MIN_PRESENCE: float = 0.5

#: Reprojection-error gate.  The original spec said 6.0 px, but measured on
#: this machine (2026-08-20, 467 frames) the RMSE was min=7.07, median=13.18,
#: max=33.42 px — a 6.0 gate rejects EVERY frame and calibration can never
#: succeed.  The residual is a SHAPE mismatch between this face and the generic
#: model; it is invariant to face_scale, so no amount of calibrating reduces it.
#: 15.0 keeps the majority of frames while still discarding bad PnP solutions.
CALIBRATION_MAX_RMSE_PX: float = 15.0
DEFAULT_CALIBRATION_DISTANCE_MM: float = 600.0

#: Key holding the calibration block inside assets/config/user.json.
FACE_CALIBRATION_KEY: str = "face_calibration"


class FaceCalibrationError(RuntimeError):
    """Calibration could not be completed. Never fall back to a default."""


@dataclass(frozen=True)
class FaceCalibration:
    """
    Result of one per-user face-scale calibration run.

    ``intrinsics_fx`` records the fx the run was measured WITH. It is not
    decoration: ``face_scale = D / median(|tvec.z|)`` and ``tvec.z`` scales with
    fx, so the scale silently absorbs whatever fx was in force. Restoring it
    under a different fx is then wrong by that ratio, with no symptom. ``0.0``
    means a pre-2026-09-22 block that never recorded it -- see
    :func:`resolve_face_scale`.
    """

    face_scale: float
    distance_mm: float
    tz_median_mm: float
    tz_std_mm: float
    valid_frames: int
    calibrated_at: str
    intrinsics_fx: float = 0.0

    def to_dict(self) -> dict:
        """Serialize for assets/config/user.json."""
        return {
            "face_scale": float(self.face_scale),
            "distance_mm": float(self.distance_mm),
            "tz_median_mm": float(self.tz_median_mm),
            "tz_std_mm": float(self.tz_std_mm),
            "valid_frames": int(self.valid_frames),
            "calibrated_at": str(self.calibrated_at),
            "intrinsics_fx": float(self.intrinsics_fx),
        }

    @classmethod
    def from_dict(cls, data: dict) -> FaceCalibration:
        """Rebuild from a user.json block, validating the sanity gate."""
        face_scale = float(data["face_scale"])
        if not FACE_SCALE_MIN <= face_scale <= FACE_SCALE_MAX:
            raise FaceCalibrationError(
                f"Stored face_scale {face_scale:.4f} is outside the sanity gate "
                f"[{FACE_SCALE_MIN}, {FACE_SCALE_MAX}]"
            )
        return cls(
            face_scale=face_scale,
            distance_mm=float(data.get("distance_mm", DEFAULT_CALIBRATION_DISTANCE_MM)),
            tz_median_mm=float(data.get("tz_median_mm", 0.0)),
            tz_std_mm=float(data.get("tz_std_mm", 0.0)),
            valid_frames=int(data.get("valid_frames", 0)),
            calibrated_at=str(data.get("calibrated_at", "")),
            intrinsics_fx=float(data.get("intrinsics_fx", 0.0)),
        )


def default_user_config_path() -> Path:
    """Path to assets/config/user.json inside this project."""
    return Path(__file__).resolve().parents[2] / "assets" / "config" / "user.json"


def compute_face_scale(
    tz_samples_mm,
    distance_mm: float = DEFAULT_CALIBRATION_DISTANCE_MM,
    *,
    min_valid_frames: int = CALIBRATION_MIN_VALID_FRAMES,
    calibrated_at: str | None = None,
    intrinsics_fx: float = 0.0,
) -> FaceCalibration:
    """
    Derive ``face_scale`` from |tvec.z| samples taken at a known distance.

    ``tz_samples_mm`` must already be filtered by presence and reprojection
    error. Raises :class:`FaceCalibrationError` when too few samples survive,
    when the distance is not positive, or when the resulting scale falls
    outside ``[FACE_SCALE_MIN, FACE_SCALE_MAX]``.

    Pass ``intrinsics_fx`` -- the fx these samples were solved with -- so the
    result stays interpretable when the intrinsics change. It defaults to 0.0
    ("unrecorded") only so existing positional callers keep working.
    """
    samples = np.abs(np.asarray(tz_samples_mm, dtype=np.float64).ravel())
    samples = samples[np.isfinite(samples)]

    if distance_mm <= 0.0:
        raise FaceCalibrationError(
            f"Calibration distance must be positive, got {distance_mm}"
        )
    if samples.size < int(min_valid_frames):
        raise FaceCalibrationError(
            f"Only {samples.size} valid frames survived filtering "
            f"(need >= {min_valid_frames}). Redo the calibration: hold still, "
            "face the camera, and keep the whole face in frame."
        )

    tz_median = float(np.median(samples))
    if tz_median <= 1e-6:
        raise FaceCalibrationError(
            f"Median |tvec.z| is degenerate ({tz_median:.6f} mm)"
        )

    face_scale = float(distance_mm) / tz_median
    if not FACE_SCALE_MIN <= face_scale <= FACE_SCALE_MAX:
        raise FaceCalibrationError(
            f"face_scale {face_scale:.4f} is outside the sanity gate "
            f"[{FACE_SCALE_MIN}, {FACE_SCALE_MAX}] "
            f"(D={distance_mm:.1f} mm, median|tvec.z|={tz_median:.1f} mm). "
            "Check the measured distance and that the face model units are "
            "millimetres. Nothing was saved."
        )

    return FaceCalibration(
        face_scale=face_scale,
        distance_mm=float(distance_mm),
        tz_median_mm=tz_median,
        tz_std_mm=float(samples.std()),
        valid_frames=int(samples.size),
        calibrated_at=calibrated_at
        or datetime.now(timezone.utc).isoformat(timespec="seconds"),
        intrinsics_fx=float(intrinsics_fx),
    )


def load_face_calibration(path: Path | None = None) -> FaceCalibration | None:
    """
    Read the stored calibration, or ``None`` when absent/unusable.

    A missing file, a missing block, or a value failing the sanity gate all
    return ``None`` so the caller stays on the legacy path.
    """
    config_path = path or default_user_config_path()
    if not config_path.exists():
        return None
    try:
        payload = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        logger.warning("Cannot read %s (%s) — no face calibration", config_path, exc)
        return None

    block = payload.get(FACE_CALIBRATION_KEY)
    if not isinstance(block, dict) or "face_scale" not in block:
        return None
    try:
        return FaceCalibration.from_dict(block)
    except (FaceCalibrationError, TypeError, ValueError) as exc:
        logger.warning("Ignoring stored face calibration: %s", exc)
        return None


#: Relative fx change below which a stored ``face_scale`` is reused untouched.
#: 0.5% is well inside the +-0.35% residual measured for the migration itself
#: (see :func:`resolve_face_scale`), so anything smaller is not worth logging.
FX_MIGRATION_TOLERANCE: float = 0.005


def resolve_face_scale(
    intrinsics: CameraIntrinsics,
    path: Path | None = None,
) -> float | None:
    """
    The ``face_scale`` valid for THESE intrinsics, or ``None`` for legacy mode.

    ``face_scale`` and fx are coupled, and nothing used to record that. tvec.z
    scales with fx, so a scale calibrated at one fx is wrong at another by the
    same ratio -- and wrong silently, because the sanity gate [0.6, 1.6] is far
    too wide to catch an 8% shift.

    MEASURED, not assumed (2026-09-22, FACE_MODEL_MM reprojected at
    tvec=[-40,30,628] with 2.8 px landmark noise, re-solved with SQPNP):

        fx        tvec_z     after x fx0/fx    residual
        1044.4    624.37     624.37            0.000%   <- reference
         960.0    575.90     626.53            0.345%
         800.0    484.10     631.99            1.220%
        1200.0    713.80     621.25           -0.501%

    So the migration ``face_scale * fx_stored / fx_live`` is NOT exact -- it
    leaves 0.3-1.2% (2-7 mm at 600 mm) across the plausible fx range, because
    the reprojection optimum shifts slightly under landmark noise. It is applied
    anyway, and here is the trade: refusing would drop the user to
    ``_monocular_scale``, which pins the scale to whatever distance they happened
    to sit at on frame 1 -- an unbounded error. A bounded 1% beats an unbounded
    one. Recalibrating removes even the 1%.

    Two things the scalar migration CANNOT cover, so they refuse instead:

    - ``fx != fy``. The migration is blind to fy, and the same measurement shows
      a 6% fy anisotropy moves the migrated z by 3.3% (20 mm at 600 mm).
    - cx/cy shifts are fine (+-15 px moved z by 0.2%), so they are not gated.
    """
    calibration = load_face_calibration(path)
    if calibration is None:
        return None

    live_fx = float(intrinsics.fx)
    live_fy = float(intrinsics.fy)
    if live_fx <= 0.0:
        logger.warning("Live fx is %.3f — ignoring stored face_scale", live_fx)
        return None

    stored_fx = float(calibration.intrinsics_fx)
    if stored_fx <= 0.0:
        # Pre-2026-09-22 block. Every calibration that ever existed came from
        # from_image_size at the running resolution -- that was the only code
        # path -- so that is what it was measured with. Stated as an assumption
        # because it is one.
        stored_fx = CameraIntrinsics.from_image_size(
            intrinsics.width, intrinsics.height
        ).fx
        logger.info(
            "Stored face_scale predates intrinsics_fx — assuming it was measured "
            "with the assumed-FOV estimate fx=%.1f px",
            stored_fx,
        )

    ratio = stored_fx / live_fx
    if abs(ratio - 1.0) <= FX_MIGRATION_TOLERANCE:
        return float(calibration.face_scale)

    if abs(live_fy - live_fx) > FX_MIGRATION_TOLERANCE * live_fx:
        logger.error(
            "REFUSING stored face_scale=%.4f: it was measured at fx=%.1f but the "
            "live intrinsics are fx=%.1f fy=%.1f (anisotropic). The fx-ratio "
            "migration cannot see fy, and 6%% of fy anisotropy moves depth 3.3%% "
            "(20 mm at 600 mm). Falling back to the legacy _monocular_scale path "
            "— rerun 'main.py --calibrate' to fix this properly.",
            calibration.face_scale, stored_fx, live_fx, live_fy,
        )
        return None

    migrated = float(calibration.face_scale) * ratio
    if not FACE_SCALE_MIN <= migrated <= FACE_SCALE_MAX:
        logger.error(
            "REFUSING stored face_scale: migrating %.4f from fx=%.1f to fx=%.1f "
            "gives %.4f, outside the sanity gate [%.1f, %.1f]. Falling back to "
            "the legacy path — rerun 'main.py --calibrate'.",
            calibration.face_scale, stored_fx, live_fx, migrated,
            FACE_SCALE_MIN, FACE_SCALE_MAX,
        )
        return None

    logger.warning(
        "Stored face_scale MIGRATED for changed intrinsics: %.4f -> %.4f "
        "(fx %.1f -> %.1f, ratio %.4f). Exact to ~1%% (2-7 mm at 600 mm) — "
        "rerun 'main.py --calibrate' to remove even that.",
        calibration.face_scale, migrated, stored_fx, live_fx, ratio,
    )
    return migrated


def save_face_calibration(
    calibration: FaceCalibration,
    path: Path | None = None,
) -> Path:
    """
    Merge the calibration into user.json, preserving every other key.

    user.json also carries the AppConfig overrides written by the calibration
    wizard, so this must never rewrite the file wholesale.
    """
    config_path = path or default_user_config_path()
    payload: dict = {}
    if config_path.exists():
        try:
            existing = json.loads(config_path.read_text(encoding="utf-8"))
            if isinstance(existing, dict):
                payload = existing
        except (OSError, ValueError) as exc:
            logger.warning("Cannot merge into %s (%s) — rewriting", config_path, exc)

    payload[FACE_CALIBRATION_KEY] = calibration.to_dict()
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    logger.info(
        "Face calibration saved to %s: face_scale=%.4f (D=%.1f mm, "
        "median|tz|=%.1f mm, std=%.1f mm, frames=%d)",
        config_path,
        calibration.face_scale,
        calibration.distance_mm,
        calibration.tz_median_mm,
        calibration.tz_std_mm,
        calibration.valid_frames,
    )
    return config_path
