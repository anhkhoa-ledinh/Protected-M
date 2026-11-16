"""Eye position estimation from head pose and iris landmarks."""

from __future__ import annotations

import numpy as np

from src.math.pose_utils import Pose6DoF
from src.pose.head_pose import HeadPoseResult, z_diagnostics
from src.tracking.iris import iris_midpoint
from src.tracking.landmarks import LandmarkSet


class EyePositionEstimator:
    """
    Refine head pose translation to the midpoint between irises.

    The virtual camera position equals the eye position in world space.

    solvePnP anchors on the nose tip (``FACE_MODEL_MM``'s origin); the viewer
    actually looks from between their eyes. This class interpolates from the PnP
    anchor toward an iris-derived estimate of the same quantity -- two estimates
    of one position, combined, in ONE origin (screen centre) and ONE set of axes
    (HCPD). Getting either of those wrong was the bug; see :meth:`estimate`.

    X and Y only. Z is passed through untouched -- the un-projection needs a
    depth to work at all, so it cannot also produce one.
    """

    #: How far to move from the PnP anchor toward the iris estimate, per frame.
    #: A true fraction now: 0.0 keeps the PnP result, 1.0 lands exactly on the
    #: iris estimate. It used to be the weight of an *added* absolute position,
    #: which is why 0.12 produced a 1.12x gain instead of a 12% blend.
    IRIS_BLEND: float = 0.12

    def __init__(
        self,
        ipd_mm: float = 63.0,
        fx: float = 640.0,
        cx: float = 640.0,
        cy: float = 360.0,
        fy: float | None = None,
        webcam_offset_mm: np.ndarray | None = None,
    ) -> None:
        self._ipd_mm = ipd_mm
        self._fx = fx
        self._cx = cx
        self._cy = cy
        self._fy = fx if fy is None else fy
        # The webcam's position in world/HCPD mm. Required to put the iris
        # estimate in the SAME origin as head_pose.translation_mm, which is
        # screen-centred. Leaving it at zeros treats the lens as sitting at the
        # screen centre; with the shipped screen.json (+103 mm on Y) that is a
        # constant -12.4 mm vertical bias on the blended result.
        self._webcam_offset_mm = (
            np.zeros(3, dtype=np.float64)
            if webcam_offset_mm is None
            else np.asarray(webcam_offset_mm, dtype=np.float64).reshape(3)
        )

    def set_webcam_offset_mm(self, webcam_offset_mm: np.ndarray) -> None:
        """Update the webcam's world position (mm, HCPD axes)."""
        self._webcam_offset_mm = np.asarray(
            webcam_offset_mm, dtype=np.float64
        ).reshape(3)

    def estimate(
        self,
        head_pose: HeadPoseResult,
        landmarks: LandmarkSet,
        timestamp: float,
    ) -> Pose6DoF:
        """Compute eye-centered 6DoF pose for virtual camera."""
        if not head_pose.detected:
            return head_pose.pose

        iris_px = iris_midpoint(landmarks)
        translation_mm = head_pose.translation_mm

        # Step 1 — un-project the iris midpoint, in the CAMERA's origin.
        #
        # translation_mm[2] is screen-centred, so the camera-frame depth is it
        # minus the lens' own Z. Only 5 mm with the shipped screen.json, but
        # mixing the two origins is the whole defect being fixed here, and
        # getting it right makes the zero-refinement invariant below exact
        # rather than exact-to-0.1-mm.
        depth = max(
            float(translation_mm[2]) - float(self._webcam_offset_mm[2]), 250.0
        )
        # Both axes negated: image +X and +Y run along world -X and -Y
        # (see R_CV_TO_HCPD in head_pose.py), so these are already HCPD axes.
        # dy divides by fy, not fx: they are equal for from_image_size, but a
        # real calibrateCamera block makes the difference a silent fy/fx - 1.
        iris_cam_mm = np.array(
            [
                -(float(iris_px[0]) - self._cx) * depth / self._fx,
                -(float(iris_px[1]) - self._cy) * depth / self._fy,
            ],
            dtype=np.float64,
        )

        # Step 2 — same origin as translation_mm: screen centre, not the lens.
        iris_world_mm = iris_cam_mm + self._webcam_offset_mm[:2]

        # Step 3 — BLEND, not add.
        #
        # iris_world_mm and translation_mm[:2] are two estimates of the SAME
        # quantity: the viewer's lateral position in world mm. The old code did
        #     eye[:2] += 0.12 * iris_cam_mm
        # which adds 12% of one absolute position to 100% of another. Measured:
        # X came out at 1.12x the true offset (+12.1 mm at a true 100 mm), and Y
        # additionally carried 0.12 * -camera_offset_y = -12.4 mm at every
        # position, because iris_cam_mm was lens-centred and translation_mm was
        # screen-centred.
        #
        # Interpolating gives 0.88 * PnP + 0.12 * iris. Its defining property:
        # when the iris projects exactly where the PnP anchor says it should,
        # the correction is EXACTLY zero. The old form could not be zero.
        eye_translation = translation_mm.copy()
        eye_translation[:2] += self.IRIS_BLEND * (
            iris_world_mm - translation_mm[:2]
        )

        # PHASE A diagnostics: iris refinement touches X and Y only — Z is
        # passed through untouched, so z_after_iris == z_after_scale + the
        # webcam Z offset added by _opencv_to_hcpd_translation.
        z_diagnostics.set(z_after_iris=float(eye_translation[2]))

        return Pose6DoF(
            translation_mm=eye_translation,
            rotation=head_pose.pose.rotation,
            timestamp=timestamp,
        )

    def update_intrinsics(
        self,
        fx: float,
        cx: float,
        cy: float,
        fy: float | None = None,
    ) -> None:
        """
        Update camera intrinsics used for iris depth scaling.

        ``fy`` defaults to ``fx`` — square pixels — which is what
        ``CameraIntrinsics.from_image_size`` produces. It only diverges once a
        real ``cv2.calibrateCamera`` block is in play, and then the vertical iris
        offset needs the real fy: dividing dy by fx is a silent error of
        ``fy/fx - 1``.
        """
        self._fx = fx
        self._cx = cx
        self._cy = cy
        self._fy = fx if fy is None else fy
