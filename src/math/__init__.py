"""Linear algebra, projection, and coordinate utilities.

There is exactly ONE basis change between the OpenCV camera frame and HCPD
world, and it lives in :data:`src.pose.head_pose.R_CV_TO_HCPD`. This package
deliberately exports no second copy of it.

Two modules used to sit here and did exactly that:

``coordinates.py``
    ``screen_mm_to_world`` computed ``point_mm - webcam_offset_mm`` while the
    live path (``head_pose._opencv_to_hcpd_translation``) *adds* the offset.
    Subtracting instead of adding is the bug that once cost 206 mm (see
    docs/document.md §3). It had no callers and no tests, so it was a landmine
    waiting for whoever needed "the other" conversion helper. Its siblings
    ``opencv_point_to_opengl`` / ``opencv_to_opengl_matrix`` applied
    ``diag(1,-1,-1)`` — a third convention, different again from both of the
    above.

``frustum.py``
    A second off-axis projection built in **millimetres** (``near_mm=1.0``,
    ``far_mm=5000``) while the live Kooima path in ``projection.py`` works in
    **metres**. Also no callers and no tests. Mixing the two silently gives
    wrong near/far.

Deleted rather than repaired: two copies of one convention is the trap this
project has been caught by repeatedly, and the repair would have left the
copies in place.
"""

from src.math.homogeneous import (
    identity4,
    invert,
    multiply,
    scale_matrix,
    to_column_major_flat,
    transform_direction,
    transform_point,
    translation_matrix,
)
from src.math.pose_utils import Pose6DoF, compose_poses, identity_pose, interpolate_pose, invert_pose
from src.math.projection import (
    ComputeOffAxisProjection,
    compute_off_axis_projection,
    off_axis_projection,
)
from src.math.quaternion import (
    conjugate,
    from_axis_angle,
    from_rotation_matrix,
    identity_quat,
    multiply as quat_multiply,
    normalize,
    rotate_vector,
    to_rotation_matrix,
)

__all__ = [
    "Pose6DoF",
    "compose_poses",
    "conjugate",
    "from_axis_angle",
    "from_rotation_matrix",
    "identity4",
    "identity_pose",
    "identity_quat",
    "interpolate_pose",
    "invert",
    "invert_pose",
    "multiply",
    "normalize",
    "off_axis_projection",
    "ComputeOffAxisProjection",
    "compute_off_axis_projection",
    "quat_multiply",
    "rotate_vector",
    "scale_matrix",
    "to_column_major_flat",
    "to_rotation_matrix",
    "transform_direction",
    "transform_point",
    "translation_matrix",
]
