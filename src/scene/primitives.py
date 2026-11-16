"""Procedural mesh generators."""

from __future__ import annotations

import logging
import math

import cv2
import numpy as np

from src.renderer.mesh import Mesh, MeshData

logger = logging.getLogger(__name__)

RELIEF_CLAMP_GAP_FRACTION = 0.4


def _pack(positions: list, normals: list, indices: list) -> MeshData:
    vertices = np.asarray(positions, dtype=np.float32)
    index_array = np.asarray(indices, dtype=np.uint32)
    return MeshData(vertices=vertices, indices=index_array, vertex_count=len(indices))


def create_plane(width: float, depth: float, y: float = 0.0) -> MeshData:
    """Create a horizontal plane on the XZ plane."""
    hw, hd = width * 0.5, depth * 0.5
    positions = [
        -hw, y, -hd, 0, 1, 0,
        hw, y, -hd, 0, 1, 0,
        hw, y, hd, 0, 1, 0,
        -hw, y, hd, 0, 1, 0,
    ]
    indices = [0, 1, 2, 0, 2, 3]
    return _pack(positions, [], indices)


def create_box(width: float, height: float, depth: float) -> MeshData:
    """Create an axis-aligned box centered at origin."""
    hw, hh, hd = width * 0.5, height * 0.5, depth * 0.5
    corners = [
        (-hw, -hh, -hd),
        (hw, -hh, -hd),
        (hw, hh, -hd),
        (-hw, hh, -hd),
        (-hw, -hh, hd),
        (hw, -hh, hd),
        (hw, hh, hd),
        (-hw, hh, hd),
    ]
    faces = [
        (0, 1, 2, 3, (0, 0, -1)),
        (5, 4, 7, 6, (0, 0, 1)),
        (4, 0, 3, 7, (-1, 0, 0)),
        (1, 5, 6, 2, (1, 0, 0)),
        (3, 2, 6, 7, (0, 1, 0)),
        (4, 5, 1, 0, (0, -1, 0)),
    ]
    positions: list[float] = []
    indices: list[int] = []
    for face in faces:
        base = len(positions) // 6
        nx, ny, nz = face[4]
        for corner_index in face[:4]:
            x, y, z = corners[corner_index]
            positions.extend([x, y, z, nx, ny, nz])
        indices.extend([base, base + 1, base + 2, base, base + 2, base + 3])
    return _pack(positions, [], indices)


def create_sphere(radius: float, segments: int = 24, rings: int = 16) -> MeshData:
    """Create a UV sphere."""
    positions: list[float] = []
    indices: list[int] = []

    for ring in range(rings + 1):
        phi = math.pi * ring / rings
        y = math.cos(phi) * radius
        ring_radius = math.sin(phi) * radius
        for seg in range(segments + 1):
            theta = 2.0 * math.pi * seg / segments
            x = math.cos(theta) * ring_radius
            z = math.sin(theta) * ring_radius
            nx, ny, nz = x / radius, y / radius, z / radius
            positions.extend([x, y, z, nx, ny, nz])

    row = segments + 1
    for ring in range(rings):
        for seg in range(segments):
            current = ring * row + seg
            next_row = (ring + 1) * row + seg
            indices.extend([current, next_row, current + 1])
            indices.extend([current + 1, next_row, next_row + 1])

    return _pack(positions, [], indices)


def _merge_mesh_parts(parts: list[MeshData]) -> MeshData:
    """Combine multiple mesh parts into one draw call."""
    positions: list[float] = []
    indices: list[int] = []
    vertex_offset = 0
    for part in parts:
        verts = part.vertices.reshape(-1, 6)
        for row in verts:
            positions.extend(float(v) for v in row)
        indices.extend(int(idx) + vertex_offset for idx in part.indices)
        vertex_offset += len(verts)
    return _pack(positions, [], indices)


def create_vertical_plane_at(
    center_x: float,
    center_y: float,
    width: float,
    height: float,
    z: float,
) -> MeshData:
    """Vertical quad centered at (center_x, center_y) facing +Z."""
    hw, hh = width * 0.5, height * 0.5
    positions = [
        center_x - hw, center_y - hh, z, 0.0, 0.0, 1.0,
        center_x + hw, center_y - hh, z, 0.0, 0.0, 1.0,
        center_x + hw, center_y + hh, z, 0.0, 0.0, 1.0,
        center_x - hw, center_y + hh, z, 0.0, 0.0, 1.0,
    ]
    indices = [0, 1, 2, 0, 2, 3]
    return _pack(positions, [], indices)


def create_window_frame_bezel(
    inner_width: float,
    inner_height: float,
    bezel_width: float,
    z: float,
) -> MeshData:
    """Hollow rectangular bezel (four bars) around a content opening."""
    iw, ih = inner_width, inner_height
    b = bezel_width
    outer_w = iw + 2.0 * b
    top_y = ih * 0.5 + b * 0.5
    bottom_y = -ih * 0.5 - b * 0.5
    left_x = -iw * 0.5 - b * 0.5
    right_x = iw * 0.5 + b * 0.5
    return _merge_mesh_parts(
        [
            create_vertical_plane_at(0.0, top_y, outer_w, b, z),
            create_vertical_plane_at(0.0, bottom_y, outer_w, b, z),
            create_vertical_plane_at(left_x, 0.0, b, ih, z),
            create_vertical_plane_at(right_x, 0.0, b, ih, z),
        ]
    )


def create_open_box(
    width: float,
    height: float,
    z_front: float,
    z_back: float,
) -> MeshData:
    """
    Hộp năm mặt (sàn, trần, hai vách, vách sau), **mở về phía người xem**.

    Pháp tuyến hướng vào TRONG hộp vì người xem đứng trong lòng nó. Mặt trước
    cố ý không có: đó là cái miệng hộp, và miệng phải trùng đúng hình chữ nhật
    màn hình thì bốn góc mới chiếu đúng vào bốn góc khung nhìn (PB-034).
    """
    hw, hh = width * 0.5, height * 0.5
    zf, zb = float(z_front), float(z_back)
    quads = [
        # (bốn đỉnh theo chiều kim đồng hồ nhìn từ trong hộp, pháp tuyến)
        ([(-hw, -hh, zb), (hw, -hh, zb), (hw, -hh, zf), (-hw, -hh, zf)],
         (0.0, 1.0, 0.0)),                                  # sàn
        ([(-hw, hh, zf), (hw, hh, zf), (hw, hh, zb), (-hw, hh, zb)],
         (0.0, -1.0, 0.0)),                                 # trần
        ([(-hw, -hh, zf), (-hw, -hh, zb), (-hw, hh, zb), (-hw, hh, zf)],
         (1.0, 0.0, 0.0)),                                  # vách trái
        ([(hw, -hh, zb), (hw, -hh, zf), (hw, hh, zf), (hw, hh, zb)],
         (-1.0, 0.0, 0.0)),                                 # vách phải
        ([(-hw, -hh, zb), (-hw, hh, zb), (hw, hh, zb), (hw, -hh, zb)],
         (0.0, 0.0, 1.0)),                                  # vách sau
    ]
    positions: list[float] = []
    indices: list[int] = []
    for corners, normal in quads:
        base = len(positions) // 6
        for x, y, z in corners:
            positions.extend([x, y, z, *normal])
        indices.extend([base, base + 1, base + 2, base, base + 2, base + 3])
    return _pack(positions, [], indices)


def create_vertical_plane(width: float, height: float, z: float = 0.0) -> MeshData:
    """Vertical quad in XY facing +Z (Phong mesh with normals)."""
    hw, hh = width * 0.5, height * 0.5
    positions = [
        -hw, -hh, z, 0.0, 0.0, 1.0,
        hw, -hh, z, 0.0, 0.0, 1.0,
        hw, hh, z, 0.0, 0.0, 1.0,
        -hw, hh, z, 0.0, 0.0, 1.0,
    ]
    indices = [0, 1, 2, 0, 2, 3]
    return _pack(positions, [], indices)


def sample_depth_map(depth: np.ndarray, u: float, v: float) -> float:
    """Bilinear sample of a depth map. UV v=0 is bottom, v=1 is top."""
    h, w = depth.shape
    px = float(np.clip(u * (w - 1), 0.0, w - 1))
    py = float(np.clip((1.0 - v) * (h - 1), 0.0, h - 1))
    x0, y0 = int(px), int(py)
    x1, y1 = min(x0 + 1, w - 1), min(y0 + 1, h - 1)
    tx, ty = px - x0, py - y0
    d00, d10 = float(depth[y0, x0]), float(depth[y0, x1])
    d01, d11 = float(depth[y1, x0]), float(depth[y1, x1])
    return (1.0 - tx) * (1.0 - ty) * d00 + tx * (1.0 - ty) * d10 + (1.0 - tx) * ty * d01 + tx * ty * d11


def sample_mask_map(mask: np.ndarray, u: float, v: float) -> float:
    """Bilinear sample of a boolean/float mask at normalized UV (0-1)."""
    h, w = mask.shape[:2]
    if h < 1 or w < 1:
        return 0.0
    # Map v so image top (v=1) matches depth/mask row 0 after flip conventions.
    fy = float(np.clip(1.0 - v, 0.0, 1.0)) * (h - 1)
    fx = float(np.clip(u, 0.0, 1.0)) * (w - 1)
    x0, y0 = int(fx), int(fy)
    x1 = min(x0 + 1, w - 1)
    y1 = min(y0 + 1, h - 1)
    tx, ty = fx - x0, fy - y0
    m00, m10 = float(mask[y0, x0]), float(mask[y0, x1])
    m01, m11 = float(mask[y1, x0]), float(mask[y1, x1])
    return (1.0 - tx) * (1.0 - ty) * m00 + tx * (1.0 - ty) * m10 + (1.0 - tx) * ty * m01 + tx * ty * m11


def _sample_bilinear_grid(
    arr: np.ndarray, u: np.ndarray, v: np.ndarray
) -> np.ndarray:
    """
    Vectorised bilinear sample of ``arr`` at UV grids (v=0 bottom, v=1 top).

    Same formula as :func:`sample_depth_map` / :func:`sample_mask_map`, but over
    whole arrays.  The scalar versions were called once per mesh vertex from a
    nested Python loop: 97x97 = 9409 vertices x 2 samplers per layer, measured
    at 413 ms for a single layer.
    """
    h, w = arr.shape[:2]
    px = np.clip(u * (w - 1), 0.0, w - 1)
    py = np.clip((1.0 - v) * (h - 1), 0.0, h - 1)
    x0 = px.astype(np.int32)
    y0 = py.astype(np.int32)
    x1 = np.minimum(x0 + 1, w - 1)
    y1 = np.minimum(y0 + 1, h - 1)
    tx = px - x0
    ty = py - y0
    a = arr.astype(np.float64)
    d00 = a[y0, x0]
    d10 = a[y0, x1]
    d01 = a[y1, x0]
    d11 = a[y1, x1]
    return (
        (1 - tx) * (1 - ty) * d00
        + tx * (1 - ty) * d10
        + (1 - tx) * ty * d01
        + tx * ty * d11
    )


def create_layer_relief_vertical_plane_uv(
    width: float,
    height: float,
    depth_map: np.ndarray,
    layer_mask: np.ndarray,
    *,
    z_base: float,
    depth_lo: float,
    depth_hi: float,
    mean_depth: float,
    depth_norm: float,
    inter_layer_gap_m: float,
    relief_amplitude: float = 0.25,
    relief_far_falloff: float = 0.3,
    relief_smooth_sigma: float = 1.5,
    segments_x: int = 96,
    segments_y: int = 96,
    layer_index: int = 1,
    layer_count: int = 1,
) -> "TexturedMeshData":
    """
    Subdivided layer quad with small intra-layer Z relief centered on the layer mean.

    Relief is applied once at mesh-build time; inter-layer ``z_base`` (occlusion gap)
    is unchanged. Displacement is clamped so vertices never cross adjacent layers.
    """
    from src.content.layer_clustering import relief_depth_scale
    from src.renderer.textured_mesh import TexturedMeshData

    sx = max(2, int(segments_x))
    sy = max(2, int(segments_y))
    depth_span = max(float(depth_hi - depth_lo), 1e-6)
    mean_norm = float(np.clip((mean_depth - depth_lo) / depth_span, 0.0, 1.0))
    gap = max(float(inter_layer_gap_m), 1e-6)
    max_relief_m = float(relief_amplitude) * gap
    clamp_m = RELIEF_CLAMP_GAP_FRACTION * gap
    scale = relief_depth_scale(depth_norm, relief_far_falloff)
    target_pct = float(relief_amplitude) * 100.0

    u_grid, v_grid = np.meshgrid(
        np.arange(sx + 1, dtype=np.float64) / sx,
        np.arange(sy + 1, dtype=np.float64) / sy,
    )
    d_grid = _sample_bilinear_grid(depth_map, u_grid, v_grid)
    local_norm = np.clip((d_grid - depth_lo) / depth_span, 0.0, 1.0)
    mask_w = _sample_bilinear_grid(layer_mask.astype(np.float32), u_grid, v_grid)
    relief = (
        (local_norm - mean_norm) * max_relief_m * scale * mask_w
    ).astype(np.float32)

    mask_pixels = layer_mask.astype(bool)
    if bool(mask_pixels.any()):
        layer_depths = depth_map[mask_pixels]
        local_norm_pixels = np.clip((layer_depths - depth_lo) / depth_span, 0.0, 1.0)
        depth_var_std = float(local_norm_pixels.std())
    else:
        depth_var_std = 0.0

    pre_min = float(relief.min())
    pre_max = float(relief.max())
    pre_peak = float(np.max(np.abs(relief)))

    sigma = max(0.0, float(relief_smooth_sigma))
    if sigma > 0.0:
        ksize = max(3, int(sigma * 6.0) | 1)
        relief = cv2.GaussianBlur(relief, (ksize, ksize), sigma)

    post_min = float(relief.min())
    post_max = float(relief.max())
    post_peak = float(np.max(np.abs(relief)))

    relief = np.clip(relief, -clamp_m, clamp_m)
    final_min = float(relief.min())
    final_max = float(relief.max())
    peak = float(np.max(np.abs(relief)))
    if peak > clamp_m + 1e-6:
        logger.warning(
            "Layer relief peak %.4f m exceeds clamp %.4f m — check amplitude",
            peak,
            clamp_m,
        )

    gap_mm = gap * 1000.0
    logger.info(
        "Layer %d/%d: inter-layer gap=%.1f mm | relief amplitude actual=%.2f..%.2f mm "
        "(target ~%.0f%% of gap) | depth_scale(t=%.2f)=%.3f",
        layer_index,
        layer_count,
        gap_mm,
        final_min * 1000.0,
        final_max * 1000.0,
        target_pct,
        depth_norm,
        scale,
    )
    logger.info(
        "Layer %d/%d relief diagnostics: depth_norm_std=%.4f | pre-smooth=%.2f..%.2f mm "
        "(peak %.2f) | post-smooth=%.2f..%.2f mm (peak %.2f) | sigma=%.2f | "
        "clamp=±%.2f mm",
        layer_index,
        layer_count,
        depth_var_std,
        pre_min * 1000.0,
        pre_max * 1000.0,
        pre_peak * 1000.0,
        post_min * 1000.0,
        post_max * 1000.0,
        post_peak * 1000.0,
        sigma,
        clamp_m * 1000.0,
    )

    hw, hh = width * 0.5, height * 0.5
    xs = -hw + u_grid * width
    ys = -hh + v_grid * height
    zs = z_base + relief.astype(np.float64)
    verts = np.stack([xs, ys, zs, u_grid, v_grid], axis=-1).ravel()

    indices: list[int] = []
    row = sx + 1
    for j in range(sy):
        for i in range(sx):
            i0 = j * row + i
            i1 = i0 + 1
            i2 = i0 + row
            i3 = i2 + 1
            indices.extend([i0, i1, i3, i0, i3, i2])

    vertices = np.asarray(verts, dtype=np.float32)
    index_array = np.asarray(indices, dtype=np.uint32)
    return TexturedMeshData(
        vertices=vertices,
        indices=index_array,
        vertex_count=len(indices),
    )


def create_displaced_vertical_plane_uv(
    width: float,
    height: float,
    depth_map: np.ndarray,
    depth_scale: float,
    segments_x: int,
    segments_y: int,
    z_base: float = 0.0,
) -> "TexturedMeshData":
    """
    Subdivided vertical plane with per-vertex Z displacement from a depth map.

    Depth 1.0 (near) displaces toward the viewer (+Z); 0.0 stays at ``z_base``.
    """
    from src.renderer.textured_mesh import TexturedMeshData

    sx = max(2, int(segments_x))
    sy = max(2, int(segments_y))
    hw, hh = width * 0.5, height * 0.5
    u_grid, v_grid = np.meshgrid(
        np.arange(sx + 1, dtype=np.float64) / sx,
        np.arange(sy + 1, dtype=np.float64) / sy,
    )
    xs = -hw + u_grid * width
    ys = -hh + v_grid * height
    zs = z_base + _sample_bilinear_grid(depth_map, u_grid, v_grid) * depth_scale
    verts = np.stack([xs, ys, zs, u_grid, v_grid], axis=-1).ravel()

    indices: list[int] = []
    row = sx + 1
    for j in range(sy):
        for i in range(sx):
            i0 = j * row + i
            i1 = i0 + 1
            i2 = i0 + row
            i3 = i2 + 1
            indices.extend([i0, i1, i3, i0, i3, i2])

    vertices = np.asarray(verts, dtype=np.float32)
    index_array = np.asarray(indices, dtype=np.uint32)
    return TexturedMeshData(
        vertices=vertices,
        indices=index_array,
        vertex_count=len(indices),
    )


def create_vertical_plane_uv(
    width: float, height: float, z: float = 0.0
) -> "TexturedMeshData":
    """
    Vertical quad in XY facing +Z, with UVs for video texturing.

    The plane sits behind the physical monitor (negative Z into the scene)
    so head movement reveals parallax against the surrounding void.
    """
    from src.renderer.textured_mesh import TexturedMeshData

    hw, hh = width * 0.5, height * 0.5
    vertices = np.array(
        [
            -hw, -hh, z, 0.0, 0.0,
            hw, -hh, z, 1.0, 0.0,
            hw, hh, z, 1.0, 1.0,
            -hw, hh, z, 0.0, 1.0,
        ],
        dtype=np.float32,
    )
    indices = np.array([0, 1, 2, 0, 2, 3], dtype=np.uint32)
    return TexturedMeshData(vertices=vertices, indices=indices, vertex_count=6)


def create_mesh(data: MeshData) -> Mesh:
    """Upload procedural mesh data to GPU."""
    return Mesh(data)
