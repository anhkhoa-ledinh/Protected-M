"""Wavefront OBJ mesh loader."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from src.renderer.mesh import Mesh, MeshData


def load_obj(path: Path) -> MeshData:
    """
    Load a simple OBJ file with triangular faces.

    CẢNH BÁO — CHỈ NHẬN FACE TAM GIÁC, VÀ IM LẶNG VỨT HÌNH HỌC KHI KHÔNG PHẢI.

    Vòng lặp đọc face dưới đây lấy ``parts[1:4]``, tức **ba corner đầu của mọi
    face**. File OBJ toàn quad thì mỗi face mất 1 corner; n-gon thì mất nhiều
    hơn. Không có cảnh báo nào, và ảnh dựng ra vẫn "trông có vẻ đúng" — đó là
    chỗ nguy hiểm. Trên một model thử ở dự án anh em (7388 quad + 41 n-gon tới
    13 đỉnh / 10869 face) sẽ mất hơn nửa hình học.

    **Hiện KHÔNG ai gọi hàm này.** Nó được export trong ``src/renderer/__init__``
    và ``load_obj_mesh`` ngay dưới có gọi, nhưng bản thân ``load_obj_mesh`` cũng
    không nơi nào dùng. Đây là **lỗi tiềm ẩn trong code chết**, không phải lỗi
    đang chạy — nên ghi lại chứ không sửa (PHẦN V mục 11 trong
    docs/document.md). Trước khi dùng lại hàm này: hoặc tam giác hoá face,
    hoặc ít nhất báo lỗi khi gặp face nhiều hơn 3 corner.
    """
    positions: list[list[float]] = []
    normals: list[list[float]] = []
    faces: list[tuple[int, int]] = []

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        tag = parts[0]
        if tag == "v":
            positions.append([float(v) for v in parts[1:4]])
        elif tag == "vn":
            normals.append([float(v) for v in parts[1:4]])
        elif tag == "f":
            for corner in parts[1:4]:
                tokens = corner.split("/")
                vi = int(tokens[0]) - 1
                ni = int(tokens[2]) - 1 if len(tokens) > 2 and tokens[2] else vi
                faces.append((vi, ni))

    if not positions:
        raise ValueError(f"OBJ has no vertices: {path}")

    if not normals:
        normals = [[0.0, 1.0, 0.0] for _ in positions]

    vertex_map: dict[tuple[int, int], int] = {}
    packed: list[float] = []
    indices: list[int] = []

    for vi, ni in faces:
        key = (vi, ni)
        if key not in vertex_map:
            vertex_map[key] = len(packed) // 6
            px, py, pz = positions[vi]
            nx, ny, nz = normals[ni]
            packed.extend([px, py, pz, nx, ny, nz])
        indices.append(vertex_map[key])

    vertices = np.asarray(packed, dtype=np.float32)
    index_array = np.asarray(indices, dtype=np.uint32)
    return MeshData(
        vertices=vertices,
        indices=index_array,
        vertex_count=int(index_array.shape[0]),
    )


def load_obj_mesh(path: Path) -> Mesh:
    """Load OBJ file and upload mesh to GPU."""
    return Mesh(load_obj(path))
