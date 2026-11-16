#!/usr/bin/env python3
"""
Convert NYU Depth V2 labeled .mat (HDF5 v7.3) to rgb/ + depth/ PNG folders.

Standard split: last 654 frames → test set (indices 795–1448 of 1449 total).

Example:
    python convert_nyu.py --mat nyu_depth_v2_labeled.mat --out ./nyu_test
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import cv2
import numpy as np

logger = logging.getLogger(__name__)

NYU_TOTAL = 1449
NYU_TEST_COUNT = 654
# Official nyu_depth_v2_labeled.mat (HDF5 v7.3) is ~2.77 GiB on disk.
NYU_LABELED_MAT_BYTES = 2_972_037_809
# Allow small server/header variance; reject clearly truncated files.
NYU_LABELED_MAT_MIN_BYTES = 2_900_000_000


def _load_h5py():
    try:
        import h5py
    except ImportError as exc:
        raise SystemExit(
            "h5py is required: pip install h5py"
        ) from exc
    return h5py


def _normalize_images(arr: np.ndarray) -> np.ndarray:
    """Return uint8 array shaped (N, H, W, 3) RGB."""
    data = np.asarray(arr)
    if data.ndim != 4:
        raise ValueError(f"Expected 4D images, got shape {data.shape}")

    # N x 3 x H x W (documented layout)
    if data.shape[1] == 3:
        return np.transpose(data, (0, 2, 3, 1))

    # N x H x W x 3
    if data.shape[-1] == 3:
        return data

    # HDF5/MATLAB column-major: H x W x 3 x N
    if data.shape[-1] >= NYU_TOTAL - 1 and data.shape[2] == 3:
        return np.transpose(data, (3, 0, 1, 2))

    # 3 x H x W x N
    if data.shape[0] == 3 and data.shape[-1] >= NYU_TOTAL - 1:
        return np.transpose(data, (3, 1, 2, 0))

    raise ValueError(f"Unrecognized images layout: {data.shape}")


def _normalize_depths(arr: np.ndarray) -> np.ndarray:
    """Return float32 array shaped (N, H, W) in meters."""
    data = np.asarray(arr, dtype=np.float32)
    if data.ndim != 3:
        raise ValueError(f"Expected 3D depths, got shape {data.shape}")

    # N x H x W
    if data.shape[0] >= NYU_TOTAL - 1:
        return data

    # H x W x N
    if data.shape[-1] >= NYU_TOTAL - 1:
        return np.transpose(data, (2, 0, 1))

    raise ValueError(f"Unrecognized depths layout: {data.shape}")


def _check_mat_file_size(mat_path: Path) -> None:
    """Fail fast when the .mat download is incomplete or suspiciously large."""
    size = mat_path.stat().st_size
    if NYU_LABELED_MAT_MIN_BYTES <= size <= NYU_LABELED_MAT_BYTES * 1.02:
        return
    if size < NYU_LABELED_MAT_MIN_BYTES:
        pct = 100.0 * size / NYU_LABELED_MAT_BYTES
        raise OSError(
            f"Incomplete NYU .mat file: {size:,} bytes ({pct:.1f}% of "
            f"{NYU_LABELED_MAT_BYTES:,} expected). "
            "Delete the partial file and re-download:\n"
            '  curl.exe -L -o nyu_depth_v2_labeled.mat '
            '"https://horatio.cs.nyu.edu/mit/silberman/nyu_depth_v2/nyu_depth_v2_labeled.mat"'
        )
    raise OSError(
        f"NYU .mat file size looks wrong: {size:,} bytes "
        f"(expected ~{NYU_LABELED_MAT_BYTES:,}). "
        "Delete it and download again from scratch (do not use -C - resume "
        "on a truncated file)."
    )


def load_nyu_mat(mat_path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Load RGB (N,H,W,3) and depth (N,H,W) from NYU labeled .mat."""
    _check_mat_file_size(mat_path)
    h5py = _load_h5py()
    logger.info("Loading %s …", mat_path)
    with h5py.File(mat_path, "r") as handle:
        if "images" not in handle or "depths" not in handle:
            keys = list(handle.keys())
            raise KeyError(f"Expected 'images' and 'depths' in .mat; found {keys}")
        images = _normalize_images(handle["images"][()])
        depths = _normalize_depths(handle["depths"][()])

    if images.shape[0] != depths.shape[0]:
        raise ValueError(
            f"Image/depth count mismatch: {images.shape[0]} vs {depths.shape[0]}"
        )
    logger.info(
        "Loaded %d frames — RGB %s, depth %s",
        images.shape[0],
        images.shape,
        depths.shape,
    )
    return images, depths


def depth_to_uint16_mm(depth_m: np.ndarray) -> np.ndarray:
    """Convert depth in meters to millimeters as uint16."""
    depth_mm = np.nan_to_num(depth_m, nan=0.0, posinf=0.0, neginf=0.0) * 1000.0
    return np.clip(depth_mm, 0.0, 65535.0).astype(np.uint16)


def _image_frame_rgb(images_ds, index: int) -> np.ndarray:
    """Return one RGB frame as uint8 (H, W, 3)."""
    frame = np.asarray(images_ds[index], dtype=np.uint8)
    if frame.ndim == 3 and frame.shape[0] == 3:
        return np.transpose(frame, (1, 2, 0))
    if frame.ndim == 3 and frame.shape[-1] == 3:
        return frame
    raise ValueError(f"Unrecognized image frame shape: {frame.shape}")


def _depth_frame_meters(handle, index: int) -> np.ndarray:
    """Return one depth frame in meters, preferring filled `depths`."""
    if "depths" in handle:
        try:
            return np.asarray(handle["depths"][index], dtype=np.float32)
        except OSError:
            logger.warning(
                "Frame %d: corrupted entry in 'depths', using 'rawDepths' instead",
                index,
            )
    if "rawDepths" not in handle:
        raise KeyError("Neither readable 'depths' nor 'rawDepths' found in .mat")
    return np.asarray(handle["rawDepths"][index], dtype=np.float32)


def _frame_count(handle) -> int:
    if "images" in handle:
        return int(handle["images"].shape[0])
    if "depths" in handle:
        return int(handle["depths"].shape[0])
    if "rawDepths" in handle:
        return int(handle["rawDepths"].shape[0])
    raise KeyError("No frame datasets found in .mat")


def export_split_streaming(
    mat_path: Path,
    out_dir: Path,
    indices: range,
) -> int:
    """Write selected frames directly from HDF5 without loading the full .mat."""
    h5py = _load_h5py()
    rgb_dir = out_dir / "rgb"
    depth_dir = out_dir / "depth"
    rgb_dir.mkdir(parents=True, exist_ok=True)
    depth_dir.mkdir(parents=True, exist_ok=True)

    count = 0
    with h5py.File(mat_path, "r") as handle:
        if "images" not in handle:
            raise KeyError("Expected 'images' in .mat")
        n = _frame_count(handle)
        for out_idx, src_idx in enumerate(indices, start=1):
            if src_idx < 0 or src_idx >= n:
                raise IndexError(f"Frame index {src_idx} out of range (0–{n - 1})")
            name = f"{out_idx:04d}.png"
            rgb = _image_frame_rgb(handle["images"], src_idx)
            bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            if not cv2.imwrite(str(rgb_dir / name), bgr):
                raise OSError(f"Failed to write {rgb_dir / name}")

            depth_u16 = depth_to_uint16_mm(_depth_frame_meters(handle, src_idx))
            if not cv2.imwrite(str(depth_dir / name), depth_u16):
                raise OSError(f"Failed to write {depth_dir / name}")
            count += 1
            if count % 50 == 0:
                logger.info("Exported %d / %d frames", count, len(indices))

    return count


def convert(mat_path: Path, out_dir: Path, test_only: bool = True) -> None:
    _check_mat_file_size(mat_path)
    h5py = _load_h5py()
    with h5py.File(mat_path, "r") as handle:
        n = _frame_count(handle)
    if n < NYU_TEST_COUNT:
        raise ValueError(f"Expected at least {NYU_TEST_COUNT} frames, got {n}")

    if test_only:
        start = n - NYU_TEST_COUNT
        indices = range(start, n)
        logger.info(
            "Exporting test split: indices %d–%d (%d frames) → %s",
            start,
            n - 1,
            len(indices),
            out_dir,
        )
    else:
        indices = range(n)
        logger.info("Exporting all %d frames → %s", n, out_dir)

    written = export_split_streaming(mat_path, out_dir, indices)
    logger.info("Done — wrote %d RGB + depth pairs to %s", written, out_dir)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert NYU Depth V2 labeled .mat to rgb/depth PNG folders.",
    )
    parser.add_argument(
        "--mat",
        type=Path,
        required=True,
        help="Path to nyu_depth_v2_labeled.mat",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("nyu_test"),
        help="Output directory (default: ./nyu_test)",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Export all 1449 frames instead of test split only",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Debug logging",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s | %(message)s",
    )

    mat_path = args.mat.expanduser().resolve()
    if not mat_path.is_file():
        logger.error("MAT file not found: %s", mat_path)
        return 1

    out_dir = args.out.expanduser().resolve()
    try:
        convert(mat_path, out_dir, test_only=not args.all)
    except Exception:
        logger.exception("Conversion failed")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
