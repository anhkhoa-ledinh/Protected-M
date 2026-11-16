"""Persist content settings to content.json."""

from __future__ import annotations

import json
from pathlib import Path

from src.config.schema import ContentSettings


def media_path_for_config(path: Path, project_root: Path) -> str:
    """Prefer a project-relative path for config storage."""
    try:
        rel = path.resolve().relative_to(project_root.resolve())
        return rel.as_posix()
    except ValueError:
        return str(path.resolve())


def save_content_settings(
    config_path: Path,
    content: ContentSettings,
    media_path: Path,
    content_type: str,
    project_root: Path,
) -> None:
    """Write updated media path and type back to content.json."""
    stored = media_path_for_config(media_path, project_root)
    payload: dict = {}
    if config_path.exists():
        payload = json.loads(config_path.read_text(encoding="utf-8"))

    payload.update(
        {
            "enabled": content.enabled,
            "mode": content.mode,
            "content_path": stored,
            "content_type": content_type,
            "video_path": stored if content_type == "video" else payload.get("video_path", stored),
            "loop_video": content.loop_video,
            "plane_position_m": list(content.plane_position_m),
            "plane_width_m": content.plane_width_m,
            "plane_height_m": content.plane_height_m,
            "frame_z_m": content.frame_z_m,
            "frame_bezel_m": content.frame_bezel_m,
            "frame_color": list(content.frame_color),
            "mid_layer_z_m": content.mid_layer_z_m,
            "mid_layer_width_m": content.mid_layer_width_m,
            "mid_layer_height_m": content.mid_layer_height_m,
            "mid_layer_offset_x_m": content.mid_layer_offset_x_m,
            "mid_layer_color_a": list(content.mid_layer_color_a),
            "mid_layer_color_b": list(content.mid_layer_color_b),
            "backdrop_z_m": content.backdrop_z_m,
            "backdrop_width_m": content.backdrop_width_m,
            "backdrop_height_m": content.backdrop_height_m,
            "backdrop_color": list(content.backdrop_color),
            "void_clear_color": list(content.void_clear_color),
            "depth_enabled": content.depth_enabled,
            "depth_scale_meters": content.depth_scale_meters,
            "depth_mesh_resolution": content.depth_mesh_resolution,
            "depth_model": content.depth_model,
            "depth_backend": content.depth_backend,
        }
    )
    config_path.parent.mkdir(parents=True, exist_ok=True)
    # ensure_ascii=False keeps non-ASCII media paths (e.g. Vietnamese folder
    # names) readable instead of escaping them to \uXXXX.
    config_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
