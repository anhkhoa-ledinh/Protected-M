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
    """Write the newly-opened media's path/type back to content.json.

    TASK_perf_textures_1119 Fix 6: this used to also write every OTHER
    ``content`` field — ``plane_position_m``, ``plane_width_m``, etc.
    By the time a sandbox-mode media open calls this, those fields on
    ``content`` already carry the RUNTIME D2 physical-window override
    (``plane_position_m`` z -0.75 -> 0.0, ``plane_width_m`` 1.28 -> 0.344
    — not something the user set), so writing them clobbered the user's
    real calibration with that override. Only ``content_path``/
    ``content_type``/``video_path`` ever need to persist here — every
    other key in the file is left exactly as it was.
    """
    stored = media_path_for_config(media_path, project_root)
    payload: dict = {}
    if config_path.exists():
        payload = json.loads(config_path.read_text(encoding="utf-8"))

    payload["content_path"] = stored
    payload["content_type"] = content_type
    if content_type == "video":
        payload["video_path"] = stored

    config_path.parent.mkdir(parents=True, exist_ok=True)
    # ensure_ascii=False keeps non-ASCII media paths (e.g. Vietnamese folder
    # names) readable instead of escaping them to \uXXXX.
    config_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
