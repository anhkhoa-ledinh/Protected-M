"""Load and save application configuration as JSON or YAML."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from src.config.schema import AppConfig

YAML_EXTENSIONS = {".yaml", ".yml"}


def _read_mapping(path: Path) -> dict[str, Any]:
    """
    Đọc file cấu hình thành dict **thô**, chưa qua dataclass nào.

    Tách riêng vì ``merge_user_config`` cần đúng những khoá người dùng thật sự
    ghi, không phải một ``AppConfig`` đã điền đầy mặc định — xem ghi chú ở đó.
    """
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    suffix = path.suffix.lower()
    text = path.read_text(encoding="utf-8")

    if suffix in YAML_EXTENSIONS:
        return _load_yaml(text)
    if suffix == ".json":
        return json.loads(text)
    raise ValueError(f"Unsupported config format: {suffix}")


def load_config(path: Path) -> AppConfig:
    """Load configuration from a JSON or YAML file."""
    return AppConfig.from_dict(_read_mapping(path))


def save_config(config: AppConfig, path: Path) -> None:
    """Persist configuration to JSON or YAML."""
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = config.to_dict()
    suffix = path.suffix.lower()

    if suffix in YAML_EXTENSIONS:
        path.write_text(_dump_yaml(payload), encoding="utf-8")
    elif suffix == ".json":
        path.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
    else:
        raise ValueError(f"Unsupported config format: {suffix}")


def merge_user_config(
    base_path: Path,
    user_path: Path | None = None,
) -> AppConfig:
    """Load defaults and overlay user overrides when present."""
    config = load_config(base_path)
    if user_path is None:
        user_path = base_path.parent / "user.json"

    if user_path.exists():
        # Dict THÔ, không qua AppConfig. Dạng cũ là
        #     user_data = load_config(user_path).to_dict()
        # — nạp user.json thành một AppConfig ĐẦY ĐỦ, nên to_dict() điền mọi
        # trường người dùng không ghi bằng **mặc định của dataclass**, rồi
        # _deep_merge phủ nguyên đống mặc định đó lên default.json. Kết quả:
        # hễ user.json tồn tại thì mọi giá trị trong default.json khác mặc định
        # đều bị xoá. Lỗi ngủ yên vì tới 2026-10-12 chỉ có đúng một khoá lệch:
        # "fullscreen": true về lại False, và app không bao giờ chạy toàn màn
        # hình dù đã bật cờ.
        user_data = _read_mapping(user_path)
        merged = _deep_merge(config.to_dict(), user_data)
        return AppConfig.from_dict(merged)

    return config


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge override into base."""
    result = dict(base)
    for key, value in override.items():
        if (
            key in result
            and isinstance(result[key], dict)
            and isinstance(value, dict)
        ):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def _load_yaml(text: str) -> dict[str, Any]:
    """Parse YAML text, requiring PyYAML when available."""
    try:
        import yaml
    except ImportError as exc:
        raise ImportError(
            "PyYAML is required for YAML config files. "
            "Install with: pip install pyyaml"
        ) from exc

    loaded = yaml.safe_load(text)
    if not isinstance(loaded, dict):
        raise ValueError("YAML config root must be a mapping")
    return loaded


def _dump_yaml(payload: dict[str, Any]) -> str:
    """Serialize payload to YAML text."""
    try:
        import yaml
    except ImportError as exc:
        raise ImportError(
            "PyYAML is required for YAML config files. "
            "Install with: pip install pyyaml"
        ) from exc

    return yaml.safe_dump(payload, sort_keys=False, allow_unicode=True)
