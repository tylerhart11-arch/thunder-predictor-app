from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv


@dataclass
class Paths:
    root: Path
    sqlite_path: Path
    raw_dir: Path
    cleaned_dir: Path
    features_dir: Path
    predictions_dir: Path
    artifacts_dir: Path
    logs_dir: Path
    reports_dir: Path
    models_dir: Path


def _read_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _resolve(root: Path, rel: str) -> Path:
    return (root / rel).resolve()


def load_config(config_path: str = "config/config.yaml") -> dict[str, Any]:
    load_dotenv(override=False)
    root = Path.cwd().resolve()
    cfg = _read_yaml(root / config_path)
    cfg["env"] = {"APP_TIMEZONE": os.getenv("APP_TIMEZONE", cfg["project"]["timezone"])}
    return cfg


def build_paths(cfg: dict[str, Any]) -> Paths:
    root = Path.cwd().resolve()
    storage = cfg["data"]["storage"]
    return Paths(
        root=root,
        sqlite_path=_resolve(root, storage["sqlite_path"]),
        raw_dir=_resolve(root, storage["raw_dir"]),
        cleaned_dir=_resolve(root, storage["cleaned_dir"]),
        features_dir=_resolve(root, storage["features_dir"]),
        predictions_dir=_resolve(root, storage["predictions_dir"]),
        artifacts_dir=_resolve(root, storage["artifacts_dir"]),
        logs_dir=(root / "logs").resolve(),
        reports_dir=(root / "reports").resolve(),
        models_dir=(root / "models").resolve(),
    )


def ensure_directories(paths: Paths) -> None:
    for p in [
        paths.raw_dir,
        paths.cleaned_dir,
        paths.features_dir,
        paths.predictions_dir,
        paths.artifacts_dir,
        paths.logs_dir,
        paths.reports_dir,
        paths.models_dir,
    ]:
        p.mkdir(parents=True, exist_ok=True)

