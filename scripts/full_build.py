from __future__ import annotations

"""Run a full historical rebuild for the Thunder predictor."""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.config import build_paths, ensure_directories, load_config
from app.logger import get_logger
from pipelines.update_pipeline import NBAPipeline


def run() -> None:
    cfg = load_config("config/config.yaml")
    paths = build_paths(cfg)
    ensure_directories(paths)
    logger = get_logger("nba_pipeline", log_file=paths.logs_dir / "pipeline.log")
    NBAPipeline(cfg=cfg, paths=paths, logger=logger).run_full_build()


if __name__ == "__main__":
    run()
