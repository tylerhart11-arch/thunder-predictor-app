from __future__ import annotations

"""Command-line entrypoint for the Thunder predictor pipeline and dashboard."""

import argparse
import subprocess
import sys

from app.config import build_paths, ensure_directories, load_config
from app.logger import get_logger
from pipelines.update_pipeline import NBAPipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="NBA Thunder Predictor CLI")
    parser.add_argument(
        "command",
        choices=["full-build", "daily-update", "run-dashboard"],
        help="Command to execute",
    )
    parser.add_argument(
        "--config",
        default="config/config.yaml",
        help="Path to config yaml",
    )
    parser.add_argument(
        "--dashboard-port",
        type=int,
        default=8501,
        help="Port for Streamlit dashboard",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    paths = build_paths(cfg)
    ensure_directories(paths)
    logger = get_logger("nba_pipeline", log_file=paths.logs_dir / "pipeline.log")

    pipeline = NBAPipeline(cfg=cfg, paths=paths, logger=logger)

    if args.command == "full-build":
        pipeline.run_full_build()
    elif args.command == "daily-update":
        pipeline.run_daily_update()
    else:
        cmd = [sys.executable, "-m", "streamlit", "run", "streamlit_app.py", "--server.port", str(args.dashboard_port)]
        logger.info("Launching dashboard: %s", " ".join(cmd))
        subprocess.run(cmd, check=True)  # noqa: S603


if __name__ == "__main__":
    main()
