"""Launch the Streamlit dashboard for the Thunder predictor."""

import subprocess
import sys
from pathlib import Path


def run() -> None:
    project_root = Path(__file__).resolve().parents[1]
    cmd = [sys.executable, "-m", "streamlit", "run", "streamlit_app.py"]
    subprocess.run(cmd, check=True, cwd=str(project_root))  # noqa: S603


if __name__ == "__main__":
    run()
