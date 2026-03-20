from __future__ import annotations

import runpy
from pathlib import Path


runpy.run_path(
    Path(__file__).resolve().parents[2] / "pages" / "3_Thunder_Tracker.py",
    run_name="__main__",
)
