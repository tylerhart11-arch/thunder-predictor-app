from __future__ import annotations

import runpy
from pathlib import Path


runpy.run_path(
    Path(__file__).resolve().parents[2] / "pages" / "5_Diagnostics.py",
    run_name="__main__",
)
