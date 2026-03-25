"""Streamlit Community Cloud entrypoint.

This wrapper must execute the dashboard page on every rerun. A plain Python
import only runs module side effects once per process, which can leave the
home page blank after a browser refresh in Streamlit Cloud.
"""

from __future__ import annotations

import runpy
from pathlib import Path


runpy.run_path(
    Path(__file__).resolve().parent / "dashboard" / "app.py",
    run_name="__main__",
)
