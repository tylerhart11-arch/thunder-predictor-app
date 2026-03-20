"""Streamlit Community Cloud entrypoint.

This thin wrapper lets the app deploy from the repository root while keeping
the actual dashboard code under `dashboard/`.
"""

from dashboard.app import *  # noqa: F401,F403
