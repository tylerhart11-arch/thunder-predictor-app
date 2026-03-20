from __future__ import annotations

import numpy as np
import streamlit as st

from dashboard.helpers import apply_casino_theme, load_upcoming


apply_casino_theme(
    page_title="Upcoming Predictions",
    subtitle="Fresh board lines from the model for the next slate of NBA games.",
)
upcoming = load_upcoming()
if upcoming.empty:
    st.warning("No upcoming predictions available right now.")
    st.stop()

upcoming = upcoming.copy()
upcoming["PREDICTED_WINNER"] = np.where(
    upcoming["PRED_HOME_WIN"] == 1,
    upcoming["HOME_TEAM_ABBREVIATION"],
    upcoming["AWAY_TEAM_ABBREVIATION"],
)
upcoming["GAME_DATE"] = upcoming["GAME_DATE"].dt.date
upcoming = upcoming.sort_values("GAME_DATE")

st.dataframe(
    upcoming[
        [
            "GAME_DATE",
            "HOME_TEAM_ABBREVIATION",
            "AWAY_TEAM_ABBREVIATION",
            "PRED_HOME_WIN_PROB",
            "PREDICTED_WINNER",
        ]
    ],
    use_container_width=True,
)
