from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

from dashboard.helpers import (
    apply_casino_theme,
    confidence_band,
    format_confidence,
    format_probability,
    latest_update_timestamp,
    load_upcoming,
    matchup_label,
    pick_label,
    render_section_grid,
    render_update_pill,
)


apply_casino_theme(
    page_title="Upcoming Predictions",
    subtitle="Fresh board lines from the model for the next slate of NBA games.",
)
update_ts = latest_update_timestamp()
if update_ts:
    render_update_pill(f"Last refresh: {update_ts}")

upcoming = load_upcoming()
if upcoming.empty:
    st.warning("No upcoming predictions available right now.")
    st.stop()

upcoming = upcoming.copy()
upcoming["MATCHUP"] = upcoming.apply(
    lambda row: matchup_label(row.get("HOME_TEAM_ABBREVIATION"), row.get("AWAY_TEAM_ABBREVIATION")),
    axis=1,
)
upcoming["PICK"] = upcoming.apply(
    lambda row: pick_label(
        row.get("PRED_HOME_WIN_PROB"),
        row.get("HOME_TEAM_ABBREVIATION"),
        row.get("AWAY_TEAM_ABBREVIATION"),
    ),
    axis=1,
)
upcoming["CONFIDENCE"] = upcoming["PRED_HOME_WIN_PROB"].map(
    lambda value: abs(float(value) - 0.5) * 2 if pd.notna(value) else np.nan
)
upcoming["CONFIDENCE_BAND"] = upcoming["PRED_HOME_WIN_PROB"].map(confidence_band)
upcoming["GAME_DATE"] = upcoming["GAME_DATE"].dt.date
upcoming = upcoming.sort_values(["GAME_DATE", "CONFIDENCE"], ascending=[True, False])

render_section_grid(
    [
        (
            "Board Size",
            f"{len(upcoming)} games are on the slate. That keeps the page focused on the actual decision set.",
        ),
        (
            "Best Edge",
            f"Top confidence call: {upcoming.iloc[0]['PICK']} at {format_probability(upcoming.iloc[0]['PRED_HOME_WIN_PROB'])}.",
        ),
        (
            "How to Read",
            "Confidence band is the model's distance from 50/50. Lean and strong edge calls deserve more attention than toss-ups.",
        ),
    ]
)

display_upcoming = upcoming[
    [
        "GAME_DATE",
        "MATCHUP",
        "PICK",
        "PRED_HOME_WIN_PROB",
        "CONFIDENCE",
        "CONFIDENCE_BAND",
    ]
].copy()
display_upcoming["PRED_HOME_WIN_PROB"] = display_upcoming["PRED_HOME_WIN_PROB"].map(format_probability)
display_upcoming["CONFIDENCE"] = display_upcoming["CONFIDENCE"].map(format_confidence)

st.dataframe(display_upcoming, use_container_width=True, hide_index=True)
