from __future__ import annotations

import streamlit as st

from dashboard.helpers import (
    REPORTS,
    apply_casino_theme,
    latest_update_timestamp,
    load_archive,
    load_clean_games,
    load_upcoming,
    render_update_pill,
    read_json,
)


st.set_page_config(page_title="NBA Thunder Predictor", page_icon=":basketball:", layout="wide")
apply_casino_theme(
    page_title="NBA Game Outcome Predictor",
    subtitle="League-wide modeling with Thunder-focused sportsbook-style tracking.",
)
updated_at = latest_update_timestamp()
if updated_at:
    render_update_pill(f"Last data refresh: {updated_at}")

metrics = read_json(REPORTS / "metrics_latest.json")
summary = read_json(REPORTS / "thunder_summary.json")
games = load_clean_games()
archive = load_archive()
upcoming = load_upcoming()

c1, c2, c3, c4 = st.columns(4)
c1.metric("Historical Games", f"{len(games):,}" if not games.empty else "0")
c2.metric("Predictions Archived", f"{len(archive):,}" if not archive.empty else "0")
c3.metric("Upcoming Predicted", f"{len(upcoming):,}" if not upcoming.empty else "0")

thunder_acc = summary.get("accuracy")
if thunder_acc is None:
    c4.metric("Thunder Accuracy", "N/A")
else:
    c4.metric("Thunder Accuracy", f"{thunder_acc:.1%}")

st.subheader("Control Room")
nav1, nav2, nav3 = st.columns(3)
with nav1:
    st.markdown("**League Overview**")
    st.caption("Standings, recent scores, and league-wide game flow.")
with nav2:
    st.markdown("**Thunder Tracker**")
    st.caption("OKC-specific picks, results, confidence, and rolling accuracy.")
with nav3:
    st.markdown("**Diagnostics**")
    st.caption("Model calibration, leakage checks, and holdout split detail.")

preview_left, preview_right = st.columns(2)

with preview_left:
    st.subheader("Upcoming Board")
    if upcoming.empty:
        st.info("No upcoming games are currently queued in the prediction board.")
    else:
        upcoming_view = upcoming.copy().sort_values("GAME_DATE").head(8)
        st.dataframe(
            upcoming_view[
                [
                    "GAME_DATE",
                    "HOME_TEAM_ABBREVIATION",
                    "AWAY_TEAM_ABBREVIATION",
                    "PRED_HOME_WIN_PROB",
                ]
            ],
            use_container_width=True,
            hide_index=True,
        )

with preview_right:
    st.subheader("Thunder Recent Results")
    thunder_games = archive[
        (archive["HOME_TEAM_ABBREVIATION"] == "OKC") | (archive["AWAY_TEAM_ABBREVIATION"] == "OKC")
    ].copy()
    thunder_completed = thunder_games[thunder_games["ACTUAL_HOME_WIN"].notna()].sort_values("GAME_DATE", ascending=False)
    if thunder_completed.empty:
        st.info("Thunder results will appear here as completed predictions accumulate.")
    else:
        st.dataframe(
            thunder_completed[
                [
                    "GAME_DATE",
                    "HOME_TEAM_ABBREVIATION",
                    "AWAY_TEAM_ABBREVIATION",
                    "PRED_HOME_WIN_PROB",
                    "ACTUAL_HOME_WIN",
                ]
            ].head(8),
            use_container_width=True,
            hide_index=True,
        )

if metrics:
    st.subheader("Latest Model Snapshot")
    cal = metrics.get("improved_test_calibrated", {})
    cols = st.columns(4)
    cols[0].metric("Accuracy", f"{cal.get('accuracy', 0):.3f}")
    cols[1].metric("Log Loss", f"{cal.get('log_loss', 0):.3f}")
    cols[2].metric("ROC AUC", f"{cal.get('roc_auc', 0):.3f}")
    cols[3].metric("Brier", f"{cal.get('brier_score', 0):.3f}")
