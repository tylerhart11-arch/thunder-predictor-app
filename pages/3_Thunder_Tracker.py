from __future__ import annotations

import plotly.express as px
import streamlit as st

from dashboard.helpers import REPORTS, apply_casino_theme, load_archive, read_csv, read_json, style_plotly


apply_casino_theme(
    page_title="Thunder Tracker (OKC)",
    subtitle="Track every pick, result, confidence swing, and rolling hit rate over time.",
)
summary = read_json(REPORTS / "thunder_summary.json")
archive = load_archive()
if archive.empty:
    st.warning("Prediction archive not found yet. Run the pipeline first.")
    st.stop()

thunder = archive[
    (archive["HOME_TEAM_ABBREVIATION"] == "OKC") | (archive["AWAY_TEAM_ABBREVIATION"] == "OKC")
].copy()
thunder["GAME_DATE"] = thunder["GAME_DATE"].dt.date
completed = thunder[thunder["ACTUAL_HOME_WIN"].notna()].copy()

c1, c2, c3, c4 = st.columns(4)
c1.metric("Tracked Thunder Games", int(summary.get("games", 0)))
c2.metric("Accuracy", f"{summary.get('accuracy', 0):.1%}" if summary.get("accuracy") is not None else "N/A")
c3.metric("Correct", int(summary.get("correct", 0)))
c4.metric("Incorrect", int(summary.get("incorrect", 0)))

st.subheader("Upcoming Thunder Predictions")
upcoming_thunder = thunder[thunder["ACTUAL_HOME_WIN"].isna()].sort_values("GAME_DATE")
st.dataframe(
    upcoming_thunder[
        [
            "GAME_DATE",
            "HOME_TEAM_ABBREVIATION",
            "AWAY_TEAM_ABBREVIATION",
            "PRED_HOME_WIN_PROB",
            "PRED_HOME_WIN",
        ]
    ],
    use_container_width=True,
)

st.subheader("Completed Thunder Predictions vs Actual")
if not completed.empty:
    completed["PRED_HOME_WIN"] = (completed["PRED_HOME_WIN_PROB"] >= 0.5).astype(int)
    completed["IS_CORRECT"] = (completed["PRED_HOME_WIN"] == completed["ACTUAL_HOME_WIN"]).astype(int)
    st.dataframe(
        completed[
            [
                "GAME_DATE",
                "HOME_TEAM_ABBREVIATION",
                "AWAY_TEAM_ABBREVIATION",
                "PRED_HOME_WIN_PROB",
                "PRED_HOME_WIN",
                "ACTUAL_HOME_WIN",
                "IS_CORRECT",
            ]
        ].sort_values("GAME_DATE", ascending=False),
        use_container_width=True,
    )

    completed_sorted = completed.sort_values("GAME_DATE").copy()
    completed_sorted["ROLLING_ACCURACY_10"] = completed_sorted["IS_CORRECT"].rolling(10, min_periods=1).mean()
    completed_sorted["CUM_ACCURACY"] = completed_sorted["IS_CORRECT"].expanding().mean()

    fig_roll = px.line(
        completed_sorted,
        x="GAME_DATE",
        y=["ROLLING_ACCURACY_10", "CUM_ACCURACY"],
        title="Thunder Prediction Accuracy Over Time",
    )
    st.plotly_chart(style_plotly(fig_roll), use_container_width=True)

    fig_prob = px.scatter(
        completed_sorted,
        x="PRED_HOME_WIN_PROB",
        y="ACTUAL_HOME_WIN",
        color="IS_CORRECT",
        title="Predicted Probability vs Actual Outcome (Thunder Games)",
        labels={"ACTUAL_HOME_WIN": "Actual Home Win (0/1)", "PRED_HOME_WIN_PROB": "Predicted Home Win Probability"},
    )
    st.plotly_chart(style_plotly(fig_prob), use_container_width=True)

weekly = read_csv(REPORTS / "thunder_weekly_summary.csv")
if not weekly.empty:
    weekly["WEEK"] = weekly["WEEK"].astype(str)
    st.subheader("Weekly Thunder Performance")
    st.dataframe(weekly, use_container_width=True)
