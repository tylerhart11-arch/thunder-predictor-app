from __future__ import annotations

import pandas as pd
import plotly.express as px
import streamlit as st

from dashboard.helpers import (
    REPORTS,
    apply_casino_theme,
    load_archive,
    load_model_maintenance_artifacts,
    load_optional_csv,
    load_upcoming,
    read_json,
    style_plotly,
)


def _first_present(df: pd.DataFrame, candidates: list[str]) -> str | None:
    for candidate in candidates:
        if candidate in df.columns:
            return candidate
    return None


def _summary_pick(summary: dict, *keys: str):
    for key in keys:
        value = summary.get(key)
        if value is not None:
            return value
    return None


def _format_pct(value) -> str:
    return f"{float(value):.1%}" if isinstance(value, (int, float)) else "N/A"


def _format_num(value) -> str:
    return f"{float(value):.3f}" if isinstance(value, (int, float)) else "N/A"


def _format_int(value) -> str:
    try:
        return str(int(value))
    except (TypeError, ValueError):
        return "N/A"


def _completed_thunder_from_archive(archive: pd.DataFrame) -> pd.DataFrame:
    thunder = archive[
        (archive["HOME_TEAM_ABBREVIATION"] == "OKC") | (archive["AWAY_TEAM_ABBREVIATION"] == "OKC")
    ].copy()
    if thunder.empty:
        return thunder

    thunder["GAME_DATE"] = pd.to_datetime(thunder["GAME_DATE"], errors="coerce")
    thunder = thunder[thunder["ACTUAL_HOME_WIN"].notna()].copy()
    thunder["PRED_HOME_WIN"] = (thunder["PRED_HOME_WIN_PROB"] >= 0.5).astype(int)
    thunder["IS_CORRECT"] = (thunder["PRED_HOME_WIN"] == thunder["ACTUAL_HOME_WIN"]).astype(int)
    thunder = thunder.sort_values("GAME_DATE").copy()
    thunder["ROLLING_ACCURACY"] = thunder["IS_CORRECT"].rolling(10, min_periods=1).mean()
    thunder["CUM_ACCURACY"] = thunder["IS_CORRECT"].expanding().mean()
    return thunder


apply_casino_theme(
    page_title="Thunder Tracker (OKC)",
    subtitle="Track every pick, result, confidence swing, and maintenance signal over time.",
)

thunder_summary = read_json(REPORTS / "thunder_summary.json")
maintenance = load_model_maintenance_artifacts()
archive = load_archive()
upcoming = load_upcoming()
if archive.empty:
    st.warning("Prediction archive not found yet. Run the pipeline first.")
    st.stop()

completed_report = load_optional_csv(REPORTS / "thunder_predictions_completed.csv", parse_dates=["GAME_DATE"])
weekly = load_optional_csv(REPORTS / "thunder_weekly_summary.csv", parse_dates=["WEEK"])

thunder = archive[
    (archive["HOME_TEAM_ABBREVIATION"] == "OKC") | (archive["AWAY_TEAM_ABBREVIATION"] == "OKC")
].copy()
thunder["GAME_DATE"] = pd.to_datetime(thunder["GAME_DATE"], errors="coerce")
if {"HOME_TEAM_ABBREVIATION", "AWAY_TEAM_ABBREVIATION"}.issubset(upcoming.columns):
    upcoming_thunder = upcoming[
        (upcoming["HOME_TEAM_ABBREVIATION"] == "OKC") | (upcoming["AWAY_TEAM_ABBREVIATION"] == "OKC")
    ].copy()
    if not upcoming_thunder.empty and "GAME_DATE" in upcoming_thunder.columns:
        upcoming_thunder["GAME_DATE"] = pd.to_datetime(upcoming_thunder["GAME_DATE"], errors="coerce")
        upcoming_thunder = upcoming_thunder.sort_values("GAME_DATE")
else:
    upcoming_thunder = pd.DataFrame()

completed = completed_report.copy()
if completed.empty:
    completed = _completed_thunder_from_archive(archive)
else:
    completed["GAME_DATE"] = pd.to_datetime(completed["GAME_DATE"], errors="coerce")
    if "PRED_HOME_WIN" not in completed.columns:
        completed["PRED_HOME_WIN"] = (completed["PRED_HOME_WIN_PROB"] >= 0.5).astype(int)
    if "IS_CORRECT" not in completed.columns:
        completed["IS_CORRECT"] = (completed["PRED_HOME_WIN"] == completed["ACTUAL_HOME_WIN"]).astype(int)
    completed = completed.sort_values("GAME_DATE").copy()
    if "ROLLING_ACCURACY" not in completed.columns:
        completed["ROLLING_ACCURACY"] = completed["IS_CORRECT"].rolling(10, min_periods=1).mean()
    if "CUM_ACCURACY" not in completed.columns:
        completed["CUM_ACCURACY"] = completed["IS_CORRECT"].expanding().mean()

completed_games = int(len(completed))
correct_games = int(completed["IS_CORRECT"].sum()) if not completed.empty else 0
incorrect_games = completed_games - correct_games
accuracy = thunder_summary.get("accuracy")
if accuracy is None and completed_games:
    accuracy = correct_games / completed_games

maintenance_summary = maintenance["summary"]
maintenance_windows = maintenance["windows"]
maintenance_segments = maintenance["segments"]
maintenance_buckets = maintenance["confidence_buckets"]
if "scope" in maintenance_windows.columns:
    maintenance_windows = maintenance_windows[maintenance_windows["scope"] == "thunder"].copy()
if "scope" in maintenance_segments.columns:
    maintenance_segments = maintenance_segments[maintenance_segments["scope"] == "thunder"].copy()
if "scope" in maintenance_buckets.columns:
    maintenance_buckets = maintenance_buckets[maintenance_buckets["scope"] == "thunder"].copy()

c1, c2, c3, c4 = st.columns(4)
c1.metric("Tracked Thunder Games", int(thunder_summary.get("games", completed_games)))
c2.metric("Accuracy", _format_pct(accuracy))
c3.metric("Correct", int(thunder_summary.get("correct", correct_games)))
c4.metric("Incorrect", int(thunder_summary.get("incorrect", incorrect_games)))

if any(
    thunder_summary.get(key) is not None
    for key in ["last_10_accuracy", "current_streak", "avg_confidence"]
):
    e1, e2, e3 = st.columns(3)
    e1.metric("Last 10 Accuracy", _format_pct(thunder_summary.get("last_10_accuracy")))
    streak_label = thunder_summary.get("current_streak_label")
    streak_value = thunder_summary.get("current_streak")
    if streak_label and streak_value is not None:
        e2.metric("Current Streak", f"{streak_label}{abs(int(streak_value))}")
    else:
        e2.metric("Current Streak", "N/A")
    e3.metric("Avg Confidence", _format_pct(thunder_summary.get("avg_confidence")))

if completed_games and completed_games < 20:
    st.caption("Thunder sample size is still small, so read the accuracy signal with caution.")

st.subheader("Upcoming Thunder Predictions")
if upcoming_thunder.empty:
    st.info("No upcoming Thunder games are currently queued.")
else:
    upcoming_cols = [
        col
        for col in [
            "GAME_DATE",
            "HOME_TEAM_ABBREVIATION",
            "AWAY_TEAM_ABBREVIATION",
            "PRED_HOME_WIN_PROB",
            "PRED_HOME_WIN",
        ]
        if col in upcoming_thunder.columns
    ]
    st.dataframe(upcoming_thunder[upcoming_cols], use_container_width=True, hide_index=True)

st.subheader("Completed Thunder Predictions vs Actual")
if completed.empty:
    st.info("Completed Thunder predictions will appear here as results accumulate.")
else:
    display_cols = [
        col
        for col in [
            "GAME_DATE",
            "HOME_TEAM_ABBREVIATION",
            "AWAY_TEAM_ABBREVIATION",
            "PRED_HOME_WIN_PROB",
            "PRED_HOME_WIN",
            "ACTUAL_HOME_WIN",
            "IS_CORRECT",
        ]
        if col in completed.columns
    ]
    st.dataframe(
        completed[display_cols].sort_values("GAME_DATE", ascending=False),
        use_container_width=True,
        hide_index=True,
    )

    trend = completed[["GAME_DATE", "ROLLING_ACCURACY", "CUM_ACCURACY"]].copy()
    fig_roll = px.line(
        trend,
        x="GAME_DATE",
        y=["ROLLING_ACCURACY", "CUM_ACCURACY"],
        title="Thunder Accuracy Trend",
        labels={"value": "Accuracy", "variable": "Series"},
    )
    st.plotly_chart(style_plotly(fig_roll), use_container_width=True)

    calibration = completed[["PRED_HOME_WIN_PROB", "ACTUAL_HOME_WIN", "IS_CORRECT"]].copy()
    fig_prob = px.scatter(
        calibration,
        x="PRED_HOME_WIN_PROB",
        y="ACTUAL_HOME_WIN",
        color="IS_CORRECT",
        title="Thunder Calibration Check",
        labels={"ACTUAL_HOME_WIN": "Actual Home Win (0/1)", "PRED_HOME_WIN_PROB": "Predicted Home Win Probability"},
    )
    st.plotly_chart(style_plotly(fig_prob), use_container_width=True)

st.subheader("Model Maintenance")
if not maintenance_summary and maintenance_windows.empty and maintenance_segments.empty and maintenance_buckets.empty:
    st.info("Maintenance artifacts are not available yet. Once the backend writes them, this panel will fill in.")
else:
    status = _summary_pick(maintenance_summary, "status", "signal", "maintenance_status", "alert_status") or "Available"
    thunder_summary_block = maintenance_summary.get("thunder", {}) if isinstance(maintenance_summary, dict) else {}
    thunder_recent = thunder_summary_block.get("recent", {}) if isinstance(thunder_summary_block, dict) else {}
    recent_games = thunder_recent.get("games")
    recent_accuracy = thunder_recent.get("accuracy")
    recent_brier = thunder_recent.get("brier_score")
    recent_log_loss = thunder_recent.get("log_loss")

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Status", str(status))
    m2.metric("Recent Games", _format_int(recent_games))
    m3.metric("Recent Accuracy", _format_pct(recent_accuracy))
    m4.metric("Recent Brier", _format_num(recent_brier))

    if recent_log_loss is not None:
        st.caption(f"Recent log loss: {_format_num(recent_log_loss)}")

    for alert in maintenance_summary.get("alerts", []):
        st.warning(str(alert))
    for warning in maintenance_summary.get("warnings", []):
        st.caption(str(warning))

    if not maintenance_windows.empty:
        st.markdown("**Maintenance Windows**")
        st.dataframe(maintenance_windows, use_container_width=True, hide_index=True)
        window_label = _first_present(
            maintenance_windows,
            ["WINDOW", "window", "WINDOW_LABEL", "window_label", "GAME_DATE", "window_end", "WINDOW_END"],
        )
        accuracy_col = _first_present(
            maintenance_windows,
            ["accuracy", "ACCURACY", "recent_accuracy", "win_rate", "hit_rate"],
        )
        brier_col = _first_present(
            maintenance_windows,
            ["brier_score", "BRIER_SCORE", "recent_brier", "brier"],
        )
        if window_label and accuracy_col:
            chart_df = maintenance_windows[[window_label, accuracy_col] + ([brier_col] if brier_col else [])].copy()
            chart_df = chart_df.rename(columns={window_label: "Window", accuracy_col: "Accuracy"})
            if brier_col:
                chart_df = chart_df.rename(columns={brier_col: "Brier"})
                melted = chart_df.melt(id_vars="Window", value_vars=["Accuracy", "Brier"], var_name="Metric", value_name="Value")
                fig_windows = px.line(melted, x="Window", y="Value", color="Metric", markers=True, title="Maintenance Window Trends")
            else:
                fig_windows = px.line(chart_df, x="Window", y="Accuracy", markers=True, title="Maintenance Window Accuracy")
            st.plotly_chart(style_plotly(fig_windows), use_container_width=True)

    if not maintenance_segments.empty:
        st.markdown("**Segment Performance**")
        st.dataframe(maintenance_segments, use_container_width=True, hide_index=True)
        segment_label = _first_present(
            maintenance_segments,
            ["segment", "SEGMENT", "bucket", "BUCKET", "group", "GROUP"],
        )
        segment_accuracy = _first_present(
            maintenance_segments,
            ["accuracy", "ACCURACY", "hit_rate", "win_rate"],
        )
        if segment_label and segment_accuracy:
            seg_df = maintenance_segments[[segment_label, segment_accuracy]].copy()
            seg_df = seg_df.rename(columns={segment_label: "Segment", segment_accuracy: "Accuracy"})
            fig_segments = px.bar(
                seg_df.sort_values("Accuracy"),
                x="Segment",
                y="Accuracy",
                title="Segment Accuracy",
            )
            st.plotly_chart(style_plotly(fig_segments), use_container_width=True)

    if not maintenance_buckets.empty:
        st.markdown("**Confidence Buckets**")
        bucket_cols = [
            col
            for col in [
                _first_present(
                    maintenance_buckets,
                    ["bucket", "BUCKET", "confidence_bucket", "prob_bucket", "prediction_bucket"],
                ),
                _first_present(
                    maintenance_buckets,
                    ["mean_pred_prob", "avg_pred_prob", "avg_confidence", "predicted_prob", "pred_prob"],
                ),
                _first_present(
                    maintenance_buckets,
                    ["observed_win_rate", "observed_accuracy", "actual_win_rate", "win_rate", "observed_rate"],
                ),
            ]
            if col
        ]
        st.dataframe(maintenance_buckets, use_container_width=True, hide_index=True)
        bucket_label = bucket_cols[0] if bucket_cols else None
        pred_col = bucket_cols[1] if len(bucket_cols) > 1 else None
        obs_col = bucket_cols[2] if len(bucket_cols) > 2 else None
        if bucket_label and pred_col and obs_col:
            bucket_plot = maintenance_buckets[[bucket_label, pred_col, obs_col]].copy()
            bucket_plot = bucket_plot.rename(
                columns={bucket_label: "Bucket", pred_col: "Predicted", obs_col: "Observed"}
            )
            bucket_plot = bucket_plot.melt(id_vars="Bucket", var_name="Metric", value_name="Value")
            fig_buckets = px.bar(
                bucket_plot,
                x="Bucket",
                y="Value",
                color="Metric",
                barmode="group",
                title="Confidence Bucket Calibration",
            )
            st.plotly_chart(style_plotly(fig_buckets), use_container_width=True)

if not weekly.empty:
    weekly["WEEK"] = weekly["WEEK"].astype(str)
    st.subheader("Weekly Thunder Performance")
    st.dataframe(weekly, use_container_width=True, hide_index=True)
