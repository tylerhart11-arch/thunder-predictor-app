from __future__ import annotations

import streamlit as st

from dashboard.helpers import (
    REPORTS,
    apply_casino_theme,
    latest_update_timestamp,
    load_archive,
    load_clean_games,
    load_model_maintenance_artifacts,
    load_upcoming,
    render_update_pill,
    read_json,
)


def _missing_columns(df, required_columns: list[str]) -> list[str]:
    return sorted(set(required_columns) - set(df.columns))


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
    upcoming_required = [
        "GAME_DATE",
        "HOME_TEAM_ABBREVIATION",
        "AWAY_TEAM_ABBREVIATION",
        "PRED_HOME_WIN_PROB",
    ]
    upcoming_missing = _missing_columns(upcoming, upcoming_required)
    if upcoming_missing:
        st.info(
            "Upcoming board is refreshing. Preview columns are not ready yet: "
            + ", ".join(upcoming_missing)
        )
    elif upcoming.empty:
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
    archive_required = [
        "GAME_DATE",
        "HOME_TEAM_ABBREVIATION",
        "AWAY_TEAM_ABBREVIATION",
        "PRED_HOME_WIN_PROB",
        "ACTUAL_HOME_WIN",
    ]
    archive_missing = _missing_columns(archive, archive_required)
    if archive_missing:
        st.info(
            "Thunder results preview is refreshing. Archive columns are not ready yet: "
            + ", ".join(archive_missing)
        )
    else:
        thunder_games = archive[
            (archive["HOME_TEAM_ABBREVIATION"] == "OKC") | (archive["AWAY_TEAM_ABBREVIATION"] == "OKC")
        ].copy()
        thunder_completed = thunder_games[
            thunder_games["ACTUAL_HOME_WIN"].notna()
        ].sort_values("GAME_DATE", ascending=False)
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

maintenance_artifacts = load_model_maintenance_artifacts()
maintenance_summary = maintenance_artifacts["summary"]
maintenance_windows = maintenance_artifacts["windows"]
maintenance_segments = maintenance_artifacts["segments"]
maintenance_buckets = maintenance_artifacts["confidence_buckets"]

if maintenance_summary or not maintenance_windows.empty or not maintenance_segments.empty or not maintenance_buckets.empty:
    st.subheader("Model Maintenance")

    def _pick(*paths: str):
        for path in paths:
            value = maintenance_summary
            for part in path.split("."):
                if not isinstance(value, dict):
                    value = None
                    break
                value = value.get(part)
            if value is not None:
                return value
        return None

    def _format_pct(value):
        return f"{float(value):.1%}" if isinstance(value, (int, float)) else "N/A"

    def _format_num(value):
        return f"{float(value):.3f}" if isinstance(value, (int, float)) else "N/A"

    def _format_int(value):
        try:
            return str(int(value))
        except (TypeError, ValueError):
            return "N/A"

    status = _pick("status", "signal", "maintenance_status", "alert_status") or "Available"
    recent_games = _pick("league.recent.games", "recent_games", "games", "window_games", "n_recent_games")
    recent_accuracy = _pick("league.recent.accuracy", "recent_accuracy", "rolling_accuracy", "accuracy_recent")
    recent_brier = _pick("league.recent.brier_score", "recent_brier_score", "recent_brier", "brier_score_recent")
    recent_log_loss = _pick("league.recent.log_loss", "recent_log_loss", "log_loss_recent", "recent_logloss")

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Status", str(status))
    m2.metric("Recent Games", _format_int(recent_games))
    m3.metric("Recent Accuracy", _format_pct(recent_accuracy))
    m4.metric("Recent Brier", _format_num(recent_brier))

    if recent_log_loss is not None:
        st.caption(f"Recent log loss: {_format_num(recent_log_loss)}")

    alerts = maintenance_summary.get("alerts", [])
    warnings = maintenance_summary.get("warnings", [])
    for alert in alerts:
        st.warning(str(alert))
    for warning in warnings:
        st.caption(str(warning))
else:
    st.subheader("Model Maintenance")
    st.caption("Maintenance artifacts will appear here once the backend starts writing them.")
