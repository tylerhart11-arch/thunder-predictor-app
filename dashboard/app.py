from __future__ import annotations

import streamlit as st

from dashboard.helpers import (
    REPORTS,
    apply_casino_theme,
    latest_update_timestamp,
    load_archive,
    load_clean_games,
    load_upcoming,
    render_section_grid,
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

render_section_grid(
    [
        ("League Overview", "Standings, game flow, and league-wide board context."),
        ("Model Performance", "Accuracy, calibration, and split-by-split scorecard."),
        ("Thunder Tracker", "Every OKC call, result, confidence swing, and hit rate."),
        ("Upcoming Predictions", "The next slate with model win probabilities and predicted winners."),
        ("Diagnostics", "Leakage checks, quality reports, and reliability detail."),
        ("Arena Notes", "This dashboard is styled like a loud OKC sportsbook board on purpose."),
    ]
)

if metrics:
    st.subheader("Latest Model Snapshot")
    cal = metrics.get("improved_test_calibrated", {})
    cols = st.columns(4)
    cols[0].metric("Accuracy", f"{cal.get('accuracy', 0):.3f}")
    cols[1].metric("Log Loss", f"{cal.get('log_loss', 0):.3f}")
    cols[2].metric("ROC AUC", f"{cal.get('roc_auc', 0):.3f}")
    cols[3].metric("Brier", f"{cal.get('brier_score', 0):.3f}")
