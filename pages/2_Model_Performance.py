from __future__ import annotations

import pandas as pd
import streamlit as st

from dashboard.helpers import REPORTS, apply_casino_theme, read_csv, read_json


apply_casino_theme(
    page_title="Model Performance",
    subtitle="Baseline vs boosted model scorecards with calibration and diagnostics.",
)
metrics = read_json(REPORTS / "metrics_latest.json")

if not metrics:
    st.warning("No model metrics found. Run full build or daily update.")
    st.stop()

split = metrics.get("split", {})
train_seasons = split.get("train_seasons", [])
valid_seasons = split.get("validation_seasons", [])
test_seasons = split.get("test_seasons", [])

st.subheader("Data Regime")
st.markdown(
    f"""
- `Training set (past seasons)`: {", ".join(train_seasons) if train_seasons else "N/A"}
- `Validation set`: {", ".join(valid_seasons) if valid_seasons else "N/A"}
- `Test set (current season so far)`: {", ".join(test_seasons) if test_seasons else "N/A"}
"""
)

split_summary = read_csv(REPORTS / "split_dataset_summary.csv")
if not split_summary.empty:
    st.dataframe(split_summary, use_container_width=True)

rows = []
for label in ["baseline_test", "improved_test_uncalibrated", "improved_test_calibrated"]:
    block = metrics.get(label, {})
    rows.append(
        {
            "model": label,
            "accuracy": block.get("accuracy"),
            "log_loss": block.get("log_loss"),
            "roc_auc": block.get("roc_auc"),
            "brier_score": block.get("brier_score"),
            "n_obs": block.get("n_obs"),
        }
    )
perf_df = pd.DataFrame(rows)
st.subheader("Test Metrics")
st.dataframe(perf_df, use_container_width=True)

st.subheader("Best Improved Model Params")
st.json(metrics.get("best_params", {}))

imp = read_csv(REPORTS / "improved_feature_importance.csv")
if not imp.empty:
    st.subheader("Improved Model Feature Importance")
    st.dataframe(imp.head(25), use_container_width=True)

base_imp = read_csv(REPORTS / "baseline_feature_importance.csv")
if not base_imp.empty:
    st.subheader("Baseline Logistic Coefficients")
    st.dataframe(base_imp.head(25), use_container_width=True)

st.subheader("Diagnostics Images")
c1, c2 = st.columns(2)
rel_path = REPORTS / "diagnostics" / "reliability_curve_test.png"
cm_path = REPORTS / "diagnostics" / "confusion_matrix_test.png"
if rel_path.exists():
    c1.image(str(rel_path), caption="Reliability Curve")
if cm_path.exists():
    c2.image(str(cm_path), caption="Confusion Matrix")
