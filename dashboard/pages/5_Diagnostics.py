from __future__ import annotations

import streamlit as st

from dashboard.helpers import REPORTS, ROOT, apply_casino_theme, read_csv, read_json


apply_casino_theme(
    page_title="Model Diagnostics",
    subtitle="Leakage checks, split boundaries, tuning trails, and calibration payloads.",
)

leakage = read_json(REPORTS / "leakage_report.json")
dq = read_json(REPORTS / "data_quality.json")

if leakage:
    st.subheader("Leakage Checks")
    st.json(leakage)
else:
    st.info("Leakage report not found.")

if dq:
    st.subheader("Data Quality Snapshot")
    st.json(dq)

split_meta = read_json(ROOT / "data" / "artifacts" / "split_meta.json")
if split_meta:
    st.subheader("Chronological Split Metadata")
    st.json(split_meta)

tuning = read_csv(REPORTS / "improved_tuning_results.csv")
if not tuning.empty:
    st.subheader("Improved Model Tuning Results")
    st.dataframe(tuning, use_container_width=True)

rel = read_csv(REPORTS / "diagnostics" / "reliability_curve_test.csv")
if not rel.empty:
    st.subheader("Reliability Curve Data")
    st.dataframe(rel, use_container_width=True)
