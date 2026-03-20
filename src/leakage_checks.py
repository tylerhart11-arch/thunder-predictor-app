from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd


FORBIDDEN_TOKENS = ["HOME_PTS", "AWAY_PTS", "POINT_DIFF", "HOME_WIN", "AWAY_WIN", "TEAM_WIN"]


def run_leakage_checks(df: pd.DataFrame, feature_cols: Sequence[str], target_col: str) -> dict:
    report: dict[str, object] = {}
    fcols = list(feature_cols)

    forbidden_cols = [c for c in fcols if any(token in c for token in FORBIDDEN_TOKENS)]
    report["forbidden_feature_names"] = forbidden_cols

    if target_col in df.columns:
        target = df[target_col].astype(float)
        suspicious = []
        for col in fcols:
            series = pd.to_numeric(df[col], errors="coerce")
            if series.nunique(dropna=True) <= 1:
                continue
            corr = series.corr(target)
            if pd.notna(corr) and abs(corr) > 0.98:
                suspicious.append({"feature": col, "corr_to_target": float(corr)})
        report["near_perfect_correlations"] = suspicious
    else:
        report["near_perfect_correlations"] = []

    report["feature_null_pct"] = (
        df[fcols].isna().mean().sort_values(ascending=False).head(20).round(4).to_dict()
        if fcols
        else {}
    )
    report["class_balance"] = (
        float(df[target_col].mean()) if target_col in df.columns and len(df) > 0 else np.nan
    )
    report["passed"] = len(forbidden_cols) == 0 and len(report["near_perfect_correlations"]) == 0
    return report

