from __future__ import annotations

from typing import Any

import pandas as pd


def basic_data_quality_report(df: pd.DataFrame, id_col: str = "GAME_ID") -> dict[str, Any]:
    if df.empty:
        return {"rows": 0, "duplicates": 0, "missing_pct": {}, "date_min": None, "date_max": None}

    missing_pct = (df.isna().mean().sort_values(ascending=False) * 100).round(2).to_dict()
    out = {
        "rows": int(len(df)),
        "duplicates": int(df[id_col].duplicated().sum()) if id_col in df.columns else 0,
        "missing_pct": missing_pct,
        "date_min": str(df["GAME_DATE"].min()) if "GAME_DATE" in df.columns else None,
        "date_max": str(df["GAME_DATE"].max()) if "GAME_DATE" in df.columns else None,
    }
    return out

