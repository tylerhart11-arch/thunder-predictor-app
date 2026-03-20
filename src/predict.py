from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd


def predict_dataframe(
    model,
    df: pd.DataFrame,
    feature_cols: Sequence[str],
    calibrator=None,
) -> pd.DataFrame:
    out = df.copy()
    X = out[list(feature_cols)]
    if calibrator is not None:
        probs = calibrator.predict_proba(X)[:, 1]
    else:
        probs = model.predict_proba(X)[:, 1]
    out["PRED_HOME_WIN_PROB"] = np.clip(probs, 1e-6, 1 - 1e-6)
    out["PRED_HOME_WIN"] = (out["PRED_HOME_WIN_PROB"] >= 0.5).astype(int)
    return out


def initialize_prediction_archive(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(
            columns=[
                "GAME_ID",
                "GAME_DATE",
                "HOME_TEAM_ABBREVIATION",
                "AWAY_TEAM_ABBREVIATION",
                "PRED_HOME_WIN_PROB",
                "PRED_HOME_WIN",
                "ACTUAL_HOME_WIN",
                "IS_CORRECT",
                "PREDICTION_TIMESTAMP_UTC",
            ]
        )
    archive = pd.read_csv(path)
    if "GAME_DATE" in archive.columns:
        archive["GAME_DATE"] = pd.to_datetime(archive["GAME_DATE"], errors="coerce")
    return archive


def reconcile_archive_with_actuals(archive_df: pd.DataFrame, actual_games: pd.DataFrame) -> pd.DataFrame:
    if archive_df.empty:
        return archive_df
    actual = actual_games[["GAME_ID", "HOME_WIN"]].copy().rename(columns={"HOME_WIN": "ACTUAL_HOME_WIN"})
    actual["GAME_ID"] = actual["GAME_ID"].astype(str)

    out = archive_df.copy()
    out["GAME_ID"] = out["GAME_ID"].astype(str)
    out = out.drop(columns=["ACTUAL_HOME_WIN", "IS_CORRECT"], errors="ignore").merge(actual, on="GAME_ID", how="left")
    known_mask = out["ACTUAL_HOME_WIN"].notna()
    out["IS_CORRECT"] = np.where(known_mask, (out["PRED_HOME_WIN"] == out["ACTUAL_HOME_WIN"]).astype(int), np.nan)
    return out
