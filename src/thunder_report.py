from __future__ import annotations

"""Thunder-specific reporting helpers used by the pipeline and dashboard."""

import pandas as pd


def thunder_predictions_only(pred_archive: pd.DataFrame, thunder_abbr: str = "OKC") -> pd.DataFrame:
    """Return only archive rows involving the configured Thunder team."""
    if pred_archive.empty:
        return pred_archive

    mask = (pred_archive["HOME_TEAM_ABBREVIATION"] == thunder_abbr) | (
        pred_archive["AWAY_TEAM_ABBREVIATION"] == thunder_abbr
    )
    return pred_archive.loc[mask].copy().sort_values("GAME_DATE")


def build_thunder_summary(thunder_df: pd.DataFrame) -> dict:
    """Summarize completed Thunder predictions for dashboard headline metrics."""
    if thunder_df.empty:
        return {
            "games": 0,
            "accuracy": None,
            "correct": 0,
            "incorrect": 0,
            "avg_confidence_correct": None,
            "avg_confidence_incorrect": None,
        }

    completed = thunder_df[thunder_df["ACTUAL_HOME_WIN"].notna()].copy()
    if completed.empty:
        return {
            "games": 0,
            "accuracy": None,
            "correct": 0,
            "incorrect": 0,
            "avg_confidence_correct": None,
            "avg_confidence_incorrect": None,
        }

    completed["PRED_HOME_WIN"] = (completed["PRED_HOME_WIN_PROB"] >= 0.5).astype(int)
    completed["IS_CORRECT"] = (completed["PRED_HOME_WIN"] == completed["ACTUAL_HOME_WIN"]).astype(int)
    completed["CONFIDENCE"] = (completed["PRED_HOME_WIN_PROB"] - 0.5).abs() * 2

    correct = int(completed["IS_CORRECT"].sum())
    total = int(len(completed))
    incorrect = total - correct
    return {
        "games": total,
        "accuracy": correct / total if total else None,
        "correct": correct,
        "incorrect": incorrect,
        "avg_confidence_correct": float(completed.loc[completed["IS_CORRECT"] == 1, "CONFIDENCE"].mean())
        if correct > 0
        else None,
        "avg_confidence_incorrect": float(completed.loc[completed["IS_CORRECT"] == 0, "CONFIDENCE"].mean())
        if incorrect > 0
        else None,
    }


def add_rolling_accuracy(thunder_df: pd.DataFrame, window: int = 10) -> pd.DataFrame:
    """Add rolling and cumulative hit-rate columns for completed Thunder predictions."""
    if thunder_df.empty:
        return thunder_df

    out = thunder_df.copy().sort_values("GAME_DATE")
    out["PRED_HOME_WIN"] = (out["PRED_HOME_WIN_PROB"] >= 0.5).astype(int)
    out["IS_CORRECT"] = (out["PRED_HOME_WIN"] == out["ACTUAL_HOME_WIN"]).astype(float)
    out["ROLLING_ACCURACY"] = out["IS_CORRECT"].rolling(window=window, min_periods=1).mean()
    out["CUM_CORRECT"] = out["IS_CORRECT"].fillna(0).cumsum()
    out["CUM_GAMES"] = out["IS_CORRECT"].notna().cumsum()
    out["CUM_ACCURACY"] = out["CUM_CORRECT"] / out["CUM_GAMES"].replace(0, pd.NA)
    return out


def weekly_performance(thunder_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate Thunder prediction performance into weekly buckets."""
    if thunder_df.empty:
        return pd.DataFrame(columns=["WEEK", "games", "accuracy"])

    df = thunder_df.copy()
    df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"])
    df["WEEK"] = df["GAME_DATE"].dt.to_period("W").dt.start_time
    df["PRED_HOME_WIN"] = (df["PRED_HOME_WIN_PROB"] >= 0.5).astype(int)
    df["IS_CORRECT"] = (df["PRED_HOME_WIN"] == df["ACTUAL_HOME_WIN"]).astype(float)
    return (
        df.groupby("WEEK", as_index=False)
        .agg(games=("GAME_ID", "count"), accuracy=("IS_CORRECT", "mean"))
        .sort_values("WEEK")
    )
