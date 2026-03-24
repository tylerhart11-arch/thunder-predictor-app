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


def _result_streak(values: pd.Series) -> tuple[int, str | None]:
    if values.empty:
        return 0, None

    streak = 0
    streak_type: int | None = None
    for value in values.fillna(0).astype(int):
        if streak_type is None or value != streak_type:
            streak_type = value
            streak = 1
        else:
            streak += 1

    if streak_type is None:
        return 0, None
    return (streak if streak_type == 1 else -streak), ("W" if streak_type == 1 else "L")


def confidence_from_probability(home_prob: float | int | None) -> float | None:
    if home_prob is None or pd.isna(home_prob):
        return None
    return abs(float(home_prob) - 0.5) * 2


def confidence_band(home_prob: float | int | None) -> str:
    confidence = confidence_from_probability(home_prob)
    if confidence is None:
        return "N/A"
    if confidence >= 0.60:
        return "Strong edge"
    if confidence >= 0.30:
        return "Lean"
    return "Toss-up"


def build_thunder_summary(thunder_df: pd.DataFrame) -> dict:
    """Summarize completed Thunder predictions for dashboard headline metrics."""
    if thunder_df.empty:
        return {
            "games": 0,
            "accuracy": None,
            "last_10_accuracy": None,
            "correct": 0,
            "incorrect": 0,
            "current_streak": 0,
            "current_streak_label": None,
            "best_streak": 0,
            "worst_streak": 0,
            "avg_confidence": None,
            "avg_confidence_correct": None,
            "avg_confidence_incorrect": None,
        }

    completed = thunder_df[thunder_df["ACTUAL_HOME_WIN"].notna()].copy()
    if completed.empty:
        return {
            "games": 0,
            "accuracy": None,
            "last_10_accuracy": None,
            "correct": 0,
            "incorrect": 0,
            "current_streak": 0,
            "current_streak_label": None,
            "best_streak": 0,
            "worst_streak": 0,
            "avg_confidence": None,
            "avg_confidence_correct": None,
            "avg_confidence_incorrect": None,
        }

    completed = completed.sort_values("GAME_DATE")
    completed["PRED_HOME_WIN"] = (completed["PRED_HOME_WIN_PROB"] >= 0.5).astype(int)
    completed["IS_CORRECT"] = (completed["PRED_HOME_WIN"] == completed["ACTUAL_HOME_WIN"]).astype(int)
    completed["CONFIDENCE"] = completed["PRED_HOME_WIN_PROB"].map(confidence_from_probability)

    streak_values: list[int] = []
    streak = 0
    current_flag: int | None = None
    for value in completed["IS_CORRECT"].fillna(0).astype(int):
        if current_flag is None or value != current_flag:
            current_flag = value
            streak = 1
        else:
            streak += 1
        streak_values.append(streak if value == 1 else -streak)
    current_streak, current_streak_label = _result_streak(completed["IS_CORRECT"])
    last_10_accuracy = float(completed["IS_CORRECT"].tail(10).mean()) if len(completed) else None
    best_streak = max((value for value in streak_values if value > 0), default=0)
    worst_streak = -max((abs(value) for value in streak_values if value < 0), default=0)

    correct = int(completed["IS_CORRECT"].sum())
    total = int(len(completed))
    incorrect = total - correct
    return {
        "games": total,
        "accuracy": correct / total if total else None,
        "last_10_accuracy": last_10_accuracy,
        "correct": correct,
        "incorrect": incorrect,
        "current_streak": current_streak,
        "current_streak_label": current_streak_label,
        "best_streak": int(best_streak),
        "worst_streak": int(worst_streak),
        "avg_confidence": float(completed["CONFIDENCE"].mean()) if total else None,
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
    out["ROLLING_ACCURACY_5"] = out["IS_CORRECT"].rolling(window=5, min_periods=1).mean()
    out["ROLLING_ACCURACY_10"] = out["IS_CORRECT"].rolling(window=10, min_periods=1).mean()
    out["CUM_CORRECT"] = out["IS_CORRECT"].fillna(0).cumsum()
    out["CUM_GAMES"] = out["IS_CORRECT"].notna().cumsum()
    out["CUM_ACCURACY"] = out["CUM_CORRECT"] / out["CUM_GAMES"].replace(0, pd.NA)
    out["CONFIDENCE"] = out["PRED_HOME_WIN_PROB"].map(confidence_from_probability)
    out["CONFIDENCE_BAND"] = out["PRED_HOME_WIN_PROB"].map(confidence_band)
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
    df["CONFIDENCE"] = df["PRED_HOME_WIN_PROB"].map(confidence_from_probability)
    return (
        df.groupby("WEEK", as_index=False)
        .agg(games=("GAME_ID", "count"), accuracy=("IS_CORRECT", "mean"), avg_confidence=("CONFIDENCE", "mean"))
        .sort_values("WEEK")
    )
