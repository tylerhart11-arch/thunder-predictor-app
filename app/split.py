from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd


@dataclass
class SplitResult:
    train_df: pd.DataFrame
    valid_df: pd.DataFrame
    test_df: pd.DataFrame
    train_end_date: pd.Timestamp
    valid_end_date: pd.Timestamp
    split_type: str = "chronological_ratio"
    train_seasons: list[str] | None = None
    valid_seasons: list[str] | None = None
    test_seasons: list[str] | None = None


def chronological_split(
    df: pd.DataFrame,
    date_col: str,
    train_ratio: float = 0.70,
    validation_ratio: float = 0.15,
) -> SplitResult:
    if df.empty:
        empty = pd.DataFrame()
        return SplitResult(empty, empty, empty, pd.NaT, pd.NaT)

    ordered = df.sort_values(date_col).reset_index(drop=True).copy()
    ordered[date_col] = pd.to_datetime(ordered[date_col])
    unique_dates = ordered[date_col].drop_duplicates().sort_values().reset_index(drop=True)

    n_dates = len(unique_dates)
    train_cut_idx = max(0, int(n_dates * train_ratio) - 1)
    valid_cut_idx = max(train_cut_idx + 1, int(n_dates * (train_ratio + validation_ratio)) - 1)
    valid_cut_idx = min(valid_cut_idx, n_dates - 2) if n_dates >= 3 else valid_cut_idx

    train_end = unique_dates.iloc[train_cut_idx]
    valid_end = unique_dates.iloc[valid_cut_idx]

    train_df = ordered[ordered[date_col] <= train_end].copy()
    valid_df = ordered[(ordered[date_col] > train_end) & (ordered[date_col] <= valid_end)].copy()
    test_df = ordered[ordered[date_col] > valid_end].copy()

    return SplitResult(
        train_df=train_df,
        valid_df=valid_df,
        test_df=test_df,
        train_end_date=train_end,
        valid_end_date=valid_end,
        split_type="chronological_ratio",
        train_seasons=sorted(train_df["SEASON"].dropna().unique().tolist()) if "SEASON" in train_df.columns else [],
        valid_seasons=sorted(valid_df["SEASON"].dropna().unique().tolist()) if "SEASON" in valid_df.columns else [],
        test_seasons=sorted(test_df["SEASON"].dropna().unique().tolist()) if "SEASON" in test_df.columns else [],
    )


def _season_start_year(season_label: str) -> int:
    # Expects NBA season labels like "2025-26".
    return int(str(season_label).split("-")[0])


def season_holdout_split(
    df: pd.DataFrame,
    date_col: str = "GAME_DATE",
    season_col: str = "SEASON",
) -> SplitResult:
    if df.empty:
        empty = pd.DataFrame()
        return SplitResult(empty, empty, empty, pd.NaT, pd.NaT, split_type="season_holdout_current")

    if season_col not in df.columns:
        return chronological_split(df=df, date_col=date_col)

    ordered = df.copy()
    ordered[date_col] = pd.to_datetime(ordered[date_col])
    ordered = ordered.sort_values([date_col, "GAME_ID"] if "GAME_ID" in ordered.columns else [date_col]).reset_index(
        drop=True
    )

    seasons = sorted(ordered[season_col].dropna().astype(str).unique().tolist(), key=_season_start_year)
    if len(seasons) < 3:
        # Fallback for minimal history.
        split = chronological_split(df=ordered, date_col=date_col)
        split.split_type = "chronological_ratio_fallback"
        return split

    test_season = seasons[-1]
    valid_season = seasons[-2]
    train_seasons = seasons[:-2]

    train_df = ordered[ordered[season_col].isin(train_seasons)].copy()
    valid_df = ordered[ordered[season_col] == valid_season].copy()
    test_df = ordered[ordered[season_col] == test_season].copy()

    # Safety fallback if train would be empty.
    if train_df.empty:
        return chronological_split(df=ordered, date_col=date_col)

    train_end = pd.to_datetime(train_df[date_col]).max() if not train_df.empty else pd.NaT
    valid_end = pd.to_datetime(valid_df[date_col]).max() if not valid_df.empty else train_end

    return SplitResult(
        train_df=train_df,
        valid_df=valid_df,
        test_df=test_df,
        train_end_date=train_end,
        valid_end_date=valid_end,
        split_type="season_holdout_current",
        train_seasons=train_seasons,
        valid_seasons=[valid_season],
        test_seasons=[test_season],
    )


def split_summary_table(split: SplitResult, date_col: str = "GAME_DATE", season_col: str = "SEASON") -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for label, frame in [("train", split.train_df), ("validation", split.valid_df), ("test", split.test_df)]:
        if frame.empty:
            rows.append(
                {
                    "split_set": label,
                    "rows": 0,
                    "date_min": None,
                    "date_max": None,
                    "seasons": "",
                    "home_win_rate": None,
                }
            )
            continue
        seasons = sorted(frame[season_col].dropna().astype(str).unique().tolist()) if season_col in frame.columns else []
        rows.append(
            {
                "split_set": label,
                "rows": int(len(frame)),
                "date_min": str(pd.to_datetime(frame[date_col]).min().date()) if date_col in frame.columns else None,
                "date_max": str(pd.to_datetime(frame[date_col]).max().date()) if date_col in frame.columns else None,
                "seasons": ", ".join(seasons),
                "home_win_rate": float(frame["home_win"].mean()) if "home_win" in frame.columns else None,
            }
        )
    return pd.DataFrame(rows)
