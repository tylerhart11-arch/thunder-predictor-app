from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd


@dataclass
class FeatureBuildResult:
    features_df: pd.DataFrame
    feature_cols: list[str]


BASE_STAT_COLS = [
    "WIN_PCT_5",
    "WIN_PCT_10",
    "PT_DIFF_5",
    "PT_DIFF_10",
    "PTS_FOR_5",
    "PTS_FOR_10",
    "PTS_AGAINST_5",
    "PTS_AGAINST_10",
    "SEASON_WIN_PCT",
    "SEASON_PT_DIFF",
    "OFF_RATING_PROXY_10",
    "DEF_RATING_PROXY_10",
]


def build_team_feature_table(clean_team_games: pd.DataFrame, min_games_for_features: int = 5) -> pd.DataFrame:
    if clean_team_games.empty:
        return clean_team_games

    df = clean_team_games.copy()
    df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"])
    df = df.sort_values(["TEAM_ID", "GAME_DATE", "GAME_ID"]).reset_index(drop=True)

    team_grp = df.groupby("TEAM_ID", group_keys=False)
    season_grp = df.groupby(["TEAM_ID", "SEASON"], group_keys=False)

    df["GAMES_PLAYED_PRE"] = season_grp.cumcount()
    df["GAMES_PLAYED_POST"] = df["GAMES_PLAYED_PRE"] + 1
    prev_date = team_grp["GAME_DATE"].shift(1)
    df["REST_DAYS_PRE"] = (df["GAME_DATE"] - prev_date).dt.days.astype(float)
    df["REST_DAYS_PRE"] = df["REST_DAYS_PRE"].fillna(4.0).clip(lower=0)
    df["B2B_PRE"] = (df["REST_DAYS_PRE"] == 1).astype(int)

    for window in (5, 10):
        df[f"WIN_PCT_{window}_PRE"] = team_grp["TEAM_WIN"].transform(
            lambda s: s.shift(1).rolling(window=window, min_periods=1).mean()
        )
        df[f"PT_DIFF_{window}_PRE"] = team_grp["POINT_DIFF"].transform(
            lambda s: s.shift(1).rolling(window=window, min_periods=1).mean()
        )
        df[f"PTS_FOR_{window}_PRE"] = team_grp["PTS"].transform(
            lambda s: s.shift(1).rolling(window=window, min_periods=1).mean()
        )
        df[f"PTS_AGAINST_{window}_PRE"] = team_grp["OPP_PTS"].transform(
            lambda s: s.shift(1).rolling(window=window, min_periods=1).mean()
        )

        df[f"WIN_PCT_{window}_POST"] = team_grp["TEAM_WIN"].transform(
            lambda s: s.rolling(window=window, min_periods=1).mean()
        )
        df[f"PT_DIFF_{window}_POST"] = team_grp["POINT_DIFF"].transform(
            lambda s: s.rolling(window=window, min_periods=1).mean()
        )
        df[f"PTS_FOR_{window}_POST"] = team_grp["PTS"].transform(
            lambda s: s.rolling(window=window, min_periods=1).mean()
        )
        df[f"PTS_AGAINST_{window}_POST"] = team_grp["OPP_PTS"].transform(
            lambda s: s.rolling(window=window, min_periods=1).mean()
        )

    df["SEASON_WIN_PCT_PRE"] = season_grp["TEAM_WIN"].transform(lambda s: s.shift(1).expanding().mean())
    df["SEASON_PT_DIFF_PRE"] = season_grp["POINT_DIFF"].transform(lambda s: s.shift(1).expanding().mean())
    df["SEASON_WIN_PCT_POST"] = season_grp["TEAM_WIN"].transform(lambda s: s.expanding().mean())
    df["SEASON_PT_DIFF_POST"] = season_grp["POINT_DIFF"].transform(lambda s: s.expanding().mean())

    df["OFF_RATING_PROXY_10_PRE"] = df["PTS_FOR_10_PRE"]
    df["DEF_RATING_PROXY_10_PRE"] = df["PTS_AGAINST_10_PRE"]
    df["OFF_RATING_PROXY_10_POST"] = df["PTS_FOR_10_POST"]
    df["DEF_RATING_PROXY_10_POST"] = df["PTS_AGAINST_10_POST"]

    neutral_fill = {
        "SEASON_WIN_PCT_PRE": 0.5,
        "SEASON_PT_DIFF_PRE": 0.0,
        "WIN_PCT_5_PRE": 0.5,
        "WIN_PCT_10_PRE": 0.5,
        "PT_DIFF_5_PRE": 0.0,
        "PT_DIFF_10_PRE": 0.0,
        "PTS_FOR_5_PRE": df["PTS"].median(),
        "PTS_FOR_10_PRE": df["PTS"].median(),
        "PTS_AGAINST_5_PRE": df["OPP_PTS"].median(),
        "PTS_AGAINST_10_PRE": df["OPP_PTS"].median(),
        "OFF_RATING_PROXY_10_PRE": df["PTS"].median(),
        "DEF_RATING_PROXY_10_PRE": df["OPP_PTS"].median(),
    }

    for col, val in neutral_fill.items():
        if col in df.columns:
            df[col] = df[col].fillna(val)

    # Reduce noise from very early-season games.
    early_mask = df["GAMES_PLAYED_PRE"] < min_games_for_features
    df.loc[early_mask, "WIN_PCT_10_PRE"] = 0.5
    df.loc[early_mask, "PT_DIFF_10_PRE"] = 0.0

    return df


def _with_prefixed_cols(df: pd.DataFrame, prefix: str, cols: Sequence[str]) -> pd.DataFrame:
    rename_map = {c: f"{prefix}{c}" for c in cols}
    return df.rename(columns=rename_map)


def _add_diff_features(df: pd.DataFrame, base_cols: Sequence[str], left_prefix: str, right_prefix: str) -> pd.DataFrame:
    out = df.copy()
    for col in base_cols:
        out[f"DIFF_{col}"] = out[f"{left_prefix}{col}"] - out[f"{right_prefix}{col}"]
    return out


def build_training_features(game_level_df: pd.DataFrame, team_feature_df: pd.DataFrame) -> FeatureBuildResult:
    if game_level_df.empty or team_feature_df.empty:
        return FeatureBuildResult(features_df=pd.DataFrame(), feature_cols=[])

    pre_cols = [
        "TEAM_ID",
        "GAME_ID",
        "GAME_DATE",
        "GAMES_PLAYED_PRE",
        "REST_DAYS_PRE",
        "B2B_PRE",
    ] + [f"{c}_PRE" for c in BASE_STAT_COLS]

    tf = team_feature_df[pre_cols].copy()
    home = _with_prefixed_cols(tf, "HOME_", [c for c in pre_cols if c != "GAME_ID"])
    away = _with_prefixed_cols(tf, "AWAY_", [c for c in pre_cols if c != "GAME_ID"])

    merged = (
        game_level_df.merge(
            home,
            left_on=["GAME_ID", "HOME_TEAM_ID"],
            right_on=["GAME_ID", "HOME_TEAM_ID"],
            how="left",
        )
        .merge(
            away,
            left_on=["GAME_ID", "AWAY_TEAM_ID"],
            right_on=["GAME_ID", "AWAY_TEAM_ID"],
            how="left",
        )
        .sort_values(["GAME_DATE", "GAME_ID"])
        .reset_index(drop=True)
    )

    diff_base = [f"{c}_PRE" for c in BASE_STAT_COLS]
    merged = _add_diff_features(merged, diff_base, "HOME_", "AWAY_")
    merged["REST_DAYS_DIFF"] = merged["HOME_REST_DAYS_PRE"] - merged["AWAY_REST_DAYS_PRE"]
    merged["B2B_DIFF"] = merged["HOME_B2B_PRE"] - merged["AWAY_B2B_PRE"]
    merged["GAMES_PLAYED_DIFF"] = merged["HOME_GAMES_PLAYED_PRE"] - merged["AWAY_GAMES_PLAYED_PRE"]
    merged["HOME_COURT_FLAG"] = 1
    merged["home_win"] = merged["HOME_WIN"].astype(int)
    merged = merged.drop(columns=["HOME_WIN"], errors="ignore")

    feature_cols = sorted(
        [c for c in merged.columns if c.startswith("DIFF_")]
        + ["REST_DAYS_DIFF", "B2B_DIFF", "GAMES_PLAYED_DIFF", "HOME_COURT_FLAG"]
    )
    return FeatureBuildResult(features_df=merged, feature_cols=feature_cols)


def build_upcoming_game_features(
    schedule_df: pd.DataFrame,
    team_feature_df: pd.DataFrame,
    training_feature_cols: Sequence[str],
) -> pd.DataFrame:
    if schedule_df.empty:
        return pd.DataFrame(columns=["GAME_ID", "GAME_DATE"] + list(training_feature_cols))

    sched = schedule_df.copy()
    sched["GAME_DATE"] = pd.to_datetime(sched["GAME_DATE"])
    sched = sched.sort_values(["GAME_DATE", "GAME_ID"]).reset_index(drop=True)

    post_cols = ["TEAM_ID", "GAME_DATE", "GAMES_PLAYED_POST"] + [f"{c}_POST" for c in BASE_STAT_COLS]
    state = team_feature_df[post_cols].copy().sort_values(["GAME_DATE", "TEAM_ID"])
    state["LAST_GAME_DATE"] = state["GAME_DATE"]

    def attach_side(frame: pd.DataFrame, side: str) -> pd.DataFrame:
        team_id_col = f"{side}_TEAM_ID"
        left = frame[["GAME_ID", "GAME_DATE", team_id_col]].copy().rename(columns={team_id_col: "TEAM_ID"})
        left = left.sort_values(["GAME_DATE", "TEAM_ID"])

        merged = pd.merge_asof(
            left,
            state,
            on="GAME_DATE",
            by="TEAM_ID",
            direction="backward",
            allow_exact_matches=False,
        )
        merged[f"{side}_REST_DAYS_PRE"] = (merged["GAME_DATE"] - merged["LAST_GAME_DATE"]).dt.days.astype(float)
        merged[f"{side}_REST_DAYS_PRE"] = merged[f"{side}_REST_DAYS_PRE"].fillna(4.0).clip(lower=0)
        merged[f"{side}_B2B_PRE"] = (merged[f"{side}_REST_DAYS_PRE"] == 1).astype(int)
        merged[f"{side}_GAMES_PLAYED_PRE"] = merged["GAMES_PLAYED_POST"].fillna(0.0)

        rename_map = {f"{c}_POST": f"{side}_{c}_PRE" for c in BASE_STAT_COLS}
        merged = merged.rename(columns=rename_map)

        keep_cols = (
            ["GAME_ID"]
            + [f"{side}_{c}_PRE" for c in BASE_STAT_COLS]
            + [f"{side}_REST_DAYS_PRE", f"{side}_B2B_PRE", f"{side}_GAMES_PLAYED_PRE"]
        )
        return merged[keep_cols]

    home = attach_side(sched, "HOME")
    away = attach_side(sched, "AWAY")
    merged = sched.merge(home, on="GAME_ID", how="left").merge(away, on="GAME_ID", how="left")

    diff_base = [f"{c}_PRE" for c in BASE_STAT_COLS]
    merged = _add_diff_features(merged, diff_base, "HOME_", "AWAY_")
    merged["REST_DAYS_DIFF"] = merged["HOME_REST_DAYS_PRE"] - merged["AWAY_REST_DAYS_PRE"]
    merged["B2B_DIFF"] = merged["HOME_B2B_PRE"] - merged["AWAY_B2B_PRE"]
    merged["GAMES_PLAYED_DIFF"] = merged["HOME_GAMES_PLAYED_PRE"] - merged["AWAY_GAMES_PLAYED_PRE"]
    merged["HOME_COURT_FLAG"] = 1

    for col in training_feature_cols:
        if col not in merged.columns:
            merged[col] = np.nan
    id_cols = [
        c
        for c in [
            "GAME_ID",
            "GAME_DATE",
            "HOME_TEAM_ID",
            "AWAY_TEAM_ID",
            "HOME_TEAM_ABBREVIATION",
            "AWAY_TEAM_ABBREVIATION",
            "GAME_STATUS_TEXT",
        ]
        if c in merged.columns
    ]
    merged = merged[id_cols + list(training_feature_cols)]
    return merged
