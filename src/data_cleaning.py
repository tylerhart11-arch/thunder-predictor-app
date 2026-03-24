from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from src.utils import normalize_game_id


def clean_league_logs(raw_df: pd.DataFrame) -> pd.DataFrame:
    if raw_df.empty:
        return raw_df

    df = raw_df.copy()
    df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"])
    df["TEAM_ID"] = df["TEAM_ID"].astype(int)
    df["GAME_ID"] = normalize_game_id(df["GAME_ID"])

    matchup = df["MATCHUP"].fillna("").astype(str)
    is_home = matchup.str.contains(r"\bvs\.?\b", case=False, regex=True)
    df["IS_HOME"] = is_home.astype(int)
    df["OPP_TEAM_ABBREVIATION"] = np.where(
        is_home,
        matchup.str.split(r"\s+vs\.?\s+", regex=True).str[-1],
        matchup.str.split(r"\s+@\s+", regex=True).str[-1],
    )
    df["TEAM_WIN"] = (df["WL"] == "W").astype(int)
    df["OPP_PTS"] = df["PTS"] - df["PLUS_MINUS"]
    df["POINT_DIFF"] = df["PLUS_MINUS"]

    keep = [
        "SEASON_ID",
        "SEASON",
        "SEASON_TYPE",
        "GAME_ID",
        "GAME_DATE",
        "TEAM_ID",
        "TEAM_ABBREVIATION",
        "TEAM_NAME",
        "OPP_TEAM_ABBREVIATION",
        "IS_HOME",
        "TEAM_WIN",
        "PTS",
        "OPP_PTS",
        "POINT_DIFF",
        "FG_PCT",
        "FG3_PCT",
        "FT_PCT",
        "REB",
        "AST",
        "TOV",
    ]
    cleaned = (
        df[keep]
        .sort_values(["GAME_DATE", "GAME_ID", "TEAM_ID"])
        .drop_duplicates(subset=["GAME_ID", "TEAM_ID"], keep="last")
        .reset_index(drop=True)
    )
    return cleaned


def build_game_level_table(clean_team_games: pd.DataFrame) -> pd.DataFrame:
    if clean_team_games.empty:
        return clean_team_games

    home = clean_team_games[clean_team_games["IS_HOME"] == 1].copy()
    away = clean_team_games[clean_team_games["IS_HOME"] == 0].copy()

    key_cols = ["GAME_ID", "GAME_DATE", "SEASON", "SEASON_TYPE"]
    home = home.drop_duplicates(subset=key_cols + ["TEAM_ID"])
    away = away.drop_duplicates(subset=key_cols + ["TEAM_ID"])

    home_counts = home.groupby("GAME_ID").size().rename("home_n")
    away_counts = away.groupby("GAME_ID").size().rename("away_n")
    valid = home_counts.to_frame().join(away_counts, how="inner")
    valid_ids = valid[(valid["home_n"] == 1) & (valid["away_n"] == 1)].index

    home = home[home["GAME_ID"].isin(valid_ids)].copy()
    away = away[away["GAME_ID"].isin(valid_ids)].copy()

    home = home.rename(
        columns={
            "TEAM_ID": "HOME_TEAM_ID",
            "TEAM_ABBREVIATION": "HOME_TEAM_ABBREVIATION",
            "TEAM_NAME": "HOME_TEAM_NAME",
            "PTS": "HOME_PTS",
            "TEAM_WIN": "HOME_WIN",
            "POINT_DIFF": "HOME_POINT_DIFF",
        }
    )
    away = away.rename(
        columns={
            "TEAM_ID": "AWAY_TEAM_ID",
            "TEAM_ABBREVIATION": "AWAY_TEAM_ABBREVIATION",
            "TEAM_NAME": "AWAY_TEAM_NAME",
            "PTS": "AWAY_PTS",
            "TEAM_WIN": "AWAY_WIN",
            "POINT_DIFF": "AWAY_POINT_DIFF",
        }
    )

    merge_cols = ["GAME_ID", "GAME_DATE", "SEASON", "SEASON_TYPE"]
    game_level = home.merge(
        away[
            merge_cols
            + [
                "AWAY_TEAM_ID",
                "AWAY_TEAM_ABBREVIATION",
                "AWAY_TEAM_NAME",
                "AWAY_PTS",
                "AWAY_WIN",
                "AWAY_POINT_DIFF",
            ]
        ],
        on=merge_cols,
        how="inner",
        validate="one_to_one",
    )

    game_level["HOME_WIN"] = game_level["HOME_WIN"].astype(int)
    game_level["AWAY_WIN"] = game_level["AWAY_WIN"].astype(int)
    game_level["POINT_DIFF"] = game_level["HOME_PTS"] - game_level["AWAY_PTS"]
    game_level["GAME_DATE"] = pd.to_datetime(game_level["GAME_DATE"])

    ordered_cols = [
        "GAME_ID",
        "GAME_DATE",
        "SEASON",
        "SEASON_TYPE",
        "HOME_TEAM_ID",
        "HOME_TEAM_ABBREVIATION",
        "HOME_TEAM_NAME",
        "AWAY_TEAM_ID",
        "AWAY_TEAM_ABBREVIATION",
        "AWAY_TEAM_NAME",
        "HOME_PTS",
        "AWAY_PTS",
        "POINT_DIFF",
        "HOME_WIN",
    ]
    return game_level[ordered_cols].sort_values(["GAME_DATE", "GAME_ID"]).reset_index(drop=True)


def merge_actuals_from_scoreboard(
    game_level_df: pd.DataFrame,
    scoreboard_df: pd.DataFrame,
) -> pd.DataFrame:
    if game_level_df.empty:
        return game_level_df
    if scoreboard_df.empty:
        return game_level_df

    base = game_level_df.copy()
    score = scoreboard_df.copy()
    score["GAME_ID"] = normalize_game_id(score["GAME_ID"])
    score["GAME_DATE"] = pd.to_datetime(score["GAME_DATE"])

    score_final = score[score["IS_FINAL"]].copy()
    if score_final.empty:
        return base

    score_final["FINAL_HOME_WIN"] = (score_final["HOME_PTS"] > score_final["AWAY_PTS"]).astype(int)
    final_keep = ["GAME_ID", "FINAL_HOME_WIN", "HOME_PTS", "AWAY_PTS", "GAME_STATUS_TEXT"]
    out = base.merge(score_final[final_keep], on="GAME_ID", how="left", suffixes=("", "_SCOREBOARD"))
    out["HOME_WIN"] = out["FINAL_HOME_WIN"].fillna(out["HOME_WIN"]).astype(int)
    out["HOME_PTS"] = out["HOME_PTS_SCOREBOARD"].combine_first(out["HOME_PTS"])
    out["AWAY_PTS"] = out["AWAY_PTS_SCOREBOARD"].combine_first(out["AWAY_PTS"])
    out["POINT_DIFF"] = out["HOME_PTS"] - out["AWAY_PTS"]
    drop_cols = ["FINAL_HOME_WIN", "HOME_PTS_SCOREBOARD", "AWAY_PTS_SCOREBOARD", "GAME_STATUS_TEXT"]
    return out.drop(columns=[c for c in drop_cols if c in out.columns])
