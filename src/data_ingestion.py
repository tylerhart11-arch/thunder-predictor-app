from __future__ import annotations

import os
import time
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import pandas as pd
from nba_api.stats.endpoints import leaguegamelog, scoreboardv2

from src.utils import daterange, historical_seasons


@dataclass
class IngestionResult:
    league_logs: pd.DataFrame
    scoreboard_games: pd.DataFrame


class NBADataIngestion:
    def __init__(self, cfg: dict[str, Any], logger):
        self.cfg = cfg
        self.logger = logger
        self._sanitize_proxy_env()

    def _sanitize_proxy_env(self) -> None:
        proxy_vars = ("HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy")
        cleared: set[str] = set()

        for name in proxy_vars:
            raw = os.environ.get(name, "").strip()
            if not raw:
                continue

            parsed = urlparse(raw if "://" in raw else f"http://{raw}")
            host = (parsed.hostname or "").lower()
            port = parsed.port

            # `127.0.0.1:9` / `localhost:9` is a common poison-pill proxy value that
            # causes requests-based clients to fail immediately.
            if host in {"127.0.0.1", "localhost", "::1"} and port == 9:
                os.environ.pop(name, None)
                cleared.add(f"{name}={raw}")

        if cleared:
            self.logger.warning(
                "Cleared invalid proxy environment variables before NBA API calls: %s",
                ", ".join(sorted(cleared)),
            )

    def _safe_call(self, func, retries: int = 4, sleep_seconds: float = 1.5):
        last_exc = None
        for attempt in range(1, retries + 1):
            try:
                return func()
            except Exception as exc:  # noqa: BLE001
                last_exc = exc
                self.logger.warning("API call failed (attempt %s/%s): %s", attempt, retries, exc)
                time.sleep(sleep_seconds * attempt)
        raise RuntimeError(f"API call failed after {retries} attempts: {last_exc}") from last_exc

    def fetch_league_logs_for_season(self, season: str, season_type: str = "Regular Season") -> pd.DataFrame:
        self.logger.info("Fetching LeagueGameLog for season=%s season_type=%s", season, season_type)

        def _call():
            endpoint = leaguegamelog.LeagueGameLog(
                counter=0,
                direction="ASC",
                league_id="00",
                player_or_team_abbreviation="T",
                season=season,
                season_type_all_star=season_type,
                sorter="DATE",
            )
            return endpoint.get_data_frames()[0]

        df = self._safe_call(_call)
        if df.empty:
            self.logger.warning("No rows returned for season=%s", season)
            return df

        df["SEASON"] = season
        return df

    def fetch_historical_league_logs(self, full_history: bool = True) -> pd.DataFrame:
        start_year = int(self.cfg["data"]["historical_start_season"])
        current_start = self._current_season_start_year()
        if full_history:
            seasons = historical_seasons(start_year, current_start)
        else:
            seasons = historical_seasons(max(start_year, current_start - 1), current_start)

        frames: list[pd.DataFrame] = []
        season_types = ["Regular Season"]
        if self.cfg["data"].get("include_playoffs", False):
            season_types.append("Playoffs")

        for season in seasons:
            for season_type in season_types:
                frame = self.fetch_league_logs_for_season(season=season, season_type=season_type)
                if not frame.empty:
                    frame["SEASON_TYPE"] = season_type
                    frames.append(frame)

        if not frames:
            return pd.DataFrame()

        league_logs = pd.concat(frames, ignore_index=True)
        league_logs = league_logs.drop_duplicates(subset=["GAME_ID", "TEAM_ID", "GAME_DATE"]).reset_index(drop=True)
        self.logger.info("Historical league logs pulled: %s rows", len(league_logs))
        return league_logs

    def fetch_scoreboard_for_date(self, d: date) -> pd.DataFrame:
        self.logger.info("Fetching ScoreboardV2 for %s", d.isoformat())
        game_date_param = d.strftime("%m/%d/%Y")

        def _call():
            endpoint = scoreboardv2.ScoreboardV2(day_offset=0, game_date=game_date_param, league_id="00")
            payload = endpoint.get_dict()
            result_sets = payload.get("resultSets", [])
            data_map: dict[str, pd.DataFrame] = {}
            for rs in result_sets:
                name = rs.get("name")
                headers = rs.get("headers", [])
                rows = rs.get("rowSet", [])
                data_map[name] = pd.DataFrame(rows, columns=headers)
            game_header = data_map.get("GameHeader", pd.DataFrame())
            line_score = data_map.get("LineScore", pd.DataFrame())
            return game_header, line_score

        game_header, line_score = self._safe_call(_call)
        if game_header.empty:
            return pd.DataFrame()

        required_line_cols = {"GAME_ID", "TEAM_ID", "TEAM_ABBREVIATION", "PTS"}
        if line_score.empty or not required_line_cols.issubset(set(line_score.columns)):
            merged = game_header.copy()
            merged["HOME_TEAM_ABBREVIATION"] = pd.NA
            merged["AWAY_TEAM_ABBREVIATION"] = pd.NA
            merged["HOME_PTS"] = pd.NA
            merged["AWAY_PTS"] = pd.NA
        else:
            home = line_score.rename(
                columns={
                    "TEAM_ID": "HOME_TEAM_ID",
                    "TEAM_ABBREVIATION": "HOME_TEAM_ABBREVIATION",
                    "PTS": "HOME_PTS",
                }
            )
            away = line_score.rename(
                columns={
                    "TEAM_ID": "AWAY_TEAM_ID",
                    "TEAM_ABBREVIATION": "AWAY_TEAM_ABBREVIATION",
                    "PTS": "AWAY_PTS",
                }
            )

            merged = game_header.merge(
                home[["GAME_ID", "HOME_TEAM_ID", "HOME_TEAM_ABBREVIATION", "HOME_PTS"]],
                on=["GAME_ID", "HOME_TEAM_ID"],
                how="left",
            ).merge(
                away[["GAME_ID", "AWAY_TEAM_ID", "AWAY_TEAM_ABBREVIATION", "AWAY_PTS"]],
                left_on=["GAME_ID", "VISITOR_TEAM_ID"],
                right_on=["GAME_ID", "AWAY_TEAM_ID"],
                how="left",
            )
            if "AWAY_TEAM_ID" in merged.columns:
                merged = merged.drop(columns=["AWAY_TEAM_ID"])

        merged["GAME_DATE"] = pd.to_datetime(merged["GAME_DATE_EST"]).dt.date
        merged["IS_FINAL"] = merged["GAME_STATUS_TEXT"].str.contains("Final", case=False, na=False)
        merged["INGESTED_AT_UTC"] = pd.Timestamp.utcnow().isoformat()
        return merged[
            [
                "GAME_ID",
                "GAME_DATE",
                "GAME_STATUS_ID",
                "GAME_STATUS_TEXT",
                "HOME_TEAM_ID",
                "VISITOR_TEAM_ID",
                "HOME_TEAM_ABBREVIATION",
                "AWAY_TEAM_ABBREVIATION",
                "HOME_PTS",
                "AWAY_PTS",
                "IS_FINAL",
                "INGESTED_AT_UTC",
            ]
        ].rename(columns={"VISITOR_TEAM_ID": "AWAY_TEAM_ID"})

    def fetch_scoreboard_window(self, center_date: date | None = None) -> pd.DataFrame:
        center_date = center_date or date.today()
        back = int(self.cfg["data"]["scoreboard_days_back"])
        fwd = int(self.cfg["data"]["scoreboard_days_forward"])
        max_failed_dates = int(self.cfg["data"].get("scoreboard_abort_after_failed_dates", 2))
        start_date = center_date - timedelta(days=back)
        end_date = center_date + timedelta(days=fwd)
        days = daterange(start_date, end_date)

        frames: list[pd.DataFrame] = []
        consecutive_failures = 0
        for d in days:
            try:
                frame = self.fetch_scoreboard_for_date(d)
                consecutive_failures = 0
            except Exception as exc:  # noqa: BLE001
                consecutive_failures += 1
                self.logger.warning("Skipping scoreboard date %s due to repeated API failures: %s", d.isoformat(), exc)
                if consecutive_failures >= max_failed_dates:
                    raise RuntimeError(
                        "Aborting scoreboard window refresh after "
                        f"{consecutive_failures} consecutive failed dates; cached scoreboard data should be used."
                    ) from exc
                continue
            if not frame.empty:
                frames.append(frame)

        if not frames:
            return pd.DataFrame()

        out = pd.concat(frames, ignore_index=True)
        out = out.sort_values(["GAME_DATE", "GAME_ID"]).drop_duplicates(subset=["GAME_ID"], keep="last")
        self.logger.info("Scoreboard window rows: %s", len(out))
        return out

    @staticmethod
    def _current_season_start_year(today: date | None = None) -> int:
        today = today or date.today()
        return today.year if today.month >= 10 else today.year - 1
