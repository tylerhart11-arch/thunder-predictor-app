from __future__ import annotations

import json
import sqlite3
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any

import pandas as pd


def season_label(start_year: int) -> str:
    return f"{start_year}-{str(start_year + 1)[-2:]}"


def current_nba_season_start(today: date | None = None) -> int:
    today = today or date.today()
    return today.year if today.month >= 10 else today.year - 1


def historical_seasons(start_year: int, end_year: int) -> list[str]:
    return [season_label(y) for y in range(start_year, end_year + 1)]


def daterange(start_date: date, end_date: date) -> list[date]:
    days = (end_date - start_date).days
    return [start_date + timedelta(days=d) for d in range(days + 1)]


def save_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def save_json(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def load_csv_if_exists(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def to_sqlite(df: pd.DataFrame, table: str, sqlite_path: Path, if_exists: str = "replace") -> None:
    if df is None:
        return
    if len(df.columns) == 0:
        return
    out = df.copy()
    datetime_cols = out.select_dtypes(include=["datetime64[ns]", "datetime64[ns, UTC]"]).columns.tolist()
    for col in datetime_cols:
        out[col] = out[col].astype(str)
    object_cols = out.select_dtypes(include=["object"]).columns.tolist()
    for col in object_cols:
        out[col] = out[col].map(
            lambda value: value.isoformat()
            if isinstance(value, (pd.Timestamp, datetime, date))
            else value
        )
    sqlite_path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(sqlite_path) as conn:
        out.to_sql(table, conn, if_exists=if_exists, index=False)


def now_utc_iso() -> str:
    return datetime.utcnow().replace(microsecond=0).isoformat() + "Z"


def normalize_game_id(series: pd.Series) -> pd.Series:
    out = series.astype(str).str.strip()
    out = out.str.replace(r"\.0$", "", regex=True)
    out = out.str.replace(r"\s+", "", regex=True)
    return out.str.zfill(10)
