from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
REPORTS = ROOT / "reports"


CASINO_THEME_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Bebas+Neue&family=DM+Sans:wght@400;500;700&display=swap');

:root {
  --felt-green: #0f5a37;
  --felt-green-dark: #0a3f27;
  --table-edge: #3a2718;
  --chip-gold: #f3c769;
  --chip-red: #d3392f;
  --chalk: #f7f3e8;
  --line-glow: #7be0a8;
}

[data-testid="stAppViewContainer"] {
  background:
    radial-gradient(circle at 15% 15%, rgba(255,255,255,0.08) 0, rgba(255,255,255,0) 35%),
    radial-gradient(circle at 85% 85%, rgba(255,255,255,0.06) 0, rgba(255,255,255,0) 45%),
    linear-gradient(130deg, var(--felt-green-dark), var(--felt-green));
  color: var(--chalk);
  font-family: "DM Sans", "Trebuchet MS", sans-serif;
}

[data-testid="stSidebar"] {
  background:
    linear-gradient(180deg, rgba(58,39,24,0.96), rgba(33,21,13,0.95)),
    repeating-linear-gradient(
      45deg,
      rgba(255,255,255,0.03) 0,
      rgba(255,255,255,0.03) 2px,
      rgba(0,0,0,0) 2px,
      rgba(0,0,0,0) 8px
    );
  border-right: 2px solid rgba(243,199,105,0.45);
}

[data-testid="stSidebar"] * {
  color: #f4ede0 !important;
}

[data-testid="stHeader"] {
  background: rgba(10, 26, 18, 0.45);
  border-bottom: 1px solid rgba(243,199,105,0.35);
}

h1, h2, h3 {
  font-family: "Bebas Neue", "Impact", sans-serif !important;
  letter-spacing: 0.8px;
}

h1, h2 {
  color: var(--chip-gold);
  text-shadow: 0 2px 0 rgba(0,0,0,0.25);
}

[data-testid="stMetric"] {
  background: linear-gradient(170deg, rgba(32, 22, 14, 0.82), rgba(21, 14, 9, 0.85));
  border: 1px solid rgba(243,199,105,0.38);
  border-radius: 14px;
  padding: 10px 14px;
  box-shadow: 0 8px 24px rgba(0,0,0,0.25);
}

[data-testid="stMetricLabel"],
[data-testid="stMetricValue"] {
  color: #fdf8ea !important;
}

.stDataFrame, [data-testid="stDataFrame"] {
  border: 1px solid rgba(243,199,105,0.35);
  border-radius: 10px;
  overflow: hidden;
  box-shadow: inset 0 0 0 1px rgba(255,255,255,0.05);
}

.casino-banner {
  border: 2px solid rgba(243,199,105,0.45);
  border-radius: 14px;
  padding: 14px 18px;
  margin: 0 0 14px 0;
  background:
    radial-gradient(circle at 20% 35%, rgba(123,224,168,0.16), rgba(123,224,168,0) 45%),
    linear-gradient(160deg, rgba(30, 20, 12, 0.92), rgba(16, 10, 7, 0.92));
  box-shadow: 0 8px 26px rgba(0,0,0,0.28);
}

.casino-kicker {
  font-family: "Bebas Neue", "Impact", sans-serif;
  color: var(--line-glow);
  letter-spacing: 1.3px;
  font-size: 1.05rem;
}

.casino-title {
  font-family: "Bebas Neue", "Impact", sans-serif;
  color: var(--chip-gold);
  font-size: 2.05rem;
  line-height: 1.02;
  margin-top: 2px;
}

.casino-sub {
  color: #f3ead6;
  margin-top: 5px;
  font-size: 0.95rem;
}

.odds-ticker-wrap {
  margin-top: 10px;
  border-top: 1px dashed rgba(243,199,105,0.4);
  padding-top: 7px;
  overflow: hidden;
  white-space: nowrap;
}

.odds-ticker {
  display: inline-block;
  min-width: 100%;
  color: #f7d691;
  font-family: "Bebas Neue", "Impact", sans-serif;
  letter-spacing: 0.9px;
  animation: odds-scroll 22s linear infinite;
}

@keyframes odds-scroll {
  from { transform: translateX(20%); }
  to { transform: translateX(-100%); }
}

.update-pill {
  display: inline-flex;
  align-items: center;
  gap: 8px;
  margin: 2px 0 16px 0;
  padding: 7px 12px;
  border-radius: 999px;
  border: 1px solid rgba(243,199,105,0.35);
  background: rgba(20, 14, 9, 0.72);
  color: #f6efdf;
  font-size: 0.88rem;
  box-shadow: 0 4px 16px rgba(0,0,0,0.18);
}

.update-dot {
  width: 8px;
  height: 8px;
  border-radius: 999px;
  background: var(--line-glow);
  box-shadow: 0 0 10px rgba(123,224,168,0.65);
}
</style>
"""


def apply_casino_theme(page_title: str, subtitle: str) -> None:
    st.markdown(CASINO_THEME_CSS, unsafe_allow_html=True)
    banner_html = f"""
    <div class="casino-banner">
      <div class="casino-kicker">THUNDER SPORTSBOOK LAB</div>
      <div class="casino-title">{page_title}</div>
      <div class="casino-sub">{subtitle}</div>
      <div class="odds-ticker-wrap">
        <div class="odds-ticker">
          HOME EDGE +3.5 | MODEL CONFIDENCE INDEX | NO PARLAYS, JUST PROBABILITY | OKC WATCH MODE ACTIVE
        </div>
      </div>
    </div>
    """
    st.markdown(banner_html, unsafe_allow_html=True)


def style_plotly(fig: go.Figure) -> go.Figure:
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(13,52,33,0.55)",
        font={"family": "DM Sans, Trebuchet MS, sans-serif", "color": "#f7f3e8"},
        title_font={"family": "Bebas Neue, Impact, sans-serif", "size": 28, "color": "#f3c769"},
        legend_font={"family": "DM Sans, Trebuchet MS, sans-serif", "color": "#f7f3e8"},
        margin={"l": 40, "r": 20, "t": 70, "b": 40},
    )
    fig.update_xaxes(
        gridcolor="rgba(255,255,255,0.12)",
        zerolinecolor="rgba(255,255,255,0.12)",
        linecolor="rgba(243,199,105,0.45)",
    )
    fig.update_yaxes(
        gridcolor="rgba(255,255,255,0.12)",
        zerolinecolor="rgba(255,255,255,0.12)",
        linecolor="rgba(243,199,105,0.45)",
    )
    return fig


def latest_update_timestamp() -> str | None:
    candidates = [
        REPORTS / "metrics_latest.json",
        DATA / "predictions" / "latest_upcoming_predictions.csv",
        DATA / "predictions" / "prediction_archive.csv",
    ]
    existing = [path for path in candidates if path.exists()]
    if not existing:
        return None

    latest = max(path.stat().st_mtime for path in existing)
    return datetime.fromtimestamp(latest).strftime("%b %d, %Y %I:%M %p")


def render_update_pill(label: str) -> None:
    st.markdown(
        f"""
        <div class="update-pill">
          <span class="update-dot"></span>
          <span>{label}</span>
        </div>
        """,
        unsafe_allow_html=True,
    )


def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def read_json(path: Path) -> dict:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def load_clean_games() -> pd.DataFrame:
    df = read_csv(DATA / "cleaned" / "games_clean.csv")
    if not df.empty and "GAME_DATE" in df.columns:
        df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"])
    return df


def load_archive() -> pd.DataFrame:
    df = read_csv(DATA / "predictions" / "prediction_archive.csv")
    if not df.empty and "GAME_DATE" in df.columns:
        df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"])
    return df


def load_upcoming() -> pd.DataFrame:
    df = read_csv(DATA / "predictions" / "latest_upcoming_predictions.csv")
    if not df.empty and "GAME_DATE" in df.columns:
        df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"])
    return df
