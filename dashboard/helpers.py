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
@import url('https://fonts.googleapis.com/css2?family=Bebas+Neue&family=Barlow+Condensed:wght@400;600;700&family=DM+Sans:wght@400;500;700&display=swap');

:root {
  --thunder-blue: #007ac1;
  --thunder-orange: #ef3b24;
  --thunder-navy: #002d62;
  --casino-gold: #f6c453;
  --arena-black: #050913;
  --arena-blue: #0d2342;
  --arena-steel: #9ecff4;
  --jumbotron-ice: #d7f1ff;
  --chalk: #eef5ff;
  --ticker-green: #8cffda;
}

[data-testid="stAppViewContainer"] {
  background:
    radial-gradient(circle at 50% -6%, rgba(215,241,255,0.34), rgba(215,241,255,0) 18%),
    radial-gradient(circle at 10% 2%, rgba(246,196,83,0.18), rgba(246,196,83,0) 16%),
    radial-gradient(circle at 90% 4%, rgba(239,59,36,0.18), rgba(239,59,36,0) 18%),
    radial-gradient(circle at 14% 14%, rgba(0,122,193,0.18), rgba(0,122,193,0) 30%),
    radial-gradient(circle at 86% 20%, rgba(0,122,193,0.14), rgba(0,122,193,0) 28%),
    linear-gradient(180deg, #03060c 0%, #091427 28%, #0b2240 58%, #091a31 100%);
  color: var(--chalk);
  font-family: "DM Sans", "Segoe UI", sans-serif;
}

[data-testid="stSidebar"] {
  background:
    linear-gradient(180deg, rgba(7,16,31,0.98), rgba(7,12,22,0.98)),
    linear-gradient(135deg, rgba(0,122,193,0.12), rgba(239,59,36,0.08));
  border-right: 1px solid rgba(158,207,244,0.18);
  box-shadow: inset -1px 0 0 rgba(255,255,255,0.05);
}

[data-testid="stSidebar"] * {
  color: #edf6ff !important;
}

[data-testid="stHeader"] {
  background: rgba(5, 9, 19, 0.42);
  border-bottom: 1px solid rgba(158,207,244,0.18);
}

[data-testid="stToolbar"] {
  right: 1rem;
}

h1, h2, h3 {
  font-family: "Bebas Neue", "Barlow Condensed", sans-serif !important;
  letter-spacing: 0.9px;
}

h1, h2 {
  color: var(--casino-gold);
  text-shadow: 0 2px 16px rgba(246,196,83,0.24);
}

h3 {
  color: var(--jumbotron-ice);
}

p, li, label {
  color: #e8f2ff;
}

[data-testid="stMetric"] {
  position: relative;
  overflow: hidden;
  background:
    linear-gradient(180deg, rgba(255,255,255,0.06), rgba(255,255,255,0) 18%),
    linear-gradient(160deg, rgba(5,15,29,0.94), rgba(6,26,49,0.92));
  border: 1px solid rgba(158,207,244,0.24);
  border-radius: 16px;
  padding: 12px 14px;
  box-shadow:
    0 14px 28px rgba(0,0,0,0.28),
    inset 0 1px 0 rgba(255,255,255,0.04);
}

[data-testid="stMetric"]::before {
  content: "";
  position: absolute;
  inset: 0 auto auto 0;
  width: 100%;
  height: 4px;
  background: linear-gradient(90deg, var(--thunder-orange), var(--casino-gold), var(--thunder-blue));
}

[data-testid="stMetricLabel"],
[data-testid="stMetricValue"],
[data-testid="stMetricDelta"] {
  color: #f6fbff !important;
}

.stDataFrame, [data-testid="stDataFrame"] {
  border: 1px solid rgba(158,207,244,0.22);
  border-radius: 12px;
  overflow: hidden;
  background: rgba(4, 11, 22, 0.66);
  box-shadow: inset 0 0 0 1px rgba(255,255,255,0.04);
}

div[data-testid="stDataFrame"] div[role="grid"] {
  background: rgba(4, 11, 22, 0.72);
}

div[data-testid="stDataFrame"] [role="columnheader"] {
  background: linear-gradient(180deg, rgba(0,122,193,0.28), rgba(0,45,98,0.35));
  color: #f2f9ff;
}

.casino-banner {
  position: relative;
  overflow: hidden;
  border: 1px solid rgba(158,207,244,0.3);
  border-radius: 24px;
  padding: 22px 22px 18px 22px;
  margin: 0 0 16px 0;
  background:
    linear-gradient(180deg, rgba(255,255,255,0.05), rgba(255,255,255,0) 16%),
    linear-gradient(140deg, rgba(5,11,22,0.96), rgba(7,28,53,0.94) 55%, rgba(0,45,98,0.9));
  box-shadow:
    0 18px 40px rgba(0,0,0,0.34),
    inset 0 1px 0 rgba(255,255,255,0.05),
    inset 0 -1px 0 rgba(239,59,36,0.18);
}

.arena-lightbank {
  display: flex;
  justify-content: space-between;
  gap: 14px;
  margin-bottom: 14px;
}

.arena-light {
  height: 10px;
  flex: 1 1 0;
  border-radius: 999px;
  background: linear-gradient(90deg, rgba(255,255,255,0.08), rgba(215,241,255,0.95), rgba(255,255,255,0.08));
  box-shadow: 0 0 16px rgba(215,241,255,0.85);
  opacity: 0.92;
}

.jumbotron-shell {
  position: relative;
  z-index: 1;
  border: 1px solid rgba(158,207,244,0.18);
  border-radius: 18px;
  padding: 16px 18px 14px 18px;
  background:
    linear-gradient(180deg, rgba(255,255,255,0.04), rgba(255,255,255,0) 14%),
    linear-gradient(145deg, rgba(0,0,0,0.28), rgba(0,45,98,0.18));
  box-shadow:
    inset 0 0 0 1px rgba(255,255,255,0.02),
    0 10px 22px rgba(0,0,0,0.18);
}

.casino-kicker {
  font-family: "Barlow Condensed", sans-serif;
  color: var(--arena-steel);
  letter-spacing: 1.8px;
  text-transform: uppercase;
  font-size: 1rem;
}

.casino-title {
  font-family: "Bebas Neue", "Barlow Condensed", sans-serif;
  color: #fff4d1;
  font-size: 3.25rem;
  line-height: 0.96;
  margin-top: 6px;
  text-shadow: 0 0 24px rgba(246,196,83,0.22);
}

.casino-sub {
  color: #e6f2ff;
  margin-top: 8px;
  font-size: 1rem;
  max-width: 880px;
}

.scoreboard-rack {
  display: grid;
  grid-template-columns: repeat(3, minmax(0, 1fr));
  gap: 12px;
  margin-top: 18px;
}

.scoreboard-chip {
  border: 1px solid rgba(158,207,244,0.22);
  border-radius: 14px;
  padding: 12px 14px;
  background:
    linear-gradient(180deg, rgba(255,255,255,0.05), rgba(255,255,255,0) 22%),
    linear-gradient(155deg, rgba(1,13,29,0.94), rgba(11,36,64,0.86));
}

.scoreboard-chip-label {
  font-family: "Barlow Condensed", sans-serif;
  text-transform: uppercase;
  letter-spacing: 1.3px;
  color: #9ecff4;
  font-size: 0.85rem;
}

.scoreboard-chip-value {
  font-family: "Bebas Neue", "Barlow Condensed", sans-serif;
  color: #fff4d1;
  font-size: 1.45rem;
  line-height: 1;
  margin-top: 6px;
}

.odds-ticker-wrap {
  margin-top: 16px;
  padding: 10px 12px;
  overflow: hidden;
  white-space: nowrap;
  border-radius: 999px;
  border: 1px solid rgba(246,196,83,0.26);
  background: linear-gradient(90deg, rgba(239,59,36,0.14), rgba(246,196,83,0.1), rgba(0,122,193,0.14));
}

.odds-ticker {
  display: inline-block;
  min-width: 100%;
  color: var(--ticker-green);
  font-family: "Bebas Neue", "Barlow Condensed", sans-serif;
  letter-spacing: 1.15px;
  animation: odds-scroll 24s linear infinite;
}

@keyframes odds-scroll {
  from { transform: translateX(18%); }
  to { transform: translateX(-100%); }
}

.update-pill {
  display: inline-flex;
  align-items: center;
  gap: 8px;
  margin: 2px 0 16px 0;
  padding: 8px 12px;
  border-radius: 999px;
  border: 1px solid rgba(158,207,244,0.24);
  background: rgba(5, 13, 26, 0.72);
  color: #f1f8ff;
  font-size: 0.88rem;
  box-shadow: 0 6px 18px rgba(0,0,0,0.2);
}

.update-dot {
  width: 8px;
  height: 8px;
  border-radius: 999px;
  background: var(--ticker-green);
  box-shadow: 0 0 12px rgba(140,255,218,0.7);
}

.arena-card-grid {
  display: grid;
  grid-template-columns: repeat(3, minmax(0, 1fr));
  gap: 14px;
  margin: 14px 0 16px 0;
}

.arena-card {
  border-radius: 16px;
  padding: 16px 16px 14px 16px;
  border: 1px solid rgba(158,207,244,0.2);
  background:
    linear-gradient(180deg, rgba(255,255,255,0.04), rgba(255,255,255,0) 16%),
    linear-gradient(150deg, rgba(4,13,25,0.94), rgba(8,29,51,0.9));
  box-shadow: 0 12px 24px rgba(0,0,0,0.22);
}

.arena-card-label {
  font-family: "Bebas Neue", "Barlow Condensed", sans-serif;
  color: #fff0c4;
  font-size: 1.35rem;
  line-height: 1;
}

.arena-card-copy {
  color: #d9eaff;
  margin-top: 8px;
  font-size: 0.94rem;
  line-height: 1.4;
}

@media (max-width: 900px) {
  .casino-title {
    font-size: 2.4rem;
  }

  .scoreboard-rack,
  .arena-card-grid {
    grid-template-columns: 1fr;
  }
}
</style>
"""


def _render_html_block(block: str) -> None:
    if hasattr(st, "html"):
        st.html(block)
    else:
        st.markdown(block, unsafe_allow_html=True)


def apply_casino_theme(page_title: str, subtitle: str) -> None:
    _render_html_block(CASINO_THEME_CSS)
    banner_html = f"""
    <div class="casino-banner">
      <div class="arena-lightbank">
        <div class="arena-light"></div>
        <div class="arena-light"></div>
        <div class="arena-light"></div>
        <div class="arena-light"></div>
        <div class="arena-light"></div>
      </div>
      <div class="jumbotron-shell">
        <div class="casino-kicker">THUNDER PREDICTOR APP | CASINO FLOOR MODE | OKC JUMBOTRON LIVE</div>
        <div class="casino-title">{page_title}</div>
        <div class="casino-sub">{subtitle}</div>
        <div class="scoreboard-rack">
          <div class="scoreboard-chip">
            <div class="scoreboard-chip-label">Arena Feed</div>
            <div class="scoreboard-chip-value">Thunder Territory</div>
          </div>
          <div class="scoreboard-chip">
            <div class="scoreboard-chip-label">Sportsbook Pulse</div>
            <div class="scoreboard-chip-value">Pregame Only</div>
          </div>
          <div class="scoreboard-chip">
            <div class="scoreboard-chip-label">Jumbotron Status</div>
            <div class="scoreboard-chip-value">Lights Hot</div>
          </div>
        </div>
        <div class="odds-ticker-wrap">
          <div class="odds-ticker">
            OKC ARENA LIGHTS LIVE | THUNDER BLUE BOARD | MODEL EDGE ONLY | NO POSTGAME LEAKAGE | JUMBOTRON SIGNAL ACTIVE
          </div>
        </div>
      </div>
    </div>
    """
    _render_html_block(banner_html)


def style_plotly(fig: go.Figure) -> go.Figure:
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(5,18,35,0.72)",
        font={"family": "DM Sans, Barlow Condensed, sans-serif", "color": "#eef5ff"},
        title_font={"family": "Bebas Neue, Barlow Condensed, sans-serif", "size": 28, "color": "#f6c453"},
        legend_font={"family": "DM Sans, sans-serif", "color": "#eef5ff"},
        colorway=["#f6c453", "#4fc3ff", "#ef3b24", "#92f3d6", "#8dbbff", "#ffd88a"],
        margin={"l": 40, "r": 20, "t": 70, "b": 40},
    )
    fig.update_xaxes(
        gridcolor="rgba(255,255,255,0.10)",
        zerolinecolor="rgba(255,255,255,0.10)",
        linecolor="rgba(158,207,244,0.35)",
    )
    fig.update_yaxes(
        gridcolor="rgba(255,255,255,0.10)",
        zerolinecolor="rgba(255,255,255,0.10)",
        linecolor="rgba(158,207,244,0.35)",
    )
    return fig


def render_section_grid(cards: list[tuple[str, str]]) -> None:
    if not cards:
        return

    cards_html = "".join(
        f"""
        <div class="arena-card">
          <div class="arena-card-label">{label}</div>
          <div class="arena-card-copy">{copy}</div>
        </div>
        """
        for label, copy in cards
    )
    _render_html_block(f'<div class="arena-card-grid">{cards_html}</div>')


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
    _render_html_block(
        f"""
        <div class="update-pill">
          <span class="update-dot"></span>
          <span>{label}</span>
        </div>
        """
    )


def format_probability(value: float | int | None) -> str:
    if value is None or pd.isna(value):
        return "N/A"
    return f"{float(value):.1%}"


def format_confidence(value: float | int | None) -> str:
    if value is None or pd.isna(value):
        return "N/A"
    return f"{float(value):.0%}"


def matchup_label(home_team: str | None, away_team: str | None) -> str:
    home = home_team or "HOME"
    away = away_team or "AWAY"
    return f"{away} @ {home}"


def probability_edge(home_prob: float | int | None) -> float | None:
    if home_prob is None or pd.isna(home_prob):
        return None
    return abs(float(home_prob) - 0.5) * 2


def confidence_band(home_prob: float | int | None) -> str:
    edge = probability_edge(home_prob)
    if edge is None:
        return "N/A"
    if edge >= 0.60:
        return "Strong edge"
    if edge >= 0.30:
        return "Lean"
    return "Toss-up"


def pick_label(home_prob: float | int | None, home_team: str | None, away_team: str | None) -> str:
    if home_prob is None or pd.isna(home_prob):
        return "N/A"
    winner = home_team if float(home_prob) >= 0.5 else away_team
    return winner or "N/A"


def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def read_json(path: Path) -> dict:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def load_optional_csv(path: Path, parse_dates: list[str] | None = None) -> pd.DataFrame:
    df = read_csv(path)
    if df.empty:
        return df

    for col in parse_dates or []:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")
    return df


def load_optional_json(path: Path) -> dict:
    return read_json(path)


def load_model_maintenance_artifacts() -> dict[str, object]:
    return {
        "summary": load_optional_json(REPORTS / "model_maintenance_summary.json"),
        "windows": load_optional_csv(
            REPORTS / "model_maintenance_windows.csv",
            parse_dates=[
                "GAME_DATE",
                "WINDOW_START",
                "WINDOW_END",
                "window_start",
                "window_end",
                "START_DATE",
                "END_DATE",
            ],
        ),
        "segments": load_optional_csv(REPORTS / "model_maintenance_segments.csv"),
        "confidence_buckets": load_optional_csv(REPORTS / "model_maintenance_confidence_buckets.csv"),
    }


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
