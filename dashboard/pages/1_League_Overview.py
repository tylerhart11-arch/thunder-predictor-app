from __future__ import annotations

import plotly.express as px
import streamlit as st

from dashboard.helpers import apply_casino_theme, load_clean_games, style_plotly


apply_casino_theme(
    page_title="League Overview",
    subtitle="Conference table heat checks, volume trends, and recent final scores.",
)
games = load_clean_games()

if games.empty:
    st.warning("No cleaned game data found. Run a full build first.")
    st.stop()

games["HOME_WIN"] = games["HOME_WIN"].astype(int)
games["AWAY_WIN"] = 1 - games["HOME_WIN"]

# Build standings table directly from game rows.
standings_home = games.groupby("HOME_TEAM_ABBREVIATION", as_index=False).agg(
    games_home=("GAME_ID", "count"),
    wins_home=("HOME_WIN", "sum"),
)
standings_away = games.groupby("AWAY_TEAM_ABBREVIATION", as_index=False).agg(
    games_away=("GAME_ID", "count"),
    wins_away=("AWAY_WIN", "sum"),
)
standings = standings_home.merge(
    standings_away,
    left_on="HOME_TEAM_ABBREVIATION",
    right_on="AWAY_TEAM_ABBREVIATION",
    how="outer",
).fillna(0)
standings["TEAM"] = standings["HOME_TEAM_ABBREVIATION"].fillna(standings["AWAY_TEAM_ABBREVIATION"])
standings["GAMES"] = standings["games_home"] + standings["games_away"]
standings["WINS"] = standings["wins_home"] + standings["wins_away"]
standings["LOSSES"] = standings["GAMES"] - standings["WINS"]
standings["WIN_PCT"] = standings["WINS"] / standings["GAMES"].replace(0, 1)
standings = standings[["TEAM", "GAMES", "WINS", "LOSSES", "WIN_PCT"]].sort_values("WIN_PCT", ascending=False)

st.subheader("Current Standings Snapshot (From Ingested Data)")
st.dataframe(standings, use_container_width=True)

daily_volume = (
    games.assign(GAME_DAY=games["GAME_DATE"].dt.date)
    .groupby("GAME_DAY", as_index=False)
    .agg(games=("GAME_ID", "count"))
    .rename(columns={"GAME_DAY": "GAME_DATE"})
)
fig = px.line(daily_volume, x="GAME_DATE", y="games", title="Games By Date")
st.plotly_chart(style_plotly(fig), use_container_width=True)

st.subheader("Recent Games")
recent = games.sort_values("GAME_DATE", ascending=False).head(25)
st.dataframe(
    recent[
        [
            "GAME_DATE",
            "HOME_TEAM_ABBREVIATION",
            "AWAY_TEAM_ABBREVIATION",
            "HOME_PTS",
            "AWAY_PTS",
            "HOME_WIN",
        ]
    ],
    use_container_width=True,
)
