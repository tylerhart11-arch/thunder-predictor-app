# NBA Thunder Predictor

Pregame NBA win-probability model with a Streamlit dashboard focused on the Oklahoma City Thunder.

The repo is now organized to match a cleaner production-minded workflow:
- `app/` holds reusable application logic
- `pipelines/` holds orchestration code
- `data/raw/` stores fetched source data
- `data/processed/` stores derived datasets and dashboard artifacts
- `scripts/` exposes simple entrypoints
- `docs/`, `tests/`, `notebooks/`, and `prompts/` are present for maintainability

## What It Does

- Predicts NBA games before tipoff
- Trains on league-wide NBA data
- Tracks Thunder-specific prediction accuracy and confidence
- Updates results as games finish
- Serves a local Streamlit dashboard

## Repo Layout

```text
Playground/
  app/
    cli.py
    config.py
    data_ingestion.py
    data_cleaning.py
    feature_engineering.py
    predict.py
    thunder_report.py
    ...
  config/
    config.yaml
  dashboard/
    app.py
    helpers.py
  data/
    raw/
    processed/
      artifacts/
      cleaned/
      features/
      predictions/
    nba.sqlite
  docs/
    repo_structure.md
  models/
  notebooks/
    README.md
  pages/
    1_League_Overview.py
    2_Model_Performance.py
    3_Thunder_Tracker.py
    4_Upcoming_Predictions.py
    5_Diagnostics.py
  pipelines/
    update_pipeline.py
  prompts/
    README.md
  reports/
    diagnostics/
  scripts/
    daily_update.py
    full_build.py
    run_dashboard.py
  tests/
    test_repo_layout.py
  .env.template
  README.md
  requirements.txt
  streamlit_app.py
```

## Setup

```powershell
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
```

Optional:

```powershell
copy .env.template .env
```

## Run

Full historical rebuild:

```powershell
python scripts\full_build.py
```

Daily refresh:

```powershell
python scripts\daily_update.py
```

Launch dashboard:

```powershell
python scripts\run_dashboard.py
```

Equivalent CLI usage:

```powershell
python -m app.cli full-build
python -m app.cli daily-update
python -m app.cli run-dashboard --dashboard-port 8512
```

Run the smoke tests:

```powershell
python -m unittest discover -s tests
```

## Configuration

Main settings live in `config/config.yaml`.

Most likely edits:
- `project.thunder_team_abbr`
- `data.historical_start_season`
- `data.scoreboard_days_back`
- `data.scoreboard_days_forward`
- `update.retrain_every_days`
- `update.min_new_games_for_retrain`

## Data Layout

- Raw pulls land in `data/raw/`
- Derived datasets land in `data/processed/cleaned/` and `data/processed/features/`
- Dashboard-ready outputs land in `data/processed/predictions/` and `reports/`
- Model binaries stay in `models/`

Main dashboard artifacts:
- `data/processed/predictions/latest_upcoming_predictions.csv`
- `data/processed/predictions/prediction_archive.csv`
- `reports/metrics_latest.json`
- `reports/thunder_summary.json`
- `reports/thunder_predictions_completed.csv`
- `reports/thunder_weekly_summary.csv`

## Automated Refresh

The deployed app refreshes itself from GitHub Actions via `.github/workflows/refresh_thunder_data.yml`.

The workflow:
- installs dependencies
- runs `python scripts/daily_update.py`
- commits refreshed raw and processed artifacts back to the repo

GitHub schedules run in UTC. The current schedule is `05:15`, `07:15`, `09:15`, and `13:15` UTC, which targets overnight and morning refresh windows in Central time.

## Notes

- `pages/` remains at the repo root because Streamlit multipage apps expect it beside the main entrypoint.
- `notebooks/` is reserved for exploration only; production logic belongs in `app/`, `pipelines/`, and `scripts/`.
- If NBA API calls fail temporarily, the update pipeline falls back to cached local raw data when available.
