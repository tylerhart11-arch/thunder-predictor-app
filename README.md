# NBA Thunder Predictor

Pregame NBA win-probability model with a Streamlit dashboard focused on the Oklahoma City Thunder.

The project:
- pulls NBA game data from `nba_api`,
- builds pregame-only features,
- trains a baseline logistic model and an improved boosted model,
- stores predictions and results locally,
- tracks Thunder predictions versus actual outcomes over time.

## What It Does

- `Predicts NBA games before tipoff`
- `Trains on league-wide NBA data`
- `Tracks Thunder-specific prediction accuracy and confidence`
- `Updates results as games finish`
- `Serves a local Streamlit dashboard`

## Repo Layout

```text
Playground/
  config/
    config.yaml
  dashboard/
    helpers.py
  data/
    raw/
    cleaned/
    features/
    predictions/
    artifacts/
    nba.sqlite
  logs/
  models/
  pages/
    1_League_Overview.py
    2_Model_Performance.py
    3_Thunder_Tracker.py
    4_Upcoming_Predictions.py
    5_Diagnostics.py
  reports/
    diagnostics/
  scripts/
    daily_update.py
    full_build.py
    run_dashboard.py
  src/
    calibration.py
    cli.py
    config.py
    data_cleaning.py
    data_ingestion.py
    data_quality.py
    evaluate.py
    feature_engineering.py
    leakage_checks.py
    logger.py
    models_baseline.py
    models_xgb.py
    predict.py
    split.py
    thunder_report.py
    update_pipeline.py
    utils.py
  .env.template
  requirements.txt
  README.md
  streamlit_app.py
```

## Key Directories

- `src/`: core pipeline logic
- `scripts/`: simple entrypoints to run the project
- `dashboard/`: shared dashboard helpers and presentation code
- `pages/`: Streamlit multipage app screens
- `config/`: project settings
- `data/`: local raw, cleaned, feature, and prediction data
- `models/`: saved trained models
- `reports/`: metrics, diagnostics, and Thunder reporting outputs

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
python -m src.cli full-build
python -m src.cli daily-update
python -m src.cli run-dashboard --dashboard-port 8512
```

## Automated Refresh

The deployed app can now refresh itself from GitHub Actions.

- Workflow file: `.github/workflows/refresh_thunder_data.yml`
- Trigger types:
  - manual run from the GitHub `Actions` tab
  - scheduled runs throughout the overnight/morning window

What it does:
- installs dependencies
- runs `python scripts/daily_update.py`
- commits updated dashboard artifacts back to `main`

Because Streamlit Cloud is connected to the repo, a successful artifact push will trigger an app redeploy with fresh data.

Schedule note:
- GitHub Actions schedules run in UTC
- this workflow is set for `05:15`, `07:15`, `09:15`, and `13:15` UTC each day
- in Central time, that targets overnight postgame and morning refresh windows

## Configuration

Main settings live in `config/config.yaml`.

Most likely edits:
- `project.thunder_team_abbr`
- `data.historical_start_season`
- `data.scoreboard_days_back`
- `data.scoreboard_days_forward`
- `update.retrain_every_days`
- `update.min_new_games_for_retrain`

## Modeling Notes

- Target: `home_win`
- Split design: chronological, with current season held out for test reporting
- Leakage controls:
  - no random shuffle,
  - rolling features use prior games only,
  - upcoming predictions use as-of snapshots,
  - postgame stats are excluded from pregame features

## Output Files

Main outputs used by the dashboard:
- `data/predictions/latest_upcoming_predictions.csv`
- `data/predictions/prediction_archive.csv`
- `reports/metrics_latest.json`
- `reports/thunder_summary.json`
- `reports/thunder_predictions_completed.csv`
- `reports/thunder_weekly_summary.csv`

## Notes

- No notebooks are currently required for the working project.
- The dashboard depends on local artifacts in `data/`, `models/`, and `reports/`.
- If NBA API calls fail temporarily, the update pipeline now falls back to cached local raw data when available.
