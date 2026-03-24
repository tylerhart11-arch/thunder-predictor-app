# Repo Structure

This repo follows a production-minded layout:

- `app/` contains reusable application logic.
- `pipelines/` contains orchestration code that wires ingestion, training, scoring, and reporting together.
- `scripts/` contains thin entrypoints for local runs.
- `data/raw/` stores fetched source data.
- `data/processed/` stores derived datasets and dashboard-ready outputs.
- `models/` stores trained model artifacts.
- `reports/` stores analyst-facing outputs and diagnostics.
- `notebooks/` is for exploration only.
- `tests/` holds lightweight regression and layout checks.
- `prompts/` is reserved for reusable analyst or automation prompts.

The only intentional exception is root-level `pages/`, which stays in place because Streamlit expects multipage files to live beside the app entrypoint.
