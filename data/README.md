# Data Layout

- `raw/` contains fetched source files used for reproducibility and API fallback behavior.
- `processed/cleaned/` contains cleaned analytical tables.
- `processed/features/` contains engineered feature tables.
- `processed/predictions/` contains current and archived prediction outputs used by the dashboard.
- `processed/artifacts/` contains auxiliary metadata such as feature lists and split metadata.

`nba.sqlite` remains at the top of `data/` as the local analytical store.
