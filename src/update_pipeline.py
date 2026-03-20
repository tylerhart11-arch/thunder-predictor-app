from __future__ import annotations

import json
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from src.calibration import fit_cv_calibrator, fit_validation_calibrator
from src.data_cleaning import build_game_level_table, clean_league_logs, merge_actuals_from_scoreboard
from src.data_ingestion import NBADataIngestion
from src.data_quality import basic_data_quality_report
from src.evaluate import (
    classification_metrics,
    reliability_curve_df,
    save_confusion_matrix_plot,
    save_reliability_plot,
)
from src.feature_engineering import build_team_feature_table, build_training_features, build_upcoming_game_features
from src.leakage_checks import run_leakage_checks
from src.models_baseline import logistic_feature_importance, predict_proba as baseline_predict_proba, train_baseline_logistic
from src.models_xgb import (
    fit_improved_model_with_params,
    improved_feature_importance,
    tune_and_train_improved_model,
)
from src.predict import initialize_prediction_archive, predict_dataframe, reconcile_archive_with_actuals
from src.split import season_holdout_split, split_summary_table
from src.thunder_report import add_rolling_accuracy, build_thunder_summary, thunder_predictions_only, weekly_performance
from src.utils import load_csv_if_exists, now_utc_iso, save_csv, save_json, to_sqlite


class NBAPipeline:
    def __init__(self, cfg: dict[str, Any], paths, logger):
        self.cfg = cfg
        self.paths = paths
        self.logger = logger
        self.ingestor = NBADataIngestion(cfg=cfg, logger=logger)

        self.raw_league_path = self.paths.raw_dir / "league_game_log_raw.csv"
        self.raw_scoreboard_path = self.paths.raw_dir / "scoreboard_raw.csv"
        self.clean_team_path = self.paths.cleaned_dir / "team_games_clean.csv"
        self.clean_games_path = self.paths.cleaned_dir / "games_clean.csv"
        self.team_features_path = self.paths.features_dir / "team_features.csv"
        self.training_features_path = self.paths.features_dir / "training_features.csv"
        self.upcoming_features_path = self.paths.features_dir / "upcoming_features.csv"
        self.archive_path = self.paths.predictions_dir / "prediction_archive.csv"
        self.latest_upcoming_path = self.paths.predictions_dir / "latest_upcoming_predictions.csv"
        self.metrics_path = self.paths.reports_dir / "metrics_latest.json"
        self.leakage_path = self.paths.reports_dir / "leakage_report.json"
        self.dq_path = self.paths.reports_dir / "data_quality.json"
        self.model_meta_path = self.paths.models_dir / "model_metadata.json"
        self.feature_cols_path = self.paths.artifacts_dir / "feature_cols.json"
        self.split_meta_path = self.paths.artifacts_dir / "split_meta.json"
        self.split_summary_path = self.paths.reports_dir / "split_dataset_summary.csv"
        self.baseline_model_path = self.paths.models_dir / "baseline_logreg.joblib"
        self.improved_model_path = self.paths.models_dir / "xgb_model.joblib"
        self.calibrator_path = self.paths.models_dir / "calibrator.joblib"

    def run_full_build(self) -> None:
        self.logger.info("=== FULL BUILD START ===")
        self._run_pipeline(full_history=True, force_retrain=True)
        self.logger.info("=== FULL BUILD END ===")

    def run_daily_update(self) -> None:
        self.logger.info("=== DAILY UPDATE START ===")
        self._run_pipeline(full_history=False, force_retrain=False)
        self.logger.info("=== DAILY UPDATE END ===")

    def _run_pipeline(self, full_history: bool, force_retrain: bool) -> None:
        league_raw, scoreboard_raw = self._ingest_and_store(full_history=full_history)
        team_games, game_level = self._clean_and_store(league_raw, scoreboard_raw)
        team_features, training_df, feature_cols = self._feature_build_and_store(team_games, game_level)
        self._save_quality_reports(team_games, game_level, training_df, feature_cols)

        should_retrain = self._should_retrain(force_retrain=force_retrain, n_games=len(training_df))
        if should_retrain:
            model_bundle = self._train_and_store_models(training_df, feature_cols)
        else:
            model_bundle = self._load_models()

        upcoming_predictions = self._score_upcoming_games(
            scoreboard_raw=scoreboard_raw,
            team_features=team_features,
            feature_cols=feature_cols,
            model=model_bundle["model"],
            calibrator=model_bundle["calibrator"],
        )
        archive = self._update_prediction_archive(upcoming_predictions, game_level)
        self._build_thunder_outputs(archive)

    def _ingest_and_store(self, full_history: bool) -> tuple[pd.DataFrame, pd.DataFrame]:
        old_league = load_csv_if_exists(self.raw_league_path)
        try:
            new_league = self.ingestor.fetch_historical_league_logs(full_history=full_history)
        except Exception as exc:  # noqa: BLE001
            if old_league.empty:
                raise RuntimeError("Historical league log refresh failed and no cached raw league data exists.") from exc
            self.logger.warning(
                "Historical league log refresh failed; falling back to cached raw league data: %s",
                exc,
            )
            new_league = pd.DataFrame()

        league_raw = pd.concat([old_league, new_league], ignore_index=True) if not old_league.empty else new_league
        if not league_raw.empty:
            league_raw["GAME_ID"] = league_raw["GAME_ID"].astype(str)
            league_raw = league_raw.drop_duplicates(subset=["GAME_ID", "TEAM_ID", "GAME_DATE"]).reset_index(drop=True)

        old_scoreboard = load_csv_if_exists(self.raw_scoreboard_path)
        try:
            new_scoreboard = self.ingestor.fetch_scoreboard_window(center_date=date.today())
        except Exception as exc:  # noqa: BLE001
            if old_scoreboard.empty:
                raise RuntimeError("Scoreboard refresh failed and no cached scoreboard data exists.") from exc
            self.logger.warning(
                "Scoreboard refresh failed; falling back to cached scoreboard data: %s",
                exc,
            )
            new_scoreboard = pd.DataFrame()

        scoreboard_raw = (
            pd.concat([old_scoreboard, new_scoreboard], ignore_index=True) if not old_scoreboard.empty else new_scoreboard
        )
        if not scoreboard_raw.empty:
            scoreboard_raw["GAME_ID"] = scoreboard_raw["GAME_ID"].astype(str)
            scoreboard_raw["GAME_DATE"] = pd.to_datetime(scoreboard_raw["GAME_DATE"])
            scoreboard_raw = scoreboard_raw.sort_values(["GAME_DATE", "GAME_ID"]).drop_duplicates(
                subset=["GAME_ID"], keep="last"
            )

        save_csv(league_raw, self.raw_league_path)
        save_csv(scoreboard_raw, self.raw_scoreboard_path)
        to_sqlite(league_raw, "raw_league_game_log", self.paths.sqlite_path, if_exists="replace")
        to_sqlite(scoreboard_raw, "raw_scoreboard", self.paths.sqlite_path, if_exists="replace")
        return league_raw, scoreboard_raw

    def _clean_and_store(self, league_raw: pd.DataFrame, scoreboard_raw: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        team_games = clean_league_logs(league_raw)
        game_level = build_game_level_table(team_games)
        game_level = merge_actuals_from_scoreboard(game_level, scoreboard_raw)

        save_csv(team_games, self.clean_team_path)
        save_csv(game_level, self.clean_games_path)
        to_sqlite(team_games, "clean_team_games", self.paths.sqlite_path, if_exists="replace")
        to_sqlite(game_level, "clean_games", self.paths.sqlite_path, if_exists="replace")
        return team_games, game_level

    def _feature_build_and_store(
        self,
        team_games: pd.DataFrame,
        game_level: pd.DataFrame,
    ) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
        min_games = int(self.cfg["data"]["min_games_for_features"])
        team_features = build_team_feature_table(team_games, min_games_for_features=min_games)
        train_res = build_training_features(game_level, team_features)
        training_df = train_res.features_df
        feature_cols = train_res.feature_cols

        save_csv(team_features, self.team_features_path)
        save_csv(training_df, self.training_features_path)
        to_sqlite(team_features, "team_features", self.paths.sqlite_path, if_exists="replace")
        to_sqlite(training_df, "training_features", self.paths.sqlite_path, if_exists="replace")
        save_json({"feature_cols": feature_cols}, self.feature_cols_path)
        return team_features, training_df, feature_cols

    def _save_quality_reports(
        self,
        team_games: pd.DataFrame,
        game_level: pd.DataFrame,
        training_df: pd.DataFrame,
        feature_cols: list[str],
    ) -> None:
        dq_report = {
            "team_games": basic_data_quality_report(team_games),
            "game_level": basic_data_quality_report(game_level),
            "training_features": basic_data_quality_report(training_df),
        }
        leak_report = run_leakage_checks(training_df, feature_cols=feature_cols, target_col="home_win")
        save_json(dq_report, self.dq_path)
        save_json(leak_report, self.leakage_path)

    def _should_retrain(self, force_retrain: bool, n_games: int) -> bool:
        if force_retrain:
            return True
        if not self.improved_model_path.exists() or not self.calibrator_path.exists() or not self.model_meta_path.exists():
            return True

        meta = json.loads(self.model_meta_path.read_text(encoding="utf-8"))
        last_retrain = pd.to_datetime(meta.get("last_retrain_utc", None), utc=True)
        prev_games = int(meta.get("n_games_trained", 0))
        retrain_days = int(self.cfg["update"]["retrain_every_days"])
        min_new_games = int(self.cfg["update"]["min_new_games_for_retrain"])

        days_since = (datetime.now(timezone.utc) - last_retrain).days if pd.notna(last_retrain) else 999
        new_games = max(0, n_games - prev_games)
        should = days_since >= retrain_days or new_games >= min_new_games
        self.logger.info(
            "Retrain check: days_since=%s, new_games=%s, thresholds=(%s days, %s games), should_retrain=%s",
            days_since,
            new_games,
            retrain_days,
            min_new_games,
            should,
        )
        return should

    def _train_and_store_models(self, training_df: pd.DataFrame, feature_cols: list[str]) -> dict[str, Any]:
        if training_df.empty:
            raise RuntimeError("Training feature table is empty. Cannot train models.")

        split = season_holdout_split(
            training_df,
            date_col="GAME_DATE",
            season_col="SEASON",
        )
        split_summary = split_summary_table(split, date_col="GAME_DATE", season_col="SEASON")
        save_csv(split_summary, self.split_summary_path)
        save_json(
            {
                "split_type": split.split_type,
                "train_end_date": str(split.train_end_date.date()) if pd.notna(split.train_end_date) else None,
                "validation_end_date": str(split.valid_end_date.date()) if pd.notna(split.valid_end_date) else None,
                "train_rows": int(len(split.train_df)),
                "validation_rows": int(len(split.valid_df)),
                "test_rows": int(len(split.test_df)),
                "train_seasons": split.train_seasons or [],
                "validation_seasons": split.valid_seasons or [],
                "test_seasons": split.test_seasons or [],
            },
            self.split_meta_path,
        )

        target_col = self.cfg["model"]["target_col"]
        train_df, valid_df, test_df = split.train_df, split.valid_df, split.test_df
        if train_df.empty or valid_df.empty or test_df.empty:
            raise RuntimeError(
                f"Season holdout split produced empty segment(s): train={len(train_df)}, "
                f"valid={len(valid_df)}, test={len(test_df)}; "
                f"seasons train={split.train_seasons}, valid={split.valid_seasons}, test={split.test_seasons}"
            )

        baseline = train_baseline_logistic(train_df, feature_cols, target_col, self.cfg)
        baseline_test_prob = baseline_predict_proba(baseline, test_df, feature_cols)
        baseline_metrics = classification_metrics(test_df[target_col], baseline_test_prob)
        baseline_imp = logistic_feature_importance(baseline, feature_cols)
        save_csv(baseline_imp, self.paths.reports_dir / "baseline_feature_importance.csv")
        joblib.dump(baseline, self.baseline_model_path)

        improved_train_model, best_params, tuning_df = tune_and_train_improved_model(
            train_df=train_df,
            valid_df=valid_df,
            feature_cols=feature_cols,
            target_col=target_col,
            cfg=self.cfg,
            logger=self.logger,
        )
        save_csv(tuning_df, self.paths.reports_dir / "improved_tuning_results.csv")

        X_valid = valid_df[feature_cols]
        y_valid = valid_df[target_col].astype(int)
        X_test = test_df[feature_cols]
        y_test = test_df[target_col].astype(int)

        improved_uncal_prob = improved_train_model.predict_proba(X_test)[:, 1]
        improved_uncal_metrics = classification_metrics(y_test, improved_uncal_prob)

        val_calibrator = fit_validation_calibrator(improved_train_model, X_valid, y_valid, method="isotonic")
        improved_cal_prob = val_calibrator.predict_proba(X_test)[:, 1]
        improved_cal_metrics = classification_metrics(y_test, improved_cal_prob)

        rel_df = reliability_curve_df(y_test, improved_cal_prob, bins=10)
        save_csv(rel_df, self.paths.reports_dir / "diagnostics" / "reliability_curve_test.csv")
        save_reliability_plot(
            rel_df,
            self.paths.reports_dir / "diagnostics" / "reliability_curve_test.png",
            "Improved Model Reliability (Test)",
        )
        save_confusion_matrix_plot(
            improved_cal_metrics,
            self.paths.reports_dir / "diagnostics" / "confusion_matrix_test.png",
            "Improved Model Confusion Matrix (Test)",
        )

        imp_df = improved_feature_importance(improved_train_model, feature_cols)
        save_csv(imp_df, self.paths.reports_dir / "improved_feature_importance.csv")

        train_valid_df = pd.concat([train_df, valid_df], ignore_index=True).sort_values("GAME_DATE")
        final_model = fit_improved_model_with_params(
            train_df=train_valid_df,
            feature_cols=feature_cols,
            target_col=target_col,
            params=best_params,
            cfg=self.cfg,
        )
        final_calibrator = fit_cv_calibrator(
            base_model=final_model,
            X_train=train_valid_df[feature_cols],
            y_train=train_valid_df[target_col].astype(int),
            method="isotonic",
            cv=3,
        )
        joblib.dump(final_model, self.improved_model_path)
        joblib.dump(final_calibrator, self.calibrator_path)

        model_metrics = {
            "split": {
                "split_type": split.split_type,
                "train_end_date": str(split.train_end_date.date()) if pd.notna(split.train_end_date) else None,
                "validation_end_date": str(split.valid_end_date.date()) if pd.notna(split.valid_end_date) else None,
                "train_seasons": split.train_seasons or [],
                "validation_seasons": split.valid_seasons or [],
                "test_seasons": split.test_seasons or [],
            },
            "baseline_test": baseline_metrics,
            "improved_test_uncalibrated": improved_uncal_metrics,
            "improved_test_calibrated": improved_cal_metrics,
            "best_params": best_params,
            "generated_at_utc": now_utc_iso(),
        }
        save_json(model_metrics, self.metrics_path)
        save_json(
            {
                "last_retrain_utc": now_utc_iso(),
                "n_games_trained": int(len(training_df)),
                "best_params": best_params,
            },
            self.model_meta_path,
        )
        return {"model": final_model, "calibrator": final_calibrator}

    def _load_models(self) -> dict[str, Any]:
        model = joblib.load(self.improved_model_path)
        calibrator = joblib.load(self.calibrator_path)
        return {"model": model, "calibrator": calibrator}

    def _score_upcoming_games(
        self,
        scoreboard_raw: pd.DataFrame,
        team_features: pd.DataFrame,
        feature_cols: list[str],
        model,
        calibrator,
    ) -> pd.DataFrame:
        if scoreboard_raw.empty:
            save_csv(pd.DataFrame(), self.latest_upcoming_path)
            return pd.DataFrame()

        schedule = scoreboard_raw.copy()
        schedule["GAME_DATE"] = pd.to_datetime(schedule["GAME_DATE"])
        upcoming = schedule[~schedule["IS_FINAL"] & (schedule["GAME_DATE"].dt.date >= date.today())].copy()
        upcoming = upcoming.sort_values(["GAME_DATE", "GAME_ID"]).drop_duplicates(subset=["GAME_ID"])
        if upcoming.empty:
            save_csv(pd.DataFrame(), self.latest_upcoming_path)
            return pd.DataFrame()

        upcoming_features = build_upcoming_game_features(
            schedule_df=upcoming,
            team_feature_df=team_features,
            training_feature_cols=feature_cols,
        )
        upcoming_features[feature_cols] = upcoming_features[feature_cols].fillna(0.0)
        save_csv(upcoming_features, self.upcoming_features_path)
        to_sqlite(upcoming_features, "upcoming_features", self.paths.sqlite_path, if_exists="replace")

        pred = predict_dataframe(model=model, calibrator=calibrator, df=upcoming_features, feature_cols=feature_cols)
        pred["PREDICTED_WINNER"] = np.where(
            pred["PRED_HOME_WIN"] == 1,
            pred["HOME_TEAM_ABBREVIATION"],
            pred["AWAY_TEAM_ABBREVIATION"],
        )
        save_csv(pred, self.latest_upcoming_path)
        to_sqlite(pred, "latest_upcoming_predictions", self.paths.sqlite_path, if_exists="replace")
        return pred

    def _update_prediction_archive(self, upcoming_predictions: pd.DataFrame, game_level: pd.DataFrame) -> pd.DataFrame:
        archive = initialize_prediction_archive(self.archive_path)
        if not archive.empty:
            archive["GAME_ID"] = archive["GAME_ID"].astype(str)

        if not upcoming_predictions.empty:
            latest = upcoming_predictions.copy()
            latest["ACTUAL_HOME_WIN"] = np.nan
            latest["IS_CORRECT"] = np.nan
            latest["PREDICTION_TIMESTAMP_UTC"] = now_utc_iso()

            keep = [
                "GAME_ID",
                "GAME_DATE",
                "HOME_TEAM_ABBREVIATION",
                "AWAY_TEAM_ABBREVIATION",
                "PRED_HOME_WIN_PROB",
                "PRED_HOME_WIN",
                "ACTUAL_HOME_WIN",
                "IS_CORRECT",
                "PREDICTION_TIMESTAMP_UTC",
            ]
            latest = latest[keep]

            if not archive.empty:
                # Replace stale pregame predictions for still-unfinished games.
                mask_replace = archive["GAME_ID"].isin(latest["GAME_ID"]) & archive["ACTUAL_HOME_WIN"].isna()
                archive = archive.loc[~mask_replace].copy()
            archive = pd.concat([archive, latest], ignore_index=True)

        archive = reconcile_archive_with_actuals(archive, game_level)
        if "GAME_DATE" in archive.columns:
            archive["GAME_DATE"] = pd.to_datetime(archive["GAME_DATE"], errors="coerce")
        archive = archive.sort_values(["GAME_DATE", "GAME_ID"]).drop_duplicates(subset=["GAME_ID"], keep="last")

        save_csv(archive, self.archive_path)
        to_sqlite(archive, "prediction_archive", self.paths.sqlite_path, if_exists="replace")
        return archive

    def _build_thunder_outputs(self, archive: pd.DataFrame) -> None:
        thunder_abbr = self.cfg["project"]["thunder_team_abbr"]
        thunder = thunder_predictions_only(archive, thunder_abbr=thunder_abbr)
        completed = thunder[thunder["ACTUAL_HOME_WIN"].notna()].copy()

        summary = build_thunder_summary(completed)
        rolling = add_rolling_accuracy(completed, window=10)
        weekly = weekly_performance(completed)

        save_csv(thunder, self.paths.reports_dir / "thunder_predictions_all.csv")
        save_csv(rolling, self.paths.reports_dir / "thunder_predictions_completed.csv")
        save_csv(weekly, self.paths.reports_dir / "thunder_weekly_summary.csv")
        save_json(summary, self.paths.reports_dir / "thunder_summary.json")
        to_sqlite(thunder, "thunder_predictions_all", self.paths.sqlite_path, if_exists="replace")
        to_sqlite(rolling, "thunder_predictions_completed", self.paths.sqlite_path, if_exists="replace")
        to_sqlite(weekly, "thunder_weekly_summary", self.paths.sqlite_path, if_exists="replace")
