from __future__ import annotations

from dataclasses import dataclass
import hashlib
from typing import Any

import numpy as np
import pandas as pd

from app.evaluate import classification_metrics
from app.utils import now_utc_iso


CONFIDENCE_BIN_EDGES = [0.50, 0.55, 0.60, 0.65, 0.75, 0.85, 1.01]
CONFIDENCE_BIN_LABELS = ["50-55%", "55-60%", "60-65%", "65-75%", "75-85%", "85-100%"]

DEFAULT_MONITORING_CFG = {
    "enabled": True,
    "recent_games_window": 30,
    "min_completed_games": 60,
    "min_prior_games": 30,
    "accuracy_drop_threshold": 0.05,
    "log_loss_increase_threshold": 0.05,
    "brier_increase_threshold": 0.02,
    "thunder_recent_games_window": 10,
}


@dataclass
class MonitoringArtifacts:
    summary: dict[str, Any]
    windows: pd.DataFrame
    segments: pd.DataFrame
    confidence_buckets: pd.DataFrame
    completed_outcomes: pd.DataFrame


def monitoring_settings(cfg: dict[str, Any]) -> dict[str, Any]:
    update_cfg = cfg.get("update", {})
    perf_cfg = update_cfg.get("performance_monitoring", {}) if isinstance(update_cfg, dict) else {}
    out = DEFAULT_MONITORING_CFG.copy()
    out.update(perf_cfg)
    return out


def feature_schema_hash(feature_cols: list[str]) -> str:
    normalized = "||".join(str(col) for col in feature_cols)
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def prepare_prediction_outcomes(pred_archive: pd.DataFrame, thunder_abbr: str = "OKC") -> pd.DataFrame:
    if pred_archive.empty:
        return pd.DataFrame()

    df = pred_archive.copy()
    if "GAME_DATE" in df.columns:
        df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"], errors="coerce")
    if "PREDICTION_TIMESTAMP_UTC" in df.columns:
        df["PREDICTION_TIMESTAMP_UTC"] = pd.to_datetime(df["PREDICTION_TIMESTAMP_UTC"], utc=True, errors="coerce")
        df["PREDICTION_TIMESTAMP_UTC"] = df["PREDICTION_TIMESTAMP_UTC"].dt.tz_localize(None)

    required_cols = {"PRED_HOME_WIN_PROB", "ACTUAL_HOME_WIN", "HOME_TEAM_ABBREVIATION", "AWAY_TEAM_ABBREVIATION"}
    if not required_cols.issubset(df.columns):
        return pd.DataFrame()

    df = df[df["ACTUAL_HOME_WIN"].notna() & df["PRED_HOME_WIN_PROB"].notna()].copy()
    if df.empty:
        return pd.DataFrame()

    df["ACTUAL_HOME_WIN"] = df["ACTUAL_HOME_WIN"].astype(int)
    df["PRED_HOME_WIN_PROB"] = df["PRED_HOME_WIN_PROB"].astype(float).clip(1e-6, 1 - 1e-6)
    df["PRED_HOME_WIN"] = (df["PRED_HOME_WIN_PROB"] >= 0.5).astype(int)
    df["IS_CORRECT"] = (df["PRED_HOME_WIN"] == df["ACTUAL_HOME_WIN"]).astype(int)
    df["PREDICTION_CONFIDENCE"] = np.where(
        df["PRED_HOME_WIN"] == 1,
        df["PRED_HOME_WIN_PROB"],
        1 - df["PRED_HOME_WIN_PROB"],
    )

    thunder_home = df["HOME_TEAM_ABBREVIATION"].eq(thunder_abbr)
    thunder_away = df["AWAY_TEAM_ABBREVIATION"].eq(thunder_abbr)
    df["THUNDER_GAME"] = thunder_home | thunder_away
    df["THUNDER_IS_HOME"] = thunder_home
    df["THUNDER_WIN_PROB"] = np.where(thunder_home, df["PRED_HOME_WIN_PROB"], 1 - df["PRED_HOME_WIN_PROB"])
    df["THUNDER_ACTUAL_WIN"] = np.where(thunder_home, df["ACTUAL_HOME_WIN"], 1 - df["ACTUAL_HOME_WIN"])
    df["THUNDER_PRED_WIN"] = (df["THUNDER_WIN_PROB"] >= 0.5).astype(int)
    df["THUNDER_PREDICTION_CONFIDENCE"] = np.where(
        df["THUNDER_PRED_WIN"] == 1,
        df["THUNDER_WIN_PROB"],
        1 - df["THUNDER_WIN_PROB"],
    )

    df["SORT_DATE"] = df["GAME_DATE"]
    if "PREDICTION_TIMESTAMP_UTC" in df.columns:
        df["SORT_DATE"] = df["SORT_DATE"].fillna(df["PREDICTION_TIMESTAMP_UTC"])

    sort_cols = ["SORT_DATE"]
    if "GAME_ID" in df.columns:
        sort_cols.append("GAME_ID")
    return df.sort_values(sort_cols).reset_index(drop=True)


def build_monitoring_artifacts(
    pred_archive: pd.DataFrame,
    thunder_abbr: str,
    monitoring_cfg: dict[str, Any],
    model_meta: dict[str, Any] | None = None,
    reference_metrics: dict[str, Any] | None = None,
) -> MonitoringArtifacts:
    completed = prepare_prediction_outcomes(pred_archive, thunder_abbr=thunder_abbr)
    if completed.empty:
        empty_windows = pd.DataFrame(columns=_metric_columns(include_window=True))
        empty_segments = pd.DataFrame(columns=_metric_columns(include_window=False))
        empty_buckets = pd.DataFrame(
            columns=["scope", "confidence_bucket", "games", "avg_confidence", "observed_accuracy"]
        )
        summary = {
            "generated_at_utc": now_utc_iso(),
            "status": "no_completed_predictions",
            "live_retrain_recommended": False,
            "alerts": [],
            "warnings": ["No completed prediction outcomes are available yet."],
            "league": {},
            "thunder": {},
        }
        return MonitoringArtifacts(
            summary=summary,
            windows=empty_windows,
            segments=empty_segments,
            confidence_buckets=empty_buckets,
            completed_outcomes=pd.DataFrame(),
        )

    league = completed.copy()
    thunder = completed[completed["THUNDER_GAME"]].copy()

    recent_window = int(monitoring_cfg["recent_games_window"])
    thunder_recent_window = int(monitoring_cfg["thunder_recent_games_window"])

    windows = _concat_frames(
        [
            _window_metrics(
                scope="league",
                frame=league,
                target_col="ACTUAL_HOME_WIN",
                prob_col="PRED_HOME_WIN_PROB",
                windows=[10, 25, recent_window, 50],
            ),
            _window_metrics(
                scope="thunder",
                frame=thunder,
                target_col="THUNDER_ACTUAL_WIN",
                prob_col="THUNDER_WIN_PROB",
                windows=[5, thunder_recent_window, 20],
            ),
        ],
        empty_columns=_metric_columns(include_window=True),
    )
    segments = _concat_frames(
        [
            _league_segments(league),
            _thunder_segments(thunder),
        ],
        empty_columns=_metric_columns(include_window=False),
    )
    confidence_buckets = _concat_frames(
        [
            _confidence_bucket_metrics(scope="league", frame=league, confidence_col="PREDICTION_CONFIDENCE"),
            _confidence_bucket_metrics(scope="thunder", frame=thunder, confidence_col="THUNDER_PREDICTION_CONFIDENCE"),
        ],
        empty_columns=["scope", "confidence_bucket", "games", "avg_confidence", "observed_accuracy"],
    )

    summary = _build_summary(
        league=league,
        thunder=thunder,
        monitoring_cfg=monitoring_cfg,
        model_meta=model_meta or {},
        reference_metrics=reference_metrics or {},
    )
    return MonitoringArtifacts(
        summary=summary,
        windows=windows,
        segments=segments,
        confidence_buckets=confidence_buckets,
        completed_outcomes=completed,
    )


def _window_metrics(
    scope: str,
    frame: pd.DataFrame,
    target_col: str,
    prob_col: str,
    windows: list[int],
) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame(columns=_metric_columns(include_window=True))

    rows = [_metric_row(scope=scope, segment="all", frame=frame, target_col=target_col, prob_col=prob_col, window_label="all")]
    seen: set[int] = set()
    for size in windows:
        size = int(size)
        if size <= 0 or size in seen:
            continue
        seen.add(size)
        subset = frame.tail(size).copy()
        if subset.empty:
            continue
        rows.append(
            _metric_row(
                scope=scope,
                segment=f"recent_{size}",
                frame=subset,
                target_col=target_col,
                prob_col=prob_col,
                window_label=f"last_{size}",
            )
        )
    return pd.DataFrame(rows)


def _league_segments(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame(columns=_metric_columns(include_window=False))

    segments = {
        "predicted_home": frame[frame["PRED_HOME_WIN"] == 1],
        "predicted_away": frame[frame["PRED_HOME_WIN"] == 0],
        "high_confidence": frame[frame["PREDICTION_CONFIDENCE"] >= 0.65],
        "coin_flip": frame[frame["PREDICTION_CONFIDENCE"] < 0.55],
    }
    return _segment_rows(scope="league", segments=segments, target_col="ACTUAL_HOME_WIN", prob_col="PRED_HOME_WIN_PROB")


def _thunder_segments(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame(columns=_metric_columns(include_window=False))

    segments = {
        "home": frame[frame["THUNDER_IS_HOME"]],
        "away": frame[~frame["THUNDER_IS_HOME"]],
        "favored": frame[frame["THUNDER_WIN_PROB"] >= 0.5],
        "underdog": frame[frame["THUNDER_WIN_PROB"] < 0.5],
        "high_confidence": frame[frame["THUNDER_PREDICTION_CONFIDENCE"] >= 0.65],
        "coin_flip": frame[frame["THUNDER_PREDICTION_CONFIDENCE"] < 0.55],
    }
    return _segment_rows(scope="thunder", segments=segments, target_col="THUNDER_ACTUAL_WIN", prob_col="THUNDER_WIN_PROB")


def _segment_rows(scope: str, segments: dict[str, pd.DataFrame], target_col: str, prob_col: str) -> pd.DataFrame:
    rows = []
    for segment_name, segment_df in segments.items():
        if segment_df.empty:
            continue
        rows.append(_metric_row(scope=scope, segment=segment_name, frame=segment_df, target_col=target_col, prob_col=prob_col))
    return pd.DataFrame(rows)


def _confidence_bucket_metrics(scope: str, frame: pd.DataFrame, confidence_col: str) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame(columns=["scope", "confidence_bucket", "games", "avg_confidence", "observed_accuracy"])

    bucketed = frame.copy()
    bucketed["confidence_bucket"] = pd.cut(
        bucketed[confidence_col],
        bins=CONFIDENCE_BIN_EDGES,
        labels=CONFIDENCE_BIN_LABELS,
        right=False,
        include_lowest=True,
    )
    summary = (
        bucketed.dropna(subset=["confidence_bucket"])
        .groupby("confidence_bucket", as_index=False, observed=False)
        .agg(
            games=("GAME_ID", "count"),
            avg_confidence=(confidence_col, "mean"),
            observed_accuracy=("IS_CORRECT", "mean"),
        )
    )
    summary.insert(0, "scope", scope)
    return summary


def _metric_row(
    scope: str,
    segment: str,
    frame: pd.DataFrame,
    target_col: str,
    prob_col: str,
    window_label: str | None = None,
) -> dict[str, Any]:
    y_true = frame[target_col].astype(int)
    y_prob = frame[prob_col].astype(float).clip(1e-6, 1 - 1e-6)
    metrics = classification_metrics(y_true, y_prob)
    confidence = np.where(y_prob >= 0.5, y_prob, 1 - y_prob)
    row = {
        "scope": scope,
        "segment": segment,
        "games": int(len(frame)),
        "accuracy": metrics["accuracy"],
        "log_loss": metrics["log_loss"],
        "roc_auc": metrics["roc_auc"],
        "brier_score": metrics["brier_score"],
        "avg_confidence": float(np.mean(confidence)),
        "correct": int(np.sum((y_prob >= 0.5).astype(int) == y_true)),
        "incorrect": int(np.sum((y_prob >= 0.5).astype(int) != y_true)),
        "date_min": _date_value(frame["SORT_DATE"].min()),
        "date_max": _date_value(frame["SORT_DATE"].max()),
    }
    if window_label is not None:
        row["window"] = window_label
    return row


def _build_summary(
    league: pd.DataFrame,
    thunder: pd.DataFrame,
    monitoring_cfg: dict[str, Any],
    model_meta: dict[str, Any],
    reference_metrics: dict[str, Any],
) -> dict[str, Any]:
    recent_window = int(monitoring_cfg["recent_games_window"])
    thunder_recent_window = int(monitoring_cfg["thunder_recent_games_window"])
    min_completed_games = int(monitoring_cfg["min_completed_games"])
    min_prior_games = int(monitoring_cfg["min_prior_games"])
    accuracy_drop = float(monitoring_cfg["accuracy_drop_threshold"])
    log_loss_increase = float(monitoring_cfg["log_loss_increase_threshold"])
    brier_increase = float(monitoring_cfg["brier_increase_threshold"])

    league_overall = _metric_summary(league, "ACTUAL_HOME_WIN", "PRED_HOME_WIN_PROB")
    league_recent = _metric_summary(league.tail(recent_window), "ACTUAL_HOME_WIN", "PRED_HOME_WIN_PROB")
    league_prior = _metric_summary(league.iloc[:-recent_window], "ACTUAL_HOME_WIN", "PRED_HOME_WIN_PROB")

    thunder_overall = _metric_summary(thunder, "THUNDER_ACTUAL_WIN", "THUNDER_WIN_PROB")
    thunder_recent = _metric_summary(thunder.tail(thunder_recent_window), "THUNDER_ACTUAL_WIN", "THUNDER_WIN_PROB")

    alerts: list[str] = []
    warnings: list[str] = []

    enough_league_history = len(league) >= min_completed_games and len(league.iloc[:-recent_window]) >= min_prior_games
    if enough_league_history:
        if _metric_available(league_recent, "accuracy") and _metric_available(league_prior, "accuracy"):
            if league_recent["accuracy"] < league_prior["accuracy"] - accuracy_drop:
                alerts.append(
                    f"League recent accuracy fell to {league_recent['accuracy']:.3f} versus {league_prior['accuracy']:.3f} prior history."
                )
        if _metric_available(league_recent, "log_loss") and _metric_available(league_prior, "log_loss"):
            if league_recent["log_loss"] > league_prior["log_loss"] + log_loss_increase:
                alerts.append(
                    f"League recent log loss rose to {league_recent['log_loss']:.3f} versus {league_prior['log_loss']:.3f} prior history."
                )
        if _metric_available(league_recent, "brier_score") and _metric_available(league_prior, "brier_score"):
            if league_recent["brier_score"] > league_prior["brier_score"] + brier_increase:
                alerts.append(
                    f"League recent Brier score rose to {league_recent['brier_score']:.3f} versus {league_prior['brier_score']:.3f} prior history."
                )
    else:
        warnings.append("Not enough completed league predictions yet for a live retrain signal.")

    if thunder.empty:
        warnings.append("No completed Thunder predictions are available yet.")
    elif len(thunder) < thunder_recent_window:
        warnings.append(
            f"Thunder sample is still small: {len(thunder)} completed games, below the recent window of {thunder_recent_window}."
        )

    if not thunder.empty and len(thunder) >= thunder_recent_window and _metric_available(thunder_recent, "accuracy"):
        if thunder_recent["accuracy"] < thunder_overall["accuracy"] - max(0.10, accuracy_drop):
            warnings.append(
                f"Thunder recent accuracy softened to {thunder_recent['accuracy']:.3f} versus {thunder_overall['accuracy']:.3f} overall."
            )

    holdout_ref = reference_metrics.get("improved_test_calibrated", {}) if isinstance(reference_metrics, dict) else {}
    if holdout_ref:
        if _metric_available(league_recent, "accuracy") and holdout_ref.get("accuracy") is not None:
            if league_recent["accuracy"] < float(holdout_ref["accuracy"]) - accuracy_drop:
                alerts.append(
                    f"League recent live accuracy is below the latest holdout benchmark ({league_recent['accuracy']:.3f} vs {float(holdout_ref['accuracy']):.3f})."
                )
        if _metric_available(league_recent, "log_loss") and holdout_ref.get("log_loss") is not None:
            if league_recent["log_loss"] > float(holdout_ref["log_loss"]) + log_loss_increase:
                alerts.append(
                    f"League recent live log loss is above the latest holdout benchmark ({league_recent['log_loss']:.3f} vs {float(holdout_ref['log_loss']):.3f})."
                )

    status = "healthy"
    if alerts:
        status = "retrain_recommended"
    elif warnings:
        status = "watch"

    return {
        "generated_at_utc": now_utc_iso(),
        "status": status,
        "live_retrain_recommended": bool(alerts),
        "alerts": alerts,
        "warnings": warnings,
        "production_context": {
            "last_retrain_utc": model_meta.get("last_retrain_utc"),
            "n_games_trained": model_meta.get("n_games_trained"),
            "production_train_seasons": model_meta.get("production_train_seasons"),
        },
        "reference_holdout_metrics": holdout_ref,
        "league": {
            "overall": league_overall,
            "recent": league_recent,
            "prior": league_prior,
            "recent_window_games": recent_window,
        },
        "thunder": {
            "overall": thunder_overall,
            "recent": thunder_recent,
            "recent_window_games": thunder_recent_window,
            "sample_warning": warnings[-1] if warnings and "Thunder sample" in warnings[-1] else None,
        },
    }


def _metric_summary(frame: pd.DataFrame, target_col: str, prob_col: str) -> dict[str, Any]:
    if frame.empty:
        return {"games": 0, "accuracy": None, "log_loss": None, "roc_auc": None, "brier_score": None, "avg_confidence": None}

    row = _metric_row(scope="tmp", segment="tmp", frame=frame, target_col=target_col, prob_col=prob_col)
    return {
        "games": row["games"],
        "accuracy": row["accuracy"],
        "log_loss": row["log_loss"],
        "roc_auc": row["roc_auc"],
        "brier_score": row["brier_score"],
        "avg_confidence": row["avg_confidence"],
        "date_min": row["date_min"],
        "date_max": row["date_max"],
    }


def _metric_columns(include_window: bool) -> list[str]:
    cols = [
        "scope",
        "segment",
        "games",
        "accuracy",
        "log_loss",
        "roc_auc",
        "brier_score",
        "avg_confidence",
        "correct",
        "incorrect",
        "date_min",
        "date_max",
    ]
    if include_window:
        cols.insert(2, "window")
    return cols


def _concat_frames(frames: list[pd.DataFrame], empty_columns: list[str]) -> pd.DataFrame:
    non_empty = [frame for frame in frames if not frame.empty]
    if not non_empty:
        return pd.DataFrame(columns=empty_columns)
    return pd.concat(non_empty, ignore_index=True)


def _metric_available(metrics: dict[str, Any], key: str) -> bool:
    return metrics.get(key) is not None


def _date_value(value: Any) -> str | None:
    if pd.isna(value):
        return None
    return str(pd.to_datetime(value).date())
