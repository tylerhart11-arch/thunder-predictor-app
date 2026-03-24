from __future__ import annotations

from itertools import product
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import log_loss
from sklearn.pipeline import Pipeline

try:
    from xgboost import XGBClassifier

    XGBOOST_AVAILABLE = True
except Exception:  # noqa: BLE001
    XGBOOST_AVAILABLE = False


def _xgb_candidates(param_grid: dict[str, list]) -> list[dict[str, Any]]:
    keys = list(param_grid.keys())
    values = [param_grid[k] for k in keys]
    return [dict(zip(keys, v)) for v in product(*values)]


def _build_xgb(params: dict[str, Any], seed: int):
    return XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=seed,
        tree_method="hist",
        n_estimators=int(params["n_estimators"]),
        max_depth=int(params["max_depth"]),
        learning_rate=float(params["learning_rate"]),
        subsample=float(params["subsample"]),
        colsample_bytree=float(params["colsample_bytree"]),
        n_jobs=-1,
    )


def _build_fallback(params: dict[str, Any], seed: int):
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            (
                "gbm",
                HistGradientBoostingClassifier(
                    learning_rate=float(params["learning_rate"]),
                    max_depth=int(params["max_depth"]),
                    max_iter=int(params["n_estimators"]),
                    random_state=seed,
                ),
            ),
        ]
    )


def fit_improved_model_with_params(
    train_df: pd.DataFrame,
    feature_cols: list[str],
    target_col: str,
    params: dict[str, Any],
    cfg: dict[str, Any],
):
    seed = int(cfg["project"]["random_seed"])
    xgb_cfg = cfg["model"]["xgboost"]
    X_train = train_df[feature_cols]
    y_train = train_df[target_col].astype(int)

    if xgb_cfg.get("use_xgboost", True) and XGBOOST_AVAILABLE:
        model = _build_xgb(params, seed=seed)
    else:
        model = _build_fallback(params, seed=seed)
    model.fit(X_train, y_train)
    return model


def tune_and_train_improved_model(
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    feature_cols: list[str],
    target_col: str,
    cfg: dict[str, Any],
    logger,
):
    xgb_cfg = cfg["model"]["xgboost"]
    seed = int(cfg["project"]["random_seed"])
    candidates = _xgb_candidates(xgb_cfg["param_grid"])

    X_train = train_df[feature_cols]
    y_train = train_df[target_col].astype(int)
    X_valid = valid_df[feature_cols]
    y_valid = valid_df[target_col].astype(int)

    results = []
    best_model = None
    best_params = None
    best_logloss = float("inf")
    early_stopping_rounds = int(xgb_cfg.get("early_stopping_rounds", 30))

    for i, params in enumerate(candidates, start=1):
        logger.info("Tuning candidate %s/%s: %s", i, len(candidates), params)
        if xgb_cfg.get("use_xgboost", True) and XGBOOST_AVAILABLE:
            model = _build_xgb(params, seed=seed)
            try:
                model.fit(
                    X_train,
                    y_train,
                    eval_set=[(X_valid, y_valid)],
                    early_stopping_rounds=early_stopping_rounds,
                    verbose=False,
                )
            except TypeError:
                model.fit(
                    X_train,
                    y_train,
                    eval_set=[(X_valid, y_valid)],
                    verbose=False,
                )
        else:
            model = _build_fallback(params, seed=seed)
            model.fit(X_train, y_train)

        val_prob = model.predict_proba(X_valid)[:, 1]
        val_ll = float(log_loss(y_valid, np.clip(val_prob, 1e-6, 1 - 1e-6)))
        results.append({"params": params, "validation_log_loss": val_ll})
        if val_ll < best_logloss:
            best_logloss = val_ll
            best_model = model
            best_params = params

    if best_model is None or best_params is None:
        raise RuntimeError("No improved-model candidate could be trained successfully.")

    results_df = pd.DataFrame(results).sort_values("validation_log_loss").reset_index(drop=True)
    logger.info("Best improved-model params: %s (validation log_loss=%.5f)", best_params, best_logloss)
    return best_model, best_params, results_df


def improved_feature_importance(model, feature_cols: list[str]) -> pd.DataFrame:
    if hasattr(model, "feature_importances_"):
        imp = model.feature_importances_
    elif hasattr(model, "named_steps") and "gbm" in model.named_steps:
        imp = model.named_steps["gbm"].feature_importances_
    else:
        imp = np.zeros(len(feature_cols))

    out = pd.DataFrame({"feature": feature_cols, "importance": imp})
    return out.sort_values("importance", ascending=False).reset_index(drop=True)
