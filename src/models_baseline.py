from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def train_baseline_logistic(
    train_df: pd.DataFrame,
    feature_cols: list[str],
    target_col: str,
    cfg: dict[str, Any],
):
    baseline_cfg = cfg["model"]["baseline"]
    model = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            (
                "logreg",
                LogisticRegression(
                    max_iter=int(baseline_cfg.get("max_iter", 3000)),
                    class_weight=baseline_cfg.get("class_weight", "balanced"),
                    random_state=int(cfg["project"]["random_seed"]),
                ),
            ),
        ]
    )
    model.fit(train_df[feature_cols], train_df[target_col].astype(int))
    return model


def predict_proba(model, df: pd.DataFrame, feature_cols: list[str]) -> np.ndarray:
    return model.predict_proba(df[feature_cols])[:, 1]


def logistic_feature_importance(model, feature_cols: list[str]) -> pd.DataFrame:
    coef = model.named_steps["logreg"].coef_[0]
    out = pd.DataFrame({"feature": feature_cols, "coefficient": coef})
    out["abs_coefficient"] = out["coefficient"].abs()
    return out.sort_values("abs_coefficient", ascending=False).reset_index(drop=True)

