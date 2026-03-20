from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from sklearn.base import clone
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression


@dataclass
class ProbabilityCalibrator:
    base_model: Any
    method: str
    calibrator_model: Any

    def predict_proba(self, X) -> np.ndarray:
        base_prob = np.clip(self.base_model.predict_proba(X)[:, 1], 1e-6, 1 - 1e-6)
        if self.method == "isotonic":
            cal_prob = self.calibrator_model.predict(base_prob)
        else:
            cal_prob = self.calibrator_model.predict_proba(base_prob.reshape(-1, 1))[:, 1]
        cal_prob = np.clip(cal_prob, 1e-6, 1 - 1e-6)
        return np.column_stack([1 - cal_prob, cal_prob])


def fit_validation_calibrator(base_model, X_calib, y_calib, method: str = "isotonic"):
    base_prob = np.clip(base_model.predict_proba(X_calib)[:, 1], 1e-6, 1 - 1e-6)
    y = np.asarray(y_calib).astype(int)

    if method == "isotonic":
        model = IsotonicRegression(out_of_bounds="clip")
        model.fit(base_prob, y)
    elif method == "sigmoid":
        model = LogisticRegression()
        model.fit(base_prob.reshape(-1, 1), y)
    else:
        raise ValueError(f"Unsupported calibration method: {method}")

    return ProbabilityCalibrator(base_model=base_model, method=method, calibrator_model=model)


def fit_cv_calibrator(base_model, X_train, y_train, method: str = "isotonic", cv: int = 3):
    # Compatibility helper when sklearn prefit calibration is unavailable.
    model_for_cal = clone(base_model)
    model_for_cal.fit(X_train, y_train)
    return fit_validation_calibrator(model_for_cal, X_train, y_train, method=method)


def calibrated_predict_proba(calibrator, X) -> np.ndarray:
    return calibrator.predict_proba(X)[:, 1]

