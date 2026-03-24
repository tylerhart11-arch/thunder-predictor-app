from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression


@dataclass
class ProbabilityCalibrator:
    base_model: Any
    method: str
    calibrator_model: Any

    def predict_proba(self, X) -> np.ndarray:
        base_prob = np.clip(self.base_model.predict_proba(X)[:, 1], 1e-6, 1 - 1e-6)
        if self.method == "identity":
            cal_prob = base_prob
        elif self.method == "isotonic":
            cal_prob = self.calibrator_model.predict(base_prob)
        else:
            cal_prob = self.calibrator_model.predict_proba(base_prob.reshape(-1, 1))[:, 1]
        cal_prob = np.clip(cal_prob, 1e-6, 1 - 1e-6)
        return np.column_stack([1 - cal_prob, cal_prob])


def fit_validation_calibrator(base_model, X_calib, y_calib, method: str = "isotonic"):
    base_prob = np.clip(base_model.predict_proba(X_calib)[:, 1], 1e-6, 1 - 1e-6)
    y = np.asarray(y_calib).astype(int)
    if len(np.unique(y)) < 2:
        return ProbabilityCalibrator(base_model=base_model, method="identity", calibrator_model=None)

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
    y_arr = np.asarray(y_train).astype(int)
    unique_classes = np.unique(y_arr)
    if len(unique_classes) < 2:
        return ProbabilityCalibrator(base_model=base_model, method="identity", calibrator_model=None)

    class_counts = np.bincount(y_arr)
    min_class_count = int(class_counts[class_counts > 0].min()) if np.any(class_counts > 0) else 0
    if min_class_count < 2:
        return ProbabilityCalibrator(base_model=base_model, method="identity", calibrator_model=None)

    effective_cv = max(2, min(int(cv), min_class_count))
    estimator = clone(base_model)
    try:
        calibrated_model = CalibratedClassifierCV(estimator=estimator, method=method, cv=effective_cv)
    except TypeError:
        calibrated_model = CalibratedClassifierCV(base_estimator=estimator, method=method, cv=effective_cv)
    calibrated_model.fit(X_train, y_arr)
    return calibrated_model


def calibrated_predict_proba(calibrator, X) -> np.ndarray:
    return calibrator.predict_proba(X)[:, 1]

