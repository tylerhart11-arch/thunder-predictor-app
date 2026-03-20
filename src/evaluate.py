from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    confusion_matrix,
    log_loss,
    roc_auc_score,
)


def classification_metrics(y_true: pd.Series, y_prob: np.ndarray, threshold: float = 0.5) -> dict[str, Any]:
    y_true_arr = y_true.astype(int).to_numpy()
    y_prob_arr = np.clip(np.asarray(y_prob, dtype=float), 1e-6, 1 - 1e-6)
    y_pred_arr = (y_prob_arr >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true_arr, y_pred_arr, labels=[0, 1]).ravel()
    out = {
        "accuracy": float(accuracy_score(y_true_arr, y_pred_arr)),
        "log_loss": float(log_loss(y_true_arr, y_prob_arr)),
        "roc_auc": float(roc_auc_score(y_true_arr, y_prob_arr)),
        "brier_score": float(brier_score_loss(y_true_arr, y_prob_arr)),
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
        "n_obs": int(len(y_true_arr)),
    }
    return out


def reliability_curve_df(y_true: pd.Series, y_prob: np.ndarray, bins: int = 10) -> pd.DataFrame:
    frac_pos, mean_pred = calibration_curve(y_true.astype(int), y_prob, n_bins=bins, strategy="quantile")
    return pd.DataFrame({"mean_pred_prob": mean_pred, "observed_win_rate": frac_pos})


def save_reliability_plot(rel_df: pd.DataFrame, path: Path, title: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(7, 5))
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Perfect calibration")
    plt.plot(rel_df["mean_pred_prob"], rel_df["observed_win_rate"], marker="o", label="Model")
    plt.xlabel("Predicted probability")
    plt.ylabel("Observed frequency")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def save_confusion_matrix_plot(metrics: dict[str, Any], path: Path, title: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cm = metrics["confusion_matrix"]
    matrix = np.array([[cm["tn"], cm["fp"]], [cm["fn"], cm["tp"]]])

    plt.figure(figsize=(5, 4))
    plt.imshow(matrix, cmap="Blues")
    plt.title(title)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.xticks([0, 1], ["Loss (0)", "Win (1)"])
    plt.yticks([0, 1], ["Loss (0)", "Win (1)"])

    for i in range(2):
        for j in range(2):
            plt.text(j, i, str(matrix[i, j]), ha="center", va="center", color="black")

    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()

