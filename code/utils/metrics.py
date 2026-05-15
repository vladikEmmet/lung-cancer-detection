"""Classification metrics + threshold calibration for imbalanced binary tasks."""
from __future__ import annotations

from typing import Dict, Iterable, List, Tuple

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

METRIC_NAMES: Tuple[str, ...] = (
    "accuracy",
    "sensitivity",
    "specificity",
    "precision",
    "f1",
    "auc_roc",
    "pr_auc",
    "balanced_accuracy",
)


def _safe_div(num: float, den: float) -> float:
    return float(num) / float(den) if den else 0.0


def compute_metrics(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    threshold: float = 0.5,
) -> Dict[str, float]:
    y_true = np.asarray(y_true).astype(int)
    y_proba = np.asarray(y_proba, dtype=float)
    y_pred = (y_proba >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    sensitivity = _safe_div(tp, tp + fn)
    specificity = _safe_div(tn, tn + fp)

    out = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "sensitivity": sensitivity,
        "specificity": specificity,
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "balanced_accuracy": (sensitivity + specificity) / 2.0,
        "tp": int(tp),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "threshold": float(threshold),
    }

    if len(np.unique(y_true)) == 2:
        out["auc_roc"] = float(roc_auc_score(y_true, y_proba))
        out["pr_auc"] = float(average_precision_score(y_true, y_proba))
    else:
        out["auc_roc"] = float("nan")
        out["pr_auc"] = float("nan")

    return out


def find_optimal_threshold(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    method: str = "youden",
) -> float:
    """Pick a decision threshold from probabilities.

    youden  — argmax(TPR - FPR) on the ROC curve.
    f1      — argmax F1 over a grid of thresholds.
    """
    y_true = np.asarray(y_true).astype(int)
    y_proba = np.asarray(y_proba, dtype=float)

    if method == "youden":
        fpr, tpr, thresholds = roc_curve(y_true, y_proba)
        j = tpr - fpr
        # For degenerate cases (e.g. KNN with discrete probabilities and a
        # perfectly separable training fold) many thresholds share the same J.
        # Picking the first via argmax biases toward 1.0; instead, pick the
        # threshold closest to 0.5 among all that reach max J, after dropping
        # the +inf sentinel sklearn returns at index 0.
        finite = np.isfinite(thresholds) & (j == j.max())
        if finite.any():
            cands = thresholds[finite]
            thr = float(cands[np.argmin(np.abs(cands - 0.5))])
        else:
            thr = 0.5
        return thr

    if method == "f1":
        grid = np.linspace(0.01, 0.99, 99)
        best_t, best_f = 0.5, -1.0
        for t in grid:
            f = f1_score(y_true, (y_proba >= t).astype(int), zero_division=0)
            if f > best_f:
                best_t, best_f = float(t), float(f)
        return best_t

    raise ValueError(f"Unknown threshold method: {method}")


def aggregate_folds(per_fold: Iterable[Dict[str, float]]) -> Dict[str, Dict[str, float]]:
    """Aggregate per-fold metrics → {metric: {mean, std, values}}."""
    per_fold = list(per_fold)
    if not per_fold:
        return {}
    keys = [k for k in METRIC_NAMES if k in per_fold[0]]
    out: Dict[str, Dict[str, float]] = {}
    for k in keys:
        vals = [float(f[k]) for f in per_fold if k in f and np.isfinite(f[k])]
        if not vals:
            out[k] = {"mean": float("nan"), "std": float("nan"), "values": []}
            continue
        out[k] = {
            "mean": float(np.mean(vals)),
            "std": float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0,
            "values": vals,
        }
    return out


def per_fold_metric_list(per_fold: Iterable[Dict[str, float]], metric: str) -> List[float]:
    return [float(f[metric]) for f in per_fold if metric in f]
