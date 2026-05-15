"""Statistical comparison + result formatting."""
from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon


def wilcoxon_compare(scores_a: Sequence[float], scores_b: Sequence[float]) -> Tuple[float, float]:
    """Paired Wilcoxon signed-rank test. Returns (statistic, p_value).

    Use with per-fold metric values from the same CV splits across two models.
    """
    a = np.asarray(scores_a, dtype=float)
    b = np.asarray(scores_b, dtype=float)
    if a.shape != b.shape:
        raise ValueError(f"Shape mismatch: {a.shape} vs {b.shape}")
    if np.allclose(a, b):
        return 0.0, 1.0
    stat, p = wilcoxon(a, b, zero_method="zsplit")
    return float(stat), float(p)


def pairwise_wilcoxon(
    model_to_scores: Dict[str, Sequence[float]],
) -> pd.DataFrame:
    """Pairwise Wilcoxon p-values across models on per-fold metric values."""
    names = list(model_to_scores.keys())
    p_matrix = pd.DataFrame(np.ones((len(names), len(names))), index=names, columns=names)
    for i, a in enumerate(names):
        for j, b in enumerate(names):
            if i == j:
                p_matrix.loc[a, b] = 1.0
                continue
            _, p = wilcoxon_compare(model_to_scores[a], model_to_scores[b])
            p_matrix.loc[a, b] = p
    return p_matrix


def format_results_table(
    aggregated_per_model: Dict[str, Dict[str, Dict[str, float]]],
    metrics: Sequence[str] = (
        "accuracy",
        "sensitivity",
        "specificity",
        "precision",
        "f1",
        "balanced_accuracy",
        "auc_roc",
        "pr_auc",
    ),
    decimals: int = 3,
) -> pd.DataFrame:
    """Build a `mean ± std` table: rows = models, cols = metrics."""
    rows = []
    for model_name, agg in aggregated_per_model.items():
        row: Dict[str, str] = {"model": model_name}
        for m in metrics:
            cell = agg.get(m, {})
            if not cell or not np.isfinite(cell.get("mean", float("nan"))):
                row[m] = "—"
                continue
            mean = cell["mean"]
            std = cell.get("std", 0.0)
            row[m] = f"{mean:.{decimals}f} ± {std:.{decimals}f}"
        rows.append(row)
    return pd.DataFrame(rows).set_index("model")


def to_latex(df: pd.DataFrame, caption: str = "", label: str = "") -> str:
    """Light LaTeX export of a results table."""
    return df.to_latex(escape=True, caption=caption, label=label) if caption else df.to_latex(escape=True)
