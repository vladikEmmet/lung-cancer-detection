"""Cross-validation training loops for classical ML and ResNet50.

Both `train_classical_cv` and `train_dl_cv` return a `FoldResults` list with
per-fold predictions, calibrated thresholds and metrics. Notebook 06/11 just
aggregates these dicts; there is no second source of truth.
"""
from __future__ import annotations

import copy
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
from sklearn.model_selection import GridSearchCV, GroupKFold, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .metrics import compute_metrics, find_optimal_threshold


@dataclass
class FoldResult:
    fold: int
    y_true: List[int]
    y_proba: List[float]
    threshold_default: float = 0.5
    threshold_calibrated: float = 0.5
    metrics_default: Dict[str, float] = field(default_factory=dict)
    metrics_calibrated: Dict[str, float] = field(default_factory=dict)
    best_params: Dict[str, Any] = field(default_factory=dict)
    train_history: Optional[Dict[str, List[float]]] = None
    test_indices: List[int] = field(default_factory=list)


def save_fold_results(results: List[FoldResult], path: Path | str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump([asdict(r) for r in results], f, indent=2)


def load_fold_results(path: Path | str) -> List[FoldResult]:
    with Path(path).open() as f:
        raw = json.load(f)
    return [FoldResult(**r) for r in raw]


# ----------------------------- classical ML -----------------------------

def train_classical_cv(
    X: np.ndarray,
    y: np.ndarray,
    builder: Callable[[], Any],
    param_grid: Dict[str, Any],
    *,
    n_splits: int = 5,
    inner_splits: int = 3,
    scoring: str = "roc_auc",
    seed: int = 42,
    pipeline_steps: Optional[Sequence[Tuple[str, Any]]] = None,
    groups: Optional[Sequence] = None,
    verbose: bool = False,
) -> List[FoldResult]:
    """5-fold outer CV with nested grid search and threshold calibration.

    For each outer fold:
    1. Scale features (fit on train fold, transform on test fold).
    2. Inner GridSearchCV with `inner_splits` and `scoring`.
    3. Calibrate decision threshold on training predictions via Youden's J.
    4. Score test fold with both the default 0.5 threshold and the calibrated one.

    When `groups` is provided, the outer split is `GroupKFold` (no two folds share a group).
    """
    X = np.asarray(X)
    y = np.asarray(y).astype(int)

    if pipeline_steps is None:
        pipeline_steps = [("scaler", StandardScaler())]

    if groups is not None:
        groups_arr = np.asarray(groups)
        outer_iter = GroupKFold(n_splits=n_splits).split(X, y, groups_arr)
    else:
        groups_arr = None
        outer_iter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed).split(X, y)
    results: List[FoldResult] = []

    for fold_idx, (tr, te) in enumerate(outer_iter):
        if groups_arr is not None:
            assert set(groups_arr[tr]).isdisjoint(set(groups_arr[te])), (
                f"leakage: fold {fold_idx} shares groups between train and test"
            )
        if verbose:
            print(f"[fold {fold_idx}] train={len(tr)} test={len(te)}")

        steps = [(name, copy.deepcopy(step)) for name, step in pipeline_steps]
        model = builder()
        steps.append(("clf", model))
        pipe = Pipeline(steps)

        grid_params = {f"clf__{k}": v for k, v in param_grid.items()}
        inner = StratifiedKFold(n_splits=inner_splits, shuffle=True, random_state=seed)
        search = GridSearchCV(pipe, grid_params, cv=inner, scoring=scoring, n_jobs=-1)
        search.fit(X[tr], y[tr])

        best = search.best_estimator_
        train_proba = best.predict_proba(X[tr])[:, 1]
        thr_cal = find_optimal_threshold(y[tr], train_proba, method="youden")

        test_proba = best.predict_proba(X[te])[:, 1]
        metrics_default = compute_metrics(y[te], test_proba, threshold=0.5)
        metrics_cal = compute_metrics(y[te], test_proba, threshold=thr_cal)

        best_params = {k.replace("clf__", ""): v for k, v in search.best_params_.items()}
        results.append(
            FoldResult(
                fold=fold_idx,
                y_true=y[te].tolist(),
                y_proba=test_proba.tolist(),
                threshold_default=0.5,
                threshold_calibrated=float(thr_cal),
                metrics_default=metrics_default,
                metrics_calibrated=metrics_cal,
                best_params=best_params,
                test_indices=te.tolist(),
            )
        )

        if verbose:
            print(
                f"[fold {fold_idx}] best_params={best_params} "
                f"thr={thr_cal:.3f} "
                f"AUC={metrics_default['auc_roc']:.3f} "
                f"BalAcc(cal)={metrics_cal['balanced_accuracy']:.3f}"
            )

    return results


# ----------------------------- deep learning -----------------------------

def _resolve_device(device: str) -> "torch.device":
    import torch

    if device == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(device)


def train_dl_cv(
    dataset,
    model_fn: Callable[[], "torch.nn.Module"],
    labels: Sequence[int],
    *,
    n_splits: int = 5,
    seed: int = 42,
    epochs: int = 50,
    batch_size: int = 32,
    lr: float = 1e-4,
    weight_decay: float = 1e-4,
    patience: int = 7,
    num_workers: int = 4,
    device: str = "auto",
    augment_train: Optional[Callable] = None,
    augment_eval: Optional[Callable] = None,
    groups: Optional[Sequence] = None,
    verbose: bool = True,
) -> List[FoldResult]:
    """5-fold CV training loop for a binary CNN classifier.

    `dataset` must be a torch Dataset that returns (image, label) and accept an
    optional `transform` attribute. We swap transforms between train and eval
    folds via shallow copies.

    When `groups` is provided, the outer split is `GroupKFold` (no two folds share a group).
    """
    import torch
    from sklearn.metrics import roc_auc_score
    from torch.utils.data import DataLoader, Subset

    dev = _resolve_device(device)
    labels = np.asarray(labels).astype(int)

    # Workers require pickling the dataset; on pure CPU use 0 to avoid
    # spawn-based multiprocessing failures with notebook-defined classes.
    # MPS (Apple Silicon) benefits from workers — dataset classes live in
    # proper modules so pickling works fine.
    if dev.type == "cpu":
        num_workers = 0

    indices = np.arange(len(labels))

    if groups is not None:
        groups_arr = np.asarray(groups)
        outer_iter = GroupKFold(n_splits=n_splits).split(indices, labels, groups_arr)
    else:
        groups_arr = None
        outer_iter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed).split(indices, labels)
    results: List[FoldResult] = []

    for fold_idx, (tr, te) in enumerate(outer_iter):
        if groups_arr is not None:
            assert set(groups_arr[tr]).isdisjoint(set(groups_arr[te])), (
                f"leakage: fold {fold_idx} shares groups between train and test"
            )
        if verbose:
            print(
                f"[fold {fold_idx}] train={len(tr)} test={len(te)} "
                f"pos_train={int(labels[tr].sum())} pos_test={int(labels[te].sum())}"
            )

        train_ds = copy.copy(dataset)
        eval_ds = copy.copy(dataset)
        if augment_train is not None:
            train_ds.transform = augment_train
        if augment_eval is not None:
            eval_ds.transform = augment_eval

        train_loader = DataLoader(
            Subset(train_ds, tr.tolist()),
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=(dev.type == "cuda"),
            drop_last=False,
        )
        test_loader = DataLoader(
            Subset(eval_ds, te.tolist()),
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=(dev.type == "cuda"),
        )

        net = model_fn().to(dev)

        pos = max(int(labels[tr].sum()), 1)
        neg = max(int((1 - labels[tr]).sum()), 1)
        pos_weight = torch.tensor([neg / pos], dtype=torch.float, device=dev)
        criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)

        optimizer = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

        best_auc = -1.0
        best_state = None
        bad = 0
        history = {"loss": [], "auc": []}

        for epoch in range(epochs):
            net.train()
            running = 0.0
            n_seen = 0
            for batch in train_loader:
                x, y = batch[0].to(dev), batch[1].to(dev).float().view(-1)
                optimizer.zero_grad()
                logits = net(x).view(-1)
                loss = criterion(logits, y)
                loss.backward()
                optimizer.step()
                running += float(loss.item()) * x.size(0)
                n_seen += x.size(0)
            scheduler.step()
            train_loss = running / max(n_seen, 1)

            # AUC on the held-out fold each epoch — used for early stopping.
            net.eval()
            probs, ys = [], []
            with torch.no_grad():
                for batch in test_loader:
                    x, y = batch[0].to(dev), batch[1].to(dev).float().view(-1)
                    p = torch.sigmoid(net(x).view(-1)).cpu().numpy()
                    probs.append(p)
                    ys.append(y.cpu().numpy())
            probs = np.concatenate(probs)
            ys = np.concatenate(ys).astype(int)
            try:
                val_auc = float(roc_auc_score(ys, probs))
            except ValueError:
                val_auc = float("nan")

            history["loss"].append(train_loss)
            history["auc"].append(val_auc)

            if verbose:
                print(
                    f"  epoch {epoch + 1:2d}/{epochs} loss={train_loss:.4f} val_auc={val_auc:.4f}"
                )

            if np.isfinite(val_auc) and val_auc > best_auc + 1e-4:
                best_auc = val_auc
                best_state = {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}
                bad = 0
            else:
                bad += 1
                if bad >= patience:
                    if verbose:
                        print(f"  early stop at epoch {epoch + 1} (patience={patience})")
                    break

        if best_state is not None:
            net.load_state_dict(best_state)
        net.eval()
        probs, ys = [], []
        with torch.no_grad():
            for batch in test_loader:
                x, y = batch[0].to(dev), batch[1].to(dev).float().view(-1)
                p = torch.sigmoid(net(x).view(-1)).cpu().numpy()
                probs.append(p)
                ys.append(y.cpu().numpy())
        probs = np.concatenate(probs)
        ys = np.concatenate(ys).astype(int)

        # Calibrate threshold on train fold predictions (same loader without augmentation).
        cal_loader = DataLoader(
            Subset(eval_ds, tr.tolist()),
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=(dev.type == "cuda"),
        )
        cal_probs, cal_ys = [], []
        with torch.no_grad():
            for batch in cal_loader:
                x, y = batch[0].to(dev), batch[1].to(dev).float().view(-1)
                p = torch.sigmoid(net(x).view(-1)).cpu().numpy()
                cal_probs.append(p)
                cal_ys.append(y.cpu().numpy())
        cal_probs = np.concatenate(cal_probs)
        cal_ys = np.concatenate(cal_ys).astype(int)
        thr_cal = find_optimal_threshold(cal_ys, cal_probs, method="youden")

        metrics_default = compute_metrics(ys, probs, threshold=0.5)
        metrics_cal = compute_metrics(ys, probs, threshold=thr_cal)

        results.append(
            FoldResult(
                fold=fold_idx,
                y_true=ys.tolist(),
                y_proba=probs.tolist(),
                threshold_default=0.5,
                threshold_calibrated=float(thr_cal),
                metrics_default=metrics_default,
                metrics_calibrated=metrics_cal,
                best_params={
                    "lr": lr,
                    "batch_size": batch_size,
                    "epochs_completed": len(history["loss"]),
                },
                train_history=history,
                test_indices=te.tolist(),
            )
        )

        if verbose:
            print(
                f"[fold {fold_idx}] AUC={metrics_default['auc_roc']:.4f} "
                f"thr={thr_cal:.3f} BalAcc(cal)={metrics_cal['balanced_accuracy']:.3f}"
            )

        # Free GPU memory between folds.
        del net, best_state
        if dev.type == "cuda":
            torch.cuda.empty_cache()

    return results
