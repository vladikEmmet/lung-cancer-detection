"""Model builders + hyperparameter grids for classical ML and ResNet50.

Hyperparameters are kept here (not in notebooks) so that all four notebooks
reference the same canonical configuration.
"""
from __future__ import annotations

from typing import Any, Dict

from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

# --------- classical ML grids ---------

SVM_GRID: Dict[str, Any] = {
    "C": [0.1, 1, 10, 100],
    "kernel": ["linear", "rbf"],
    "gamma": ["scale", 0.01, 0.1],
}

RF_GRID: Dict[str, Any] = {
    "n_estimators": [100, 200, 500],
    "max_depth": [None, 10, 20],
    "min_samples_split": [2, 5, 10],
}

KNN_GRID: Dict[str, Any] = {
    "n_neighbors": [3, 5, 7, 9, 11, 15, 21],
    "weights": ["uniform", "distance"],
    "p": [1, 2],
}


def build_svm(seed: int = 42) -> SVC:
    return SVC(probability=True, class_weight="balanced", random_state=seed)


def build_rf(seed: int = 42) -> RandomForestClassifier:
    return RandomForestClassifier(class_weight="balanced", random_state=seed, n_jobs=-1)


def build_knn() -> KNeighborsClassifier:
    return KNeighborsClassifier(n_jobs=-1)


CLASSICAL_REGISTRY = {
    "SVM": (build_svm, SVM_GRID),
    "RF": (build_rf, RF_GRID),
    "KNN": (build_knn, KNN_GRID),
}


# --------- ResNet50 ---------

# Fixed DL config (matches diploma Section 3 + report fixes).
RESNET50_CONFIG: Dict[str, Any] = {
    "lr": 1e-4,
    "weight_decay": 1e-4,
    "batch_size": 32,
    "epochs": 50,
    "optimizer": "AdamW",
    "scheduler": "CosineAnnealingLR",
    "scheduler_t_max": 50,
    "patience": 7,
    "loss": "BCEWithLogitsLoss(pos_weight)",
}


def build_resnet50(pretrained: bool = True, num_classes: int = 1):
    """Binary-head ResNet50 (num_classes=1 + BCEWithLogitsLoss).

    Returns a torch.nn.Module. Importing torch lazily so that the module
    can be loaded in environments that have only the classical stack.
    """
    import torch
    from torchvision import models

    weights = models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
    net = models.resnet50(weights=weights)
    in_features = net.fc.in_features
    net.fc = torch.nn.Linear(in_features, num_classes)
    return net
