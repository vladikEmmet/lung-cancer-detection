"""Handcrafted features for histology images: GLCM, LBP, color histograms, HED moments.

These are the histology analogue of pyradiomics features, used to train SVM/RF/KNN
on the same task as ResNet50 (per reviewer fix: one dataset → all models).
"""
from __future__ import annotations

from typing import List, Sequence, Tuple

import numpy as np
from tqdm import tqdm


def _to_gray_uint8(image: np.ndarray) -> np.ndarray:
    from skimage.color import rgb2gray
    from skimage.util import img_as_ubyte

    if image.ndim == 3:
        return img_as_ubyte(rgb2gray(image))
    return img_as_ubyte(image)


def extract_color_features(image: np.ndarray, bins: int = 16) -> Tuple[np.ndarray, List[str]]:
    """RGB + HSV histograms + first four moments per channel."""
    from skimage.color import rgb2hsv

    values: List[float] = []
    names: List[str] = []

    for c, name in enumerate(("R", "G", "B")):
        chan = image[:, :, c].astype(np.float32) / 255.0
        hist, _ = np.histogram(chan, bins=bins, range=(0.0, 1.0), density=True)
        values.extend(hist.tolist())
        names.extend([f"hist_{name}_{i}" for i in range(bins)])
        values.extend([
            float(chan.mean()),
            float(chan.std()),
            float(((chan - chan.mean()) ** 3).mean()),
            float(((chan - chan.mean()) ** 4).mean()),
        ])
        names.extend([f"{name}_mean", f"{name}_std", f"{name}_skew", f"{name}_kurt"])

    hsv = rgb2hsv(image.astype(np.float32) / 255.0)
    for c, name in enumerate(("H", "S", "V")):
        chan = hsv[:, :, c]
        hist, _ = np.histogram(chan, bins=bins, range=(0.0, 1.0), density=True)
        values.extend(hist.tolist())
        names.extend([f"hist_{name}_{i}" for i in range(bins)])
        values.extend([float(chan.mean()), float(chan.std())])
        names.extend([f"{name}_mean", f"{name}_std"])

    return np.array(values, dtype=np.float32), names


def extract_glcm_features(
    image: np.ndarray,
    distances: Sequence[int] = (1, 2, 3),
    angles: Sequence[float] = (0.0, np.pi / 4, np.pi / 2, 3 * np.pi / 4),
    levels: int = 32,
) -> Tuple[np.ndarray, List[str]]:
    from skimage.feature import graycomatrix, graycoprops

    gray = _to_gray_uint8(image)
    # Quantise to `levels` bins so the GLCM stays small.
    gray = (gray.astype(np.float32) / 256.0 * levels).astype(np.uint8)

    glcm = graycomatrix(
        gray,
        distances=list(distances),
        angles=list(angles),
        levels=levels,
        symmetric=True,
        normed=True,
    )

    props = ["contrast", "dissimilarity", "homogeneity", "energy", "correlation", "ASM"]
    values: List[float] = []
    names: List[str] = []
    for prop in props:
        arr = graycoprops(glcm, prop)  # shape (n_dist, n_angle)
        for d_idx, d in enumerate(distances):
            for a_idx, _ in enumerate(angles):
                values.append(float(arr[d_idx, a_idx]))
                names.append(f"glcm_{prop}_d{d}_a{a_idx}")
    return np.array(values, dtype=np.float32), names


def extract_lbp_features(
    image: np.ndarray,
    radius: int = 1,
    n_points: int = 8,
) -> Tuple[np.ndarray, List[str]]:
    from skimage.feature import local_binary_pattern

    gray = _to_gray_uint8(image)
    lbp = local_binary_pattern(gray, P=n_points, R=radius, method="uniform")
    n_bins = n_points + 2
    hist, _ = np.histogram(lbp.ravel(), bins=n_bins, range=(0, n_bins), density=True)
    names = [f"lbp_{i}" for i in range(n_bins)]
    return hist.astype(np.float32), names


def extract_hed_features(image: np.ndarray) -> Tuple[np.ndarray, List[str]]:
    """H&E color deconvolution → channel-wise mean/std/skew/kurtosis."""
    from skimage.color import rgb2hed

    hed = rgb2hed(image.astype(np.float32) / 255.0)
    values: List[float] = []
    names: List[str] = []
    for c, name in enumerate(("hema", "eosin", "dab")):
        chan = hed[:, :, c]
        m = float(chan.mean())
        s = float(chan.std())
        sk = float(((chan - m) ** 3).mean())
        ku = float(((chan - m) ** 4).mean())
        values.extend([m, s, sk, ku])
        names.extend([f"hed_{name}_mean", f"hed_{name}_std", f"hed_{name}_skew", f"hed_{name}_kurt"])
    return np.array(values, dtype=np.float32), names


def extract_all_features(image: np.ndarray) -> Tuple[np.ndarray, List[str]]:
    vecs, names = [], []
    for fn in (extract_color_features, extract_glcm_features, extract_lbp_features, extract_hed_features):
        v, n = fn(image)
        vecs.append(v)
        names.extend(n)
    return np.concatenate(vecs), names


def build_feature_matrix(
    images: Sequence[np.ndarray],
    verbose: bool = True,
) -> Tuple[np.ndarray, List[str]]:
    """Stack features for every image. Returns (X, feature_names)."""
    feature_names: List[str] = []
    rows: List[np.ndarray] = []

    iterator = enumerate(images)
    if verbose:
        iterator = tqdm(list(iterator), desc="histology features")

    for _, img in iterator:
        vec, names = extract_all_features(img)
        if not feature_names:
            feature_names = names
        rows.append(vec)

    X = np.stack(rows).astype(np.float32) if rows else np.zeros((0, 0), dtype=np.float32)
    return X, feature_names
