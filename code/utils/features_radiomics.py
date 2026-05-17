"""Thin wrapper around pyradiomics for extracting features from 3D patches."""
from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
from tqdm import tqdm


def _get_extractor():
    from radiomics.featureextractor import RadiomicsFeatureExtractor

    settings = {
        "binWidth": 25,
        "interpolator": "sitkBSpline",
        "resampledPixelSpacing": None,  # patches already at uniform spacing
        "force2D": False,
        "label": 1,
        "geometryTolerance": 1e-3,
    }
    extractor = RadiomicsFeatureExtractor(**settings)
    extractor.disableAllFeatures()
    extractor.enableFeatureClassByName("firstorder")
    extractor.enableFeatureClassByName("glcm")
    extractor.enableFeatureClassByName("glrlm")
    extractor.enableFeatureClassByName("gldm")
    extractor.enableFeatureClassByName("ngtdm")
    extractor.enableFeatureClassByName("shape")
    return extractor


def _to_sitk(volume: np.ndarray):
    import SimpleITK as sitk

    img = sitk.GetImageFromArray(volume.astype(np.float32))
    img.SetSpacing((1.0, 1.0, 1.0))
    return img


def extract_features(patch: np.ndarray, mask: np.ndarray) -> Dict[str, float]:
    """Run the canonical pyradiomics feature classes on a single (patch, mask) pair."""
    import SimpleITK as sitk

    if mask.sum() == 0:
        raise ValueError("Empty mask — cannot extract radiomics features.")

    extractor = _get_extractor()
    img = _to_sitk(patch)
    mask_img = sitk.GetImageFromArray(mask.astype(np.uint8))
    mask_img.SetSpacing(img.GetSpacing())
    mask_img.SetOrigin(img.GetOrigin())
    mask_img.SetDirection(img.GetDirection())

    raw = extractor.execute(img, mask_img)
    out: Dict[str, float] = {}
    for k, v in raw.items():
        if k.startswith("diagnostics_"):
            continue
        try:
            out[k] = float(v)
        except (TypeError, ValueError):
            continue
    return out


def build_feature_matrix(
    patches: np.ndarray,
    masks: np.ndarray,
    verbose: bool = True,
) -> Tuple[np.ndarray, List[str]]:
    """Extract features for every (patch, mask). Returns (X, feature_names).

    Failed samples are filled with NaN; the caller is expected to drop or impute them.
    """
    if len(patches) != len(masks):
        raise ValueError(f"patches/masks length mismatch: {len(patches)} vs {len(masks)}")

    feature_names: List[str] = []
    rows: List[List[float]] = []

    iterator = range(len(patches))
    if verbose:
        iterator = tqdm(iterator, desc="pyradiomics")

    for i in iterator:
        try:
            feats = extract_features(patches[i], masks[i])
        except Exception as exc:  # pyradiomics raises a zoo of exception types
            if verbose:
                print(f"  patch {i} failed: {exc}")
            feats = {}
        if not feature_names and feats:
            feature_names = sorted(feats.keys())
        rows.append([feats.get(name, np.nan) for name in feature_names])

    X = np.asarray(rows, dtype=np.float32) if rows else np.zeros((0, 0), dtype=np.float32)
    return X, feature_names
