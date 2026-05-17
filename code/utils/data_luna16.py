"""LUNA16 loading and patch extraction.

Reproduces the diploma's preprocessing pipeline: read .mhd volumes, normalise
HU values, optionally segment lungs, then extract 3D patches (for radiomics)
and 2D axial slices (for ResNet50). All centres come from `annotations.csv`
(positive nodules) and `candidates_V2.csv` (negative candidates).
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


# ---------------------------- IO helpers ----------------------------

def load_annotations(annotations_csv: str | Path) -> pd.DataFrame:
    df = pd.read_csv(annotations_csv)
    expected = {"seriesuid", "coordX", "coordY", "coordZ", "diameter_mm"}
    missing = expected.difference(df.columns)
    if missing:
        raise ValueError(f"annotations.csv missing columns: {missing}")
    df["label"] = 1
    return df


def load_candidates(candidates_csv: str | Path, only_negative: bool = True) -> pd.DataFrame:
    df = pd.read_csv(candidates_csv)
    expected = {"seriesuid", "coordX", "coordY", "coordZ", "class"}
    missing = expected.difference(df.columns)
    if missing:
        raise ValueError(f"candidates_V2.csv missing columns: {missing}")
    if only_negative:
        df = df[df["class"] == 0].copy()
    df["label"] = df["class"].astype(int)
    df["diameter_mm"] = np.nan
    return df[["seriesuid", "coordX", "coordY", "coordZ", "diameter_mm", "label"]]


def find_mhd_path(root: Path, seriesuid: str, subsets: Sequence[str]) -> Optional[Path]:
    for subset in subsets:
        candidate = Path(root) / subset / f"{seriesuid}.mhd"
        if candidate.exists():
            return candidate
    return None


def load_mhd(path: str | Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (volume_zyx, spacing_zyx, origin_zyx)."""
    import SimpleITK as sitk

    img = sitk.ReadImage(str(path))
    volume = sitk.GetArrayFromImage(img)  # (z, y, x)
    spacing = np.array(img.GetSpacing())[::-1]  # SITK gives x,y,z → reverse to z,y,x
    origin = np.array(img.GetOrigin())[::-1]
    return volume, spacing, origin


# ---------------------------- coord conversion ----------------------------

def world_to_voxel(
    world_coord_xyz: Sequence[float],
    origin_zyx: np.ndarray,
    spacing_zyx: np.ndarray,
) -> np.ndarray:
    world_zyx = np.array(world_coord_xyz, dtype=float)[::-1]
    return np.round((world_zyx - origin_zyx) / spacing_zyx).astype(int)


# ---------------------------- preprocessing ----------------------------

def normalize_hu(volume: np.ndarray, low: float = -1000.0, high: float = 400.0) -> np.ndarray:
    v = np.clip(volume.astype(np.float32), low, high)
    return (v - low) / (high - low)


def segment_lungs(volume_hu: np.ndarray) -> np.ndarray:
    """Lightweight lung mask: HU thresholding + morphology.

    Returns a binary 3D mask the same shape as `volume_hu` (HU units, NOT normalised).
    Used both for the optional masking step and as the ROI for pyradiomics.
    """
    from scipy import ndimage as ndi
    from skimage.morphology import binary_closing, binary_opening, ball

    mask = volume_hu < -320
    mask = binary_opening(mask, ball(2))
    mask = binary_closing(mask, ball(3))
    # Keep large connected components only (drops trachea/air outside body).
    labels, n = ndi.label(mask)
    if n == 0:
        return mask
    sizes = ndi.sum(mask, labels, range(n + 1))
    keep = sizes >= max(sizes.max() * 0.05, 1000)
    keep[0] = False
    return keep[labels]


# ---------------------------- patch extraction ----------------------------

##################
def is_valid_center(volume_shape: Sequence[int], center_voxel: Sequence[int], patch_size_3d: int, patch_size_2d: int) -> bool:
    """Check if the center is valid for extracting both 3D and 2D patches."""
    half_3d = patch_size_3d // 2
    half_2d = patch_size_2d // 2
    z, y, x = center_voxel
    
    # Check 3D bounds
    if z - half_3d < 0 or z + half_3d >= volume_shape[0]:
        return False
    if y - half_3d < 0 or y + half_3d >= volume_shape[1]:
        return False
    if x - half_3d < 0 or x + half_3d >= volume_shape[2]:
        return False
    
    # Check 2D bounds
    if y - half_2d < 0 or y + half_2d >= volume_shape[1]:
        return False
    if x - half_2d < 0 or x + half_2d >= volume_shape[2]:
        return False
    
    return True


def _slice_3d(volume: np.ndarray, center: Sequence[int], size: int) -> np.ndarray:
    """Crop a cubic patch with zero-padding when the centre is near an edge."""
    half = size // 2
    z, y, x = center
    z0, y0, x0 = z - half, y - half, x - half
    z1, y1, x1 = z0 + size, y0 + size, x0 + size

    z0c, y0c, x0c = max(z0, 0), max(y0, 0), max(x0, 0)
    z1c, y1c, x1c = min(z1, volume.shape[0]), min(y1, volume.shape[1]), min(x1, volume.shape[2])

    patch = np.zeros((size, size, size), dtype=volume.dtype)
    patch[z0c - z0 : z1c - z0, y0c - y0 : y1c - y0, x0c - x0 : x1c - x0] = (
        volume[z0c:z1c, y0c:y1c, x0c:x1c]
    )
    return patch


def extract_patch_3d(volume: np.ndarray, center_voxel: Sequence[int], size: int = 64) -> np.ndarray:
    return _slice_3d(volume, center_voxel, size)


def extract_slice_2d(volume: np.ndarray, center_voxel: Sequence[int], size: int = 224) -> np.ndarray:
    """Central axial slice as a 2D HxW patch."""
    z = int(center_voxel[0])
    z = min(max(z, 0), volume.shape[0] - 1)
    slab = volume[z]
    half = size // 2
    y, x = int(center_voxel[1]), int(center_voxel[2])
    y0, x0 = y - half, x - half
    y1, x1 = y0 + size, x0 + size

    y0c, x0c = max(y0, 0), max(x0, 0)
    y1c, x1c = min(y1, slab.shape[0]), min(x1, slab.shape[1])

    patch = np.zeros((size, size), dtype=slab.dtype)
    patch[y0c - y0 : y1c - y0, x0c - x0 : x1c - x0] = slab[y0c:y1c, x0c:x1c]
    return patch


def make_sphere_mask(shape: Tuple[int, int, int], radius_voxels: float) -> np.ndarray:
    """Spherical mask centred on the patch — used as ROI for pyradiomics
    when no ground-truth segmentation is available."""
    cz = (shape[0] - 1) / 2.0
    cy = (shape[1] - 1) / 2.0
    cx = (shape[2] - 1) / 2.0
    z = np.arange(shape[0])[:, None, None]
    y = np.arange(shape[1])[None, :, None]
    x = np.arange(shape[2])[None, None, :]
    return ((z - cz) ** 2 + (y - cy) ** 2 + (x - cx) ** 2) <= radius_voxels ** 2


# ---------------------------- dataset builder ----------------------------

def sample_balanced_negatives(
    candidates_neg: pd.DataFrame,
    n_positives_per_series: pd.Series,
    ratio: float,
    seed: int = 42,
) -> pd.DataFrame:
    """Per-series subsample of negatives so the negative:positive ratio ≈ `ratio`."""
    rng = np.random.default_rng(seed)
    keep_rows: List[pd.DataFrame] = []
    for sid, group in candidates_neg.groupby("seriesuid"):
        n_pos = int(n_positives_per_series.get(sid, 0))
        n_keep = max(int(round(n_pos * ratio)), 0)
        if n_keep == 0 or len(group) == 0:
            continue
        idx = rng.choice(len(group), size=min(n_keep, len(group)), replace=False)
        keep_rows.append(group.iloc[idx])
    return pd.concat(keep_rows, ignore_index=True) if keep_rows else candidates_neg.iloc[:0]


def build_patch_dataset(
    annotations: pd.DataFrame,
    negatives: pd.DataFrame,
    luna_root: str | Path,
    subsets: Sequence[str],
    out_dir: str | Path,
    *,
    patch_size_3d: int = 64,
    patch_size_2d: int = 224,
    hu_low: float = -1000.0,
    hu_high: float = 400.0,
    roi_radius_vox: float = 4.0,
    apply_lung_mask: bool = False,
    verbose: bool = True,
) -> pd.DataFrame:
    """Iterate the union of positives and negatives, save (patch_3d, slice_2d, mask).

    Outputs in `out_dir`:
        - patches_3d.npy     (N, D, H, W)  normalised [0, 1]
        - slices_2d.npy      (N, H, W)     normalised [0, 1]
        - masks_3d.npy       (N, D, H, W)  uint8 sphere mask used as ROI
        - labels.csv         metadata + label column
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    luna_root = Path(luna_root)

    rows = pd.concat([annotations, negatives], ignore_index=True)
    rows = rows.sample(frac=1.0, random_state=42).reset_index(drop=True)

    patches3d: List[np.ndarray] = []
    slices2d: List[np.ndarray] = []
    masks3d: List[np.ndarray] = []
    records: List[dict] = []

    grouped = rows.groupby("seriesuid", sort=False)
    n_series = len(grouped)

    for i, (sid, group) in enumerate(grouped):
        path = find_mhd_path(luna_root, sid, subsets)
        if path is None:
            if verbose:
                print(f"[{i + 1}/{n_series}] skip {sid} (no .mhd)")
            continue
        if verbose:
            print(f"[{i + 1}/{n_series}] {sid} → {len(group)} candidates")

        volume, spacing, origin = load_mhd(path)
        if apply_lung_mask:
            lung_mask = segment_lungs(volume)
            volume = np.where(lung_mask, volume, -1000)

        volume_norm = normalize_hu(volume, hu_low, hu_high)

        for _, row in group.iterrows():
            voxel = world_to_voxel(
                (row["coordX"], row["coordY"], row["coordZ"]), origin, spacing
            )
            
            # Skip if the center is too close to edges
            if not is_valid_center(volume_norm.shape, voxel, patch_size_3d, patch_size_2d):
                continue
            
            patch_3d = extract_patch_3d(volume_norm, voxel, patch_size_3d)
            slice_2d = extract_slice_2d(volume_norm, voxel, patch_size_2d)

            # Fixed-radius ROI for both classes — using `diameter_mm` for positives
            # leaks the label into the mask size (and therefore into every shape /
            # intensity feature computed inside it).
            diameter_mm = float(row.get("diameter_mm", np.nan))
            mask = make_sphere_mask(patch_3d.shape, roi_radius_vox).astype(np.uint8)

            patches3d.append(patch_3d.astype(np.float32))
            slices2d.append(slice_2d.astype(np.float32))
            masks3d.append(mask)
            records.append(
                {
                    "seriesuid": sid,
                    "coordX": float(row["coordX"]),
                    "coordY": float(row["coordY"]),
                    "coordZ": float(row["coordZ"]),
                    "voxel_z": int(voxel[0]),
                    "voxel_y": int(voxel[1]),
                    "voxel_x": int(voxel[2]),
                    "diameter_mm": diameter_mm,
                    "label": int(row["label"]),
                }
            )

    patches_arr = np.stack(patches3d) if patches3d else np.zeros((0, patch_size_3d, patch_size_3d, patch_size_3d))
    slices_arr = np.stack(slices2d) if slices2d else np.zeros((0, patch_size_2d, patch_size_2d))
    masks_arr = np.stack(masks3d) if masks3d else np.zeros((0, patch_size_3d, patch_size_3d, patch_size_3d), dtype=np.uint8)
    labels_df = pd.DataFrame(records)

    np.save(out_dir / "patches_3d.npy", patches_arr)
    np.save(out_dir / "slices_2d.npy", slices_arr)
    np.save(out_dir / "masks_3d.npy", masks_arr)
    labels_df.to_csv(out_dir / "labels.csv", index=False)

    if verbose:
        print(
            f"saved {len(labels_df)} samples (pos={int(labels_df['label'].sum())}) → {out_dir}"
        )
    return labels_df


# ---------------------- PyTorch Dataset wrapper ----------------------

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


class SlicesDataset:
    """Wrap a (N, H, W) float32 array of 2D axial slices for ResNet50.

    Each slice is replicated to 3 channels and optionally transformed.
    The dataset is picklable (no lambda captures) so num_workers > 0 works.
    """

    def __init__(self, slices: np.ndarray, labels: np.ndarray, transform=None):
        import torch
        self.slices = slices
        self.labels = labels.astype(np.int64)
        self.transform = transform
        self._mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
        self._std = torch.tensor(IMAGENET_STD).view(3, 1, 1)

    def __len__(self) -> int:
        return len(self.slices)

    def __getitem__(self, idx: int):
        import torch
        arr = self.slices[idx].astype(np.float32)
        arr = np.stack([arr, arr, arr], axis=0)   # (3, H, W)
        t = torch.from_numpy(arr)
        t = (t - self._mean) / self._std          # ImageNet normalisation
        if self.transform is not None:
            t = self.transform(t)
        return t, int(self.labels[idx])
