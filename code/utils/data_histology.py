"""LC25000 histology data utilities (lung binary task)."""
from __future__ import annotations

from pathlib import Path
from typing import Callable, List, Optional, Tuple

import numpy as np
import pandas as pd


LC25000_CLASS_FOLDERS = {
    "lung_aca": ("lung", 1),
    "lung_n": ("lung", 0),
    "lung_scc": ("lung", 1),
    "colon_aca": ("colon", 1),
    "colon_n": ("colon", 0),
}


def load_lc25000(root: str | Path) -> pd.DataFrame:
    """Discover all images under `root`. Expects subfolders matching `LC25000_CLASS_FOLDERS`.

    Layout: <root>/lung_image_sets/{lung_aca, lung_n, lung_scc}/*.jpeg
            <root>/colon_image_sets/{colon_aca, colon_n}/*.jpeg
    or flat: <root>/{lung_aca, ...}/*.jpeg
    """
    root = Path(root)
    rows: List[dict] = []
    for class_name, (tissue, label) in LC25000_CLASS_FOLDERS.items():
        for candidate in (root / class_name, root / f"{tissue}_image_sets" / class_name):
            if candidate.is_dir():
                for img_path in sorted(candidate.glob("*.jpeg")):
                    rows.append(
                        {
                            "path": str(img_path),
                            "class": class_name,
                            "tissue": tissue,
                            "label_binary": label,
                        }
                    )
                break
    if not rows:
        raise FileNotFoundError(f"No LC25000 images found under {root}")
    return pd.DataFrame(rows)


def subset_lung_binary(df: pd.DataFrame) -> pd.DataFrame:
    """Lung tissue only; label_binary already encodes malignant (1) vs benign (0)."""
    return df[df["tissue"] == "lung"].reset_index(drop=True)


def stratified_subsample(df: pd.DataFrame, per_class: int, seed: int = 42) -> pd.DataFrame:
    """Optional balanced subsample to keep runtimes manageable."""
    return (
        df.groupby("class", group_keys=False)
        .apply(lambda g: g.sample(n=min(per_class, len(g)), random_state=seed))
        .reset_index(drop=True)
    )


# ---------------------------- torch Dataset ----------------------------

def _default_eval_transform(image_size: int = 224):
    from torchvision import transforms

    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )


def _default_train_transform(image_size: int = 224):
    from torchvision import transforms

    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )


def get_transforms(image_size: int = 224) -> Tuple[Callable, Callable]:
    return _default_train_transform(image_size), _default_eval_transform(image_size)


class LC25000Dataset:
    """Lightweight Dataset compatible with torch DataLoader."""

    def __init__(
        self,
        df: pd.DataFrame,
        transform: Optional[Callable] = None,
        label_col: str = "label_binary",
    ):
        self.paths = df["path"].tolist()
        self.labels = df[label_col].astype(int).tolist()
        self.transform = transform or _default_eval_transform()

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int):
        from PIL import Image

        img = Image.open(self.paths[idx]).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        return img, self.labels[idx]


def load_image_array(path: str, image_size: int = 224) -> np.ndarray:
    """Load + resize an image into a NumPy array (uint8 RGB) — for feature extraction."""
    from PIL import Image

    img = Image.open(path).convert("RGB").resize((image_size, image_size))
    return np.asarray(img, dtype=np.uint8)
