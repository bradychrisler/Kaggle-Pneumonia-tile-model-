"""
Dataset and DataLoader helpers for the RSNA Pneumonia Texture Analysis competition.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import albumentations as A
from albumentations.pytorch import ToTensorV2


@dataclass
class DatasetConfig:
    metadata_csv: Path
    image_column: str = "image_path"
    label_column: str = "opacity"
    fold_column: str = "fold"
    patient_column: Optional[str] = "PatientSex"
    image_size: int = 224
    mean: Tuple[float, float, float] = (0.485, 0.456, 0.406)
    std: Tuple[float, float, float] = (0.229, 0.224, 0.225)


class PneumoniaDataset(Dataset):
    def __init__(
        self,
        dataframe: pd.DataFrame,
        image_root: Path,
        transforms_fn: Optional[Callable] = None,
        label_column: str = "opacity",
    ) -> None:
        self.df = dataframe.reset_index(drop=True)
        self.image_root = image_root
        self.transforms_fn = transforms_fn
        self.label_column = label_column

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        image_path = self.image_root / row["image_path"]
        image = Image.open(image_path).convert("RGB")
        image = np.array(image)
        
        if self.transforms_fn:
            # Check if it's albumentations (A.Compose) or torchvision
            if isinstance(self.transforms_fn, A.Compose):
                # Albumentations - expects dict input
                transformed = self.transforms_fn(image=image)
                image = transformed["image"]
            else:
                # Torchvision
                image = Image.fromarray(image)
                image = self.transforms_fn(image)
        else:
            image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
        
        label = torch.tensor(float(row[self.label_column]), dtype=torch.float32)
        return image, label


def default_transforms(image_size: int) -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ]
    )


def train_transforms(image_size: int) -> A.Compose:
    """Enhanced augmentation for training with medical imaging best practices."""
    return A.Compose(
        [
            A.Resize(image_size, image_size),
            A.HorizontalFlip(p=0.5),
            A.Rotate(limit=15, p=0.5),
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
            A.GaussNoise(std_range=(0.04, 0.2), p=0.3),
            A.GaussianBlur(blur_limit=(3, 7), p=0.3),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ]
    )


def valid_transforms(image_size: int) -> A.Compose:
    """Minimal transforms for validation."""
    return A.Compose(
        [
            A.Resize(image_size, image_size),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ]
    )


def create_folds(df: pd.DataFrame, n_splits: int = 5, target_col: str = "opacity") -> pd.DataFrame:
    df = df.copy()
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    df["fold"] = -1
    for fold, (_, valid_idx) in enumerate(skf.split(df, df[target_col])):
        df.loc[df.index[valid_idx], "fold"] = fold
    return df


def build_dataloaders(
    df: pd.DataFrame,
    image_root: Path,
    fold: int,
    batch_size: int,
    num_workers: int = 4,
    image_size: int = 224,
    use_albumentations: bool = True,
) -> Tuple[DataLoader, DataLoader]:
    train_df = df[df["fold"] != fold].reset_index(drop=True)
    valid_df = df[df["fold"] == fold].reset_index(drop=True)

    if use_albumentations:
        train_tf = train_transforms(image_size)
        valid_tf = valid_transforms(image_size)
    else:
        train_tf = default_transforms(image_size)
        valid_tf = default_transforms(image_size)
    
    train_ds = PneumoniaDataset(train_df, image_root, train_tf)
    valid_ds = PneumoniaDataset(valid_df, image_root, valid_tf)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )

    valid_loader = DataLoader(
        valid_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    return train_loader, valid_loader

