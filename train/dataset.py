"""
PyTorch Dataset for NEU-DET with OpenCV/Albumentations augmentations.
"""
import cv2
import numpy as np
from pathlib import Path
from typing import Optional, Callable

import torch
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2

from train.config import (
    DATA_DIR,
    CLASSES,
    CLASS_TO_IDX,
    IMG_SIZE,
    MEAN,
    STD,
    BATCH_SIZE,
    NUM_WORKERS,
    AUGMENTATION_PROB,
)


def get_train_transforms() -> A.Compose:
    """Augmentations for the training set."""
    return A.Compose([
        # 1. Random rotation ±30°
        A.Rotate(limit=30, p=AUGMENTATION_PROB),
        # 2. Random horizontal and vertical flip
        A.HorizontalFlip(p=AUGMENTATION_PROB),
        A.VerticalFlip(p=0.2),
        # 3. Random brightness and contrast
        A.RandomBrightnessContrast(
            brightness_limit=0.2,
            contrast_limit=0.2,
            p=AUGMENTATION_PROB,
        ),
        # 4. Gaussian noise
        A.GaussNoise(var_limit=(10.0, 50.0), p=0.3),
        # 5. Blur (simulates lack of focus)
        A.Blur(blur_limit=3, p=0.2),
        # 6. Resize to fixed size
        A.Resize(IMG_SIZE, IMG_SIZE),
        # 7. Normalization (ImageNet) + tensor conversion
        A.Normalize(mean=MEAN, std=STD),
        ToTensorV2(),
    ])


def get_val_transforms() -> A.Compose:
    """Only resize + normalization for validation/test."""
    return A.Compose([
        A.Resize(IMG_SIZE, IMG_SIZE),
        A.Normalize(mean=MEAN, std=STD),
        ToTensorV2(),
    ])


class NEUDETDataset(Dataset):
    """
    Dataset loading NEU-DET images from directory structure:
    data/neu-det/{train,validation}/images/{class_name}/*.jpg
    """

    def __init__(
        self,
        split: str = "train",
        transforms: Optional[Callable] = None,
    ):
        """
        Args:
            split: "train" or "validation"
            transforms: Albumentations augmentation composition
        """
        self.split = split
        self.transforms = transforms or (get_train_transforms() if split == "train" else get_val_transforms())

        self.image_dir = DATA_DIR / split / "images"
        self.samples: list[tuple[str, int]] = []  # (path, label)

        # Scan class directories
        for class_name in CLASSES:
            class_dir = self.image_dir / class_name
            if not class_dir.exists():
                print(f"  [WARN] Directory missing: {class_dir}")
                continue

            label = CLASS_TO_IDX[class_name]
            for img_path in sorted(class_dir.glob("*.jpg")):
                self.samples.append((str(img_path), label))

        print(f"  Loaded {len(self.samples)} images for split='{split}'")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        img_path, label = self.samples[idx]

        # Load image with OpenCV (BGR -> RGB)
        img = cv2.imread(img_path)
        if img is None:
            raise FileNotFoundError(f"Cannot load image: {img_path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Augmentation
        if self.transforms:
            augmented = self.transforms(image=img)
            img_tensor = augmented["image"]
        else:
            # Fallback: manual conversion
            img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
            img_tensor = torch.from_numpy(img.transpose(2, 0, 1)).float() / 255.0

        return img_tensor, label


def get_dataloaders() -> tuple[DataLoader, DataLoader]:
    """Returns DataLoaders for training and validation."""
    print("Preparing DataLoaders...")

    train_dataset = NEUDETDataset(split="train", transforms=get_train_transforms())
    val_dataset = NEUDETDataset(split="validation", transforms=get_val_transforms())

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )

    return train_loader, val_loader


if __name__ == "__main__":
    # Test: load and display statistics
    train_loader, val_loader = get_dataloaders()

    print(f"\nTrain batches: {len(train_loader)} batches of {BATCH_SIZE}")
    print(f"Val batches:   {len(val_loader)} batches of {BATCH_SIZE}")

    # Show sample batch
    images, labels = next(iter(train_loader))
    print(f"\nTensor shape: {images.shape}")     # [B, 3, 224, 224]
    print(f"Label shape:  {labels.shape}")       # [B]
    print(f"Label range:  {labels.min().item()} - {labels.max().item()}")
    print(f"Pixel range:  {images.min().item():.3f} - {images.max().item():.3f}")