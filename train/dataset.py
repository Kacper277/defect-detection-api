"""
PyTorch Dataset dla NEU-DET z augmentacją OpenCV/Albumentations.
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
    """Augmentacje dla zbioru treningowego."""
    return A.Compose([
        # 1. Losowa rotacja ±30°
        A.Rotate(limit=30, p=AUGMENTATION_PROB),
        # 2. Losowe odbicie poziome i pionowe
        A.HorizontalFlip(p=AUGMENTATION_PROB),
        A.VerticalFlip(p=0.2),
        # 3. Losowa zmiana jasności i kontrastu
        A.RandomBrightnessContrast(
            brightness_limit=0.2,
            contrast_limit=0.2,
            p=AUGMENTATION_PROB,
        ),
        # 4. Szum Gaussian
        A.GaussNoise(var_limit=(10.0, 50.0), p=0.3),
        # 5. Rozmycie (symulacja nieostrości)
        A.Blur(blur_limit=3, p=0.2),
        # 6. Skalowanie do stałego rozmiaru
        A.Resize(IMG_SIZE, IMG_SIZE),
        # 7. Normalizacja (ImageNet) + konwersja do tensora
        A.Normalize(mean=MEAN, std=STD),
        ToTensorV2(),
    ])


def get_val_transforms() -> A.Compose:
    """Tylko resize + normalizacja dla walidacji/testu."""
    return A.Compose([
        A.Resize(IMG_SIZE, IMG_SIZE),
        A.Normalize(mean=MEAN, std=STD),
        ToTensorV2(),
    ])


class NEUDETDataset(Dataset):
    """
    Dataset ładujący obrazy NEU-DET ze struktury katalogów:
    data/neu-det/{train,validation}/images/{class_name}/*.jpg
    """

    def __init__(
        self,
        split: str = "train",
        transforms: Optional[Callable] = None,
    ):
        """
        Args:
            split: "train" lub "validation"
            transforms: kompozycja augmentacji Albumentations
        """
        self.split = split
        self.transforms = transforms or (get_train_transforms() if split == "train" else get_val_transforms())

        self.image_dir = DATA_DIR / split / "images"
        self.samples: list[tuple[str, int]] = []  # (ścieżka, label)

        # Skanuj katalogi klas
        for class_name in CLASSES:
            class_dir = self.image_dir / class_name
            if not class_dir.exists():
                print(f"  [WARN] Brak katalogu: {class_dir}")
                continue

            label = CLASS_TO_IDX[class_name]
            for img_path in sorted(class_dir.glob("*.jpg")):
                self.samples.append((str(img_path), label))

        print(f"  Załadowano {len(self.samples)} obrazów dla split='{split}'")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        img_path, label = self.samples[idx]

        # Wczytaj obraz za pomocą OpenCV (BGR -> RGB)
        img = cv2.imread(img_path)
        if img is None:
            raise FileNotFoundError(f"Nie można wczytać obrazu: {img_path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Augmentacja
        if self.transforms:
            augmented = self.transforms(image=img)
            img_tensor = augmented["image"]
        else:
            # Fallback: ręczna konwersja
            img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
            img_tensor = torch.from_numpy(img.transpose(2, 0, 1)).float() / 255.0

        return img_tensor, label


def get_dataloaders() -> tuple[DataLoader, DataLoader]:
    """Zwraca DataLoadery dla treningu i walidacji."""
    print("Przygotowywanie DataLoaderów...")

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
    # Test: załaduj i wyświetl statystyki
    train_loader, val_loader = get_dataloaders()

    print(f"\nBatch train: {len(train_loader)} batchów po {BATCH_SIZE}")
    print(f"Batch val:   {len(val_loader)} batchów po {BATCH_SIZE}")

    # Pokaż przykładowy batch
    images, labels = next(iter(train_loader))
    print(f"\nTensor shape: {images.shape}")     # [B, 3, 224, 224]
    print(f"Label shape:  {labels.shape}")       # [B]
    print(f"Label range:  {labels.min().item()} - {labels.max().item()}")
    print(f"Pixel range:  {images.min().item():.3f} - {images.max().item():.3f}")
