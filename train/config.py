"""
Konfiguracja projektu - ścieżki, hiperparametry, stałe.
"""
from pathlib import Path

# --- Ścieżki ---
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data" / "neu-det"
CHECKPOINT_DIR = BASE_DIR / "checkpoints"

# --- Klasy NEU-DET ---
CLASSES = [
    "crazing",
    "inclusion",
    "patches",
    "pitted_surface",
    "rolled-in_scale",
    "scratches",
]
NUM_CLASSES = len(CLASSES)
CLASS_TO_IDX = {cls: i for i, cls in enumerate(CLASSES)}

# --- Obrazy ---
IMG_SIZE = 224  # ResNet-18 input size
IMG_CHANNELS = 3
MEAN = [0.485, 0.456, 0.406]  # ImageNet mean
STD = [0.229, 0.224, 0.225]   # ImageNet std

# --- Trening ---
BATCH_SIZE = 32
EPOCHS = 30
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-5
NUM_WORKERS = 0  # 0 dla Windows (unikamy problemów z multiprocessingiem)
SEED = 42
DEVICE = "cpu"  # "cuda" lub "cpu"

# --- Augmentacja ---
AUGMENTATION_PROB = 0.5
