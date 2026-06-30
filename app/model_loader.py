"""
Ładowanie wytrenowanego modelu i Grad-CAM.
"""
import sys
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torchvision.transforms import Compose, Resize, ToTensor, Normalize, ToPILImage

# Dodaj katalog główny do ścieżki
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from train.config import (
    CLASSES,
    NUM_CLASSES,
    IMG_SIZE,
    MEAN,
    STD,
    DEVICE,
    CHECKPOINT_DIR,
)


class GradCAM:
    """
    Grad-CAM: wizualizacja aktywacji modelu dla danej klasy.
    """

    def __init__(self, model: nn.Module, target_layer: nn.Module):
        self.model = model
        self.target_layer = target_layer
        self.gradients: Optional[torch.Tensor] = None
        self.activations: Optional[torch.Tensor] = None
        self._register_hooks()

    def _register_hooks(self):
        """Rejestruje hooki do przechwytywania gradientów i aktywacji."""

        def forward_hook(module, input, output):
            self.activations = output

        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0]

        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_full_backward_hook(backward_hook)

    def generate(self, x: torch.Tensor, class_idx: Optional[int] = None) -> np.ndarray:
        """
        Generuje heatmapę Grad-CAM.

        Args:
            x: tensor obrazu [1, 3, H, W] (po normalizacji)
            class_idx: indeks klasy target. Jeśli None, używa argmax.

        Returns:
            heatmapa jako np.ndarray [H, W] w zakresie [0, 1]
        """
        # Forward pass
        self.model.zero_grad()
        output = self.model(x)

        if class_idx is None:
            class_idx = output.argmax(dim=1).item()

        # Backward pass dla wybranej klasy
        one_hot = torch.zeros_like(output)
        one_hot[0, class_idx] = 1
        output.backward(gradient=one_hot, retain_graph=True)

        # Pobierz gradienty i aktywacje
        gradients = self.gradients.detach()  # [1, C, H, W]
        activations = self.activations.detach()  # [1, C, H, W]

        # Global Average Pooling na gradientach
        weights = gradients.mean(dim=(2, 3), keepdim=True)  # [1, C, 1, 1]

        # Ważona suma aktywacji
        cam = (weights * activations).sum(dim=1, keepdim=True)  # [1, 1, H, W]
        cam = F.relu(cam)  # tylko pozytywne aktywacje

        # Normalizacja do [0, 1]
        cam = cam.squeeze().cpu().numpy()
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)

        # Resize do oryginalnego rozmiaru
        cam = cv2.resize(cam, (IMG_SIZE, IMG_SIZE))
        return cam


class ModelService:
    """
    Serwis do ładowania modelu i wykonywania predykcji.
    """

    def __init__(self, checkpoint_path: Optional[Path] = None):
        self.device = torch.device(DEVICE)
        self.classes = CLASSES

        # Transformacje dla obrazu wejściowego
        self.transform = Compose([
            ToPILImage(),
            Resize((IMG_SIZE, IMG_SIZE)),
            ToTensor(),
            Normalize(mean=MEAN, std=STD),
        ])

        # Załaduj model
        self.model = self._load_model(checkpoint_path)
        self.model.eval()

        # Grad-CAM na layer4 (ostatni blok konwolucyjny ResNet)
        self.gradcam = GradCAM(self.model, self.model.layer4)

        print(f"ModelService: model załadowany na {self.device}")
        print(f"  Klasy: {self.classes}")

    def _load_model(self, checkpoint_path: Optional[Path] = None) -> nn.Module:
        """Ładuje checkpoint modelu."""
        if checkpoint_path is None:
            checkpoint_path = CHECKPOINT_DIR / "best_model.pth"

        if not checkpoint_path.exists():
            raise FileNotFoundError(
                f"Checkpoint nie znaleziony: {checkpoint_path}\n"
                f"Uruchom najpierw trening: python -m train.train"
            )

        # Zbuduj model
        model = models.resnet18(weights=None)
        in_features = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(in_features, NUM_CLASSES),
        )

        # Wczytaj wagi
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        model.load_state_dict(checkpoint["model_state_dict"])
        model = model.to(self.device)

        print(f"  Załadowano checkpoint z epoch {checkpoint.get('epoch', '?')}, "
              f"F1={checkpoint.get('val_f1', '?'):.4f}")

        return model

    def predict(self, image: np.ndarray) -> dict:
        """
        Wykonuje predykcję dla obrazu.

        Args:
            image: obraz w formacie RGB (OpenCV) jako np.ndarray [H, W, 3]

        Returns:
            dict z predykcją, pewnością i heatmapą
        """
        # Preprocessing
        input_tensor = self.transform(image).unsqueeze(0).to(self.device)

        # Predykcja
        with torch.no_grad():
            output = self.model(input_tensor)
            probs = F.softmax(output, dim=1)

        predicted_class = output.argmax(dim=1).item()
        confidence = probs[0, predicted_class].item()

        # Wszystkie prawdopodobieństwa
        all_probs = probs[0].cpu().numpy().tolist()

        # Grad-CAM
        heatmap = self.gradcam.generate(input_tensor, class_idx=predicted_class)

        mean_pixel = float(image.mean())

        return {
            "predicted_class": self.classes[predicted_class],
            "predicted_index": predicted_class,
            "confidence": round(confidence, 4),
            "all_probabilities": {
                cls: round(prob, 4)
                for cls, prob in zip(self.classes, all_probs)
            },
            "heatmap": heatmap.tolist(),
            "heatmap_shape": heatmap.shape,
        }

    def predict_bytes(self, image_bytes: bytes) -> dict:
        """
        Wykonuje predykcję dla obrazu w postaci bajtów.

        Args:
            image_bytes: surowe bajty obrazu (JPEG, PNG, itp.)

        Returns:
            dict z predykcją
        """
        # Dekoduj obraz
        image_array = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError("Nie można zdekodować obrazu. Upewnij się, że przesłano poprawny plik graficzny.")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        return self.predict(image)


# Singleton serwisu
_service: Optional[ModelService] = None


def get_model_service() -> ModelService:
    """Zwraca singleton ModelService."""
    global _service
    if _service is None:
        _service = ModelService()
    return _service


if __name__ == "__main__":
    # Test
    service = get_model_service()
    print(f"Service ready: {service.classes}")

    # Test na obrazie z datasetu
    test_img_path = ROOT / "data" / "neu-det" / "validation" / "images" / "crazing" / "crazing_241.jpg"
    if test_img_path.exists():
        img = cv2.imread(str(test_img_path))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        result = service.predict(img)
        print(f"Predykcja: {result['predicted_class']} ({result['confidence']:.2%})")
        print(f"Probabilities: {result['all_probabilities']}")
        print(f"Heatmap shape: {result['heatmap_shape']}")
    else:
        print("Brak obrazu testowego")
