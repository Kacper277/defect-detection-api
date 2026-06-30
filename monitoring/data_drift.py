"""
Monitorowanie dryfu danych (data drift) dla API.

Rejestruje statystyki rozkładu predykcji i cech wejściowych,
porównując je z bazowym rozkładem z treningu.
"""
import json
import time
import hashlib
from pathlib import Path
from datetime import datetime
from typing import Optional

import numpy as np
from scipy.stats import ks_2samp

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from train.config import CHECKPOINT_DIR, CLASSES

LOG_DIR = Path(__file__).parent / "logs"
BASELINE_FILE = CHECKPOINT_DIR / "baseline_distribution.json"


def compute_image_stats(image_bytes: bytes) -> dict:
    """
    Oblicza statystyki obrazu wejściowego dla monitoringu dryfu.

    Args:
        image_bytes: surowe bajty obrazu

    Returns:
        dict z: średnia, std, histogram (16 binów), rozmiar, hash
    """
    import cv2
    import numpy as np

    img_array = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    if img is None:
        return {}

    # Statystyki podstawowe
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    hist = cv2.calcHist([gray], [0], None, [16], [0, 256]).flatten().tolist()

    return {
        "mean_pixel": float(gray.mean()),
        "std_pixel": float(gray.std()),
        "histogram_16": hist,
        "width": img.shape[1],
        "height": img.shape[0],
        "channels": img.shape[2],
        "file_size": len(image_bytes),
        "timestamp": datetime.utcnow().isoformat(),
        "image_hash": hashlib.md5(image_bytes).hexdigest()[:12],
    }


def log_prediction(result: dict, image_bytes: bytes):
    """
    Zapisuje pojedynczą predykcję do dziennika monitoringu.
    """
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    log_entry = {
        "timestamp": datetime.utcnow().isoformat(),
        "predicted_class": result["predicted_class"],
        "predicted_index": result["predicted_index"],
        "confidence": result["confidence"],
        "probabilities": result["all_probabilities"],
    }

    # Dodaj statystyki obrazu
    img_stats = compute_image_stats(image_bytes)
    log_entry.update(img_stats)

    # Log rotacyjny: zapisz do pliku JSONL (jeden JSON na linię)
    log_file = LOG_DIR / f"predictions_{datetime.utcnow().strftime('%Y%m')}.jsonl"
    with open(log_file, "a") as f:
        f.write(json.dumps(log_entry) + "\n")

    return log_entry


def compute_baseline():
    """
    Oblicza bazowy rozkład z danych treningowych.
    Uruchom raz po treningu.
    """
    print("Obliczanie baseline distribution z danych treningowych...")

    import cv2
    from pathlib import Path
    from train.config import DATA_DIR

    train_dir = DATA_DIR / "train" / "images"
    all_means, all_stds, all_hists = [], [], []

    for class_name in CLASSES:
        class_dir = train_dir / class_name
        if not class_dir.exists():
            continue

        for img_path in list(class_dir.glob("*.jpg"))[:50]:  # 50 na klasę = 300 obrazów
            img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            all_means.append(float(img.mean()))
            all_stds.append(float(img.std()))
            hist = cv2.calcHist([img], [0], None, [16], [0, 256]).flatten().tolist()
            all_hists.append(hist)

    baseline = {
        "mean_mean": float(np.mean(all_means)),
        "mean_std": float(np.std(all_means)),
        "std_mean": float(np.mean(all_stds)),
        "std_std": float(np.std(all_stds)),
        "hist_mean": np.mean(all_hists, axis=0).tolist(),
        "n_samples": len(all_means),
        "classes": CLASSES,
        "timestamp": datetime.utcnow().isoformat(),
    }

    with open(BASELINE_FILE, "w") as f:
        json.dump(baseline, f, indent=2)

    print(f"Baseline zapisany do {BASELINE_FILE}")
    print(f"  Próbki: {baseline['n_samples']}")
    print(f"  Średnia pixeli: {baseline['mean_mean']:.2f} ± {baseline['mean_std']:.2f}")
    return baseline


def detect_drift(stats: dict, threshold: float = 0.05) -> dict:
    """
    Wykrywa dryf danych przez porównanie z baseline.

    Args:
        stats: statystyki obrazu (z compute_image_stats)
        threshold: próg p-value dla testu KS

    Returns:
        dict z wynikami detekcji dryfu
    """
    if not BASELINE_FILE.exists():
        return {"drift_detected": False, "reason": "No baseline available"}

    with open(BASELINE_FILE) as f:
        baseline = json.load(f)

    # Test K-S na średniej pixeli
    z_score = (stats["mean_pixel"] - baseline["mean_mean"]) / (baseline["mean_std"] + 1e-8)

    # Prosty test: jeśli mean odbiega > 3 sigma, mamy dryf
    drift_mean = abs(z_score) > 3.0
    drift_confidence = stats["confidence"] < 0.5

    drift_detected = drift_mean or drift_confidence

    result = {
        "drift_detected": drift_detected,
        "z_score_mean_pixel": round(z_score, 3),
        "current_mean": stats["mean_pixel"],
        "baseline_mean": baseline["mean_mean"],
        "current_confidence": stats["confidence"],
        "timestamp": datetime.utcnow().isoformat(),
    }

    if drift_detected:
        # Zapisz alert
        alerts_file = LOG_DIR / "drift_alerts.jsonl"
        with open(alerts_file, "a") as f:
            f.write(json.dumps(result) + "\n")

    return result


def generate_report(n_last: int = 100) -> dict:
    """
    Generuje raport z ostatnich N predykcji.
    """
    log_files = sorted(LOG_DIR.glob("predictions_*.jsonl"))
    if not log_files:
        return {"error": "Brak danych monitoringu"}

    # Wczytaj ostatnie N wpisów
    records = []
    for log_file in reversed(log_files):
        with open(log_file) as f:
            for line in f:
                records.append(json.loads(line.strip()))
            if len(records) >= n_last:
                break

    records = records[-n_last:]

    if not records:
        return {"error": "Brak danych"}

    # Statystyki
    classes = [r["predicted_class"] for r in records]
    confidences = [r["confidence"] for r in records]
    means = [r["mean_pixel"] for r in records if "mean_pixel" in r]

    class_distribution = {}
    for cls in set(classes):
        class_distribution[cls] = classes.count(cls)

    report = {
        "total_predictions": len(records),
        "class_distribution": class_distribution,
        "avg_confidence": round(float(np.mean(confidences)), 4),
        "min_confidence": round(float(np.min(confidences)), 4),
        "avg_mean_pixel": round(float(np.mean(means)), 2) if means else None,
        "drift_alerts": list(LOG_DIR.glob("drift_alerts.jsonl"))[0].stat().st_size
        if any(LOG_DIR.glob("drift_alerts.jsonl")) else 0,
        "time_range": f"{records[0]['timestamp']} -> {records[-1]['timestamp']}"
        if len(records) > 1 else records[0]["timestamp"],
    }

    return report


if __name__ == "__main__":
    # Test: oblicz baseline
    if not BASELINE_FILE.exists():
        compute_baseline()
    else:
        print(f"Baseline już istnieje: {BASELINE_FILE}")

    # Test: wygeneruj raport
    report = generate_report()
    print(f"\nRaport monitoringu: {json.dumps(report, indent=2)}")
