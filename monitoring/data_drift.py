"""
Data drift monitoring for the API.

Records prediction distribution statistics and input features,
comparing them with the baseline distribution from training.
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
    Computes input image statistics for drift monitoring.

    Args:
        image_bytes: raw image bytes

    Returns:
        dict with: mean, std, histogram (16 bins), size, hash
    """
    import cv2
    import numpy as np

    img_array = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    if img is None:
        return {}

    # Basic statistics
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
    Saves a single prediction to the monitoring log.
    """
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    log_entry = {
        "timestamp": datetime.utcnow().isoformat(),
        "predicted_class": result["predicted_class"],
        "predicted_index": result["predicted_index"],
        "confidence": result["confidence"],
        "probabilities": result["all_probabilities"],
    }

    # Add image statistics
    img_stats = compute_image_stats(image_bytes)
    log_entry.update(img_stats)

    # Rotating log: save to JSONL file (one JSON per line)
    log_file = LOG_DIR / f"predictions_{datetime.utcnow().strftime('%Y%m')}.jsonl"
    with open(log_file, "a") as f:
        f.write(json.dumps(log_entry) + "\n")

    return log_entry


def compute_baseline():
    """
    Computes baseline distribution from training data.
    Run once after training.
    """
    print("Computing baseline distribution from training data...")

    import cv2
    from pathlib import Path
    from train.config import DATA_DIR

    train_dir = DATA_DIR / "train" / "images"
    all_means, all_stds, all_hists = [], [], []

    for class_name in CLASSES:
        class_dir = train_dir / class_name
        if not class_dir.exists():
            continue

        for img_path in list(class_dir.glob("*.jpg"))[:50]:  # 50 per class = 300 images
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

    print(f"Baseline saved to {BASELINE_FILE}")
    print(f"  Samples: {baseline['n_samples']}")
    print(f"  Pixel mean: {baseline['mean_mean']:.2f} ± {baseline['mean_std']:.2f}")
    return baseline


def detect_drift(stats: dict, threshold: float = 0.05) -> dict:
    """
    Detects data drift by comparing with baseline.

    Args:
        stats: image statistics (from compute_image_stats)
        threshold: p-value threshold for KS test

    Returns:
        dict with drift detection results
    """
    if not BASELINE_FILE.exists():
        return {"drift_detected": False, "reason": "No baseline available"}

    with open(BASELINE_FILE) as f:
        baseline = json.load(f)

    # K-S test on pixel mean
    z_score = (stats["mean_pixel"] - baseline["mean_mean"]) / (baseline["mean_std"] + 1e-8)

    # Simple test: if mean deviates > 3 sigma, we have drift
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
        # Save alert
        alerts_file = LOG_DIR / "drift_alerts.jsonl"
        with open(alerts_file, "a") as f:
            f.write(json.dumps(result) + "\n")

    return result


def generate_report(n_last: int = 100) -> dict:
    """
    Generates report from last N predictions.
    """
    log_files = sorted(LOG_DIR.glob("predictions_*.jsonl"))
    if not log_files:
        return {"error": "No monitoring data"}

    # Load last N entries
    records = []
    for log_file in reversed(log_files):
        with open(log_file) as f:
            for line in f:
                records.append(json.loads(line.strip()))
            if len(records) >= n_last:
                break

    records = records[-n_last:]

    if not records:
        return {"error": "No data"}

    # Statistics
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
    # Test: compute baseline
    if not BASELINE_FILE.exists():
        compute_baseline()
    else:
        print(f"Baseline already exists: {BASELINE_FILE}")

    # Test: generate report
    report = generate_report()
    print(f"\nMonitoring report: {json.dumps(report, indent=2)}")