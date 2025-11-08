"""
MediLearn Core Prediction Module
--------------------------------
Uses the global model bundle saved by the MediLearn backend:
{
  "weights": [coef, intercept],
  "scaler": {"mean": [...], "scale": [...]},
  "meta": {...}
}

Performs deterministic logistic prediction using stored weights and scaler.
Outputs a readable risk assessment for dashboard or API display.
"""

import json
import numpy as np
import os
from datetime import datetime

GLOBAL_MODEL_FILE = "global_model.json"

# ----------------------------- #
# 📦 Load Global Model Bundle
# ----------------------------- #
def load_global_model_bundle():
    """Loads and validates the global model bundle."""
    if not os.path.exists(GLOBAL_MODEL_FILE):
        raise FileNotFoundError("No trained global model found (global_model.json missing).")

    with open(GLOBAL_MODEL_FILE, "r", encoding="utf-8") as f:
        gm = json.load(f)

    if isinstance(gm, dict) and "weights" in gm and "scaler" in gm:
        return gm
    elif isinstance(gm, list):  # legacy format
        coef = np.array(gm[0])
        intercept = np.array(gm[1])
        n_features = coef.shape[-1]
        mean = np.zeros(n_features)
        scale = np.ones(n_features)
        return {"weights": gm, "scaler": {"mean": mean.tolist(), "scale": scale.tolist()}}
    else:
        raise ValueError("Invalid global model structure.")

# ----------------------------- #
# ⚙️ Scaling & Logistic Helpers
# ----------------------------- #
def safe_scale_input(features, scaler):
    """Standardize input using saved scaler stats."""
    x = np.array(features, dtype=float)
    mean = np.array(scaler.get("mean", np.zeros_like(x)))
    scale = np.array(scaler.get("scale", np.ones_like(x)))
    scale[scale == 0] = 1.0
    return (x - mean) / scale

def sigmoid(z):
    """Numerically stable sigmoid function."""
    return 1.0 / (1.0 + np.exp(-z)) if z >= 0 else np.exp(z) / (1.0 + np.exp(z))

def logistic_predict(coef, intercept, x_scaled):
    """Return (p0, p1) for binary logistic regression."""
    coef = np.array(coef).reshape(-1)
    intercept = float(np.array(intercept).reshape(-1)[0])
    z = float(np.dot(coef, x_scaled) + intercept)
    p1 = sigmoid(z)
    return 1 - p1, p1

# ----------------------------- #
# 🧠 Main Prediction Function
# ----------------------------- #
def predict_disease(features):
    """
    Given patient features, returns human-readable risk prediction.
    Automatically loads the global model bundle and uses its scaler.
    """
    bundle = load_global_model_bundle()
    weights = bundle["weights"]
    scaler = bundle["scaler"]

    coef = weights[0]
    intercept = weights[1]

    if len(features) != len(coef[0]) if isinstance(coef[0], list) else len(coef):
        raise ValueError(f"Expected {len(coef[0]) if isinstance(coef[0], list) else len(coef)} features, got {len(features)}.")

    x_scaled = safe_scale_input(features, scaler)
    p0, p1 = logistic_predict(coef, intercept, x_scaled)
    prediction = 1 if p1 > p0 else 0
    confidence = round(float(max(p0, p1) * 100), 1)

    # Risk classification
    if prediction == 1:
        if confidence >= 85:
            risk = "High Risk"
        elif confidence >= 65:
            risk = "Moderate Risk"
        else:
            risk = "Low Risk"
    else:
        if confidence >= 75:
            risk = "Low Risk"
        else:
            risk = "Very Low Risk"

    label = "🔴 Positive (Heart Disease Detected)" if prediction == 1 else "🟢 Negative (No Heart Disease)"
    result = {
        "result": label,
        "prediction": int(prediction),
        "confidence": confidence,
        "risk_level": risk,
        "probabilities": {"class_0": round(p0, 4), "class_1": round(p1, 4)},
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    return result
