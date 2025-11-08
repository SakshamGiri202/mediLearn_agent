# ml_core/explain_model.py
import numpy as np
import shap
import matplotlib.pyplot as plt
import os

PLOT_DIR = "ml_core/plots"
os.makedirs(PLOT_DIR, exist_ok=True)

def generate_explanation(global_weights):
    """Generates SHAP-like feature importance visualization for global model."""
    if not global_weights:
        raise ValueError("No global model weights provided.")

    # Real-world medical feature names
    feature_names = [
        "Age", "Sex", "Chest Pain Type", "Resting BP", "Cholesterol",
        "Fasting Blood Sugar", "Rest ECG", "Max Heart Rate",
        "Exercise Angina", "ST Depression"
    ]

    coef = np.array(global_weights[0]).flatten()
    coef = coef[:len(feature_names)]  # just in case
    importance = np.abs(coef)

    # Plot
    plt.figure(figsize=(8, 6))
    plt.barh(feature_names, importance, color="dodgerblue")
    plt.xlabel("Feature Importance (|Weight|)")
    plt.ylabel("Health Indicators")
    plt.title("Global Model Feature Importance")
    plt.tight_layout()

    path = os.path.join(PLOT_DIR, "importance.png")
    plt.savefig(path)
    plt.close()
    print(f"🧩 SHAP-like feature importance plot saved → {path}")
    return path
