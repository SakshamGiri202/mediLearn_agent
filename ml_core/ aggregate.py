# ml_core/aggregate.py
import json
import os
import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# ------------------------------------------------------------
# 🧠 FEDERATED AGGREGATION CORE (FedAvg)
# ------------------------------------------------------------

def aggregate_fedavg(results):
    """
    Compute the weighted average accuracy of all hospital models.
    This function is called by the controller after each round.
    """
    total_samples = sum(r.get("samples", 0) for r in results if r)
    if total_samples == 0:
        return 0.0
    weighted_sum = sum(r["accuracy"] * r["samples"] for r in results if r)
    return round(weighted_sum / total_samples, 3)


def aggregate_model_weights(results):
    """
    Combine weights from multiple hospital models into a single global model
    using simple element-wise mean (FedAvg).
    """
    valid = [r for r in results if "weights" in r]
    if not valid:
        return None

    coefs = [np.array(r["weights"][0]) for r in valid]
    intercepts = [np.array(r["weights"][1]) for r in valid]

    avg_coef = np.mean(coefs, axis=0).tolist()
    avg_intercept = np.mean(intercepts, axis=0).tolist()

    return [avg_coef, avg_intercept]


def save_global_model(weights, path="global_model.json"):
    """Save global model weights after aggregation."""
    if weights:
        with open(path, "w") as f:
            json.dump(weights, f, indent=2)
    return path


def load_global_model(path="global_model.json"):
    """Load previously saved global model weights."""
    if os.path.exists(path):
        with open(path, "r") as f:
            return json.load(f)
    return None


# ------------------------------------------------------------
# 🧪 GLOBAL VALIDATION (Evaluation on Test Dataset)
# ------------------------------------------------------------

def evaluate_global_model(
    weights_path="global_model.json",
    dataset_path="ml_core/dataset/heart_disease.csv"
):
    """
    Evaluate the global model weights on a test dataset
    (can be any dataset representative of global distribution).
    """
    if not os.path.exists(weights_path):
        raise FileNotFoundError("Global model not found. Train first.")

    # --- Load and prepare dataset ---
    df = pd.read_csv(dataset_path)
    X = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # --- Load global weights ---
    with open(weights_path, "r") as f:
        weights = json.load(f)

    coef_, intercept_ = np.array(weights[0]), np.array(weights[1])

    # --- Create model using loaded weights ---
    model = SGDClassifier(loss="log_loss", random_state=42)
    model.classes_ = np.array([0, 1])
    model.coef_ = coef_
    model.intercept_ = intercept_

    # --- Predict and compute metrics ---
    y_pred = model.predict(X_scaled)

    metrics = {
        "accuracy": round(accuracy_score(y, y_pred), 3),
        "precision": round(precision_score(y, y_pred, zero_division=0), 3),
        "recall": round(recall_score(y, y_pred, zero_division=0), 3),
        "f1_score": round(f1_score(y, y_pred, zero_division=0), 3),
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }

    return metrics
