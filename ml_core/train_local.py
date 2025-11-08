# ml_core/train_local.py
import numpy as np
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.datasets import make_classification
from datetime import datetime
import json, os
import random


# ---------------------------------------------------------
# 🧱 Data generator
# ---------------------------------------------------------
def get_hospital_data(dataset_name: str):
    """
    Generates evolving synthetic data for each hospital across rounds.
    Each call will slightly differ because of the dynamic random_state.
    """
    base_seed = {
        "heart_disease.csv": 42,
        "diabetes.csv": 13,
        "stroke.csv": 7
    }.get(dataset_name, 1)
    
    # Add a small random offset to seed so data evolves each round
    evolving_seed = base_seed + random.randint(0, 999)

    if dataset_name == "heart_disease.csv":
        X, y = make_classification(
            n_samples=150, n_features=10, n_informative=6, n_redundant=0, random_state=evolving_seed
        )
    elif dataset_name == "diabetes.csv":
        X, y = make_classification(
            n_samples=100, n_features=10, n_informative=5, n_redundant=2, random_state=evolving_seed
        )
    elif dataset_name == "stroke.csv":
        X, y = make_classification(
            n_samples=80, n_features=10, n_informative=4, n_redundant=2, random_state=evolving_seed
        )
    else:
        X, y = make_classification(n_samples=60, n_features=10, random_state=evolving_seed)
    
    return X, y


# ---------------------------------------------------------
# 🧠 Local training function (with privacy)
# ---------------------------------------------------------
def train_on_local_data(dataset_name: str, global_model_weights: list | None = None):
    print(f"\n🏥 Training on {dataset_name} ({datetime.now().strftime('%H:%M:%S')})")

    # 1️⃣ Load Data
    X, y = get_hospital_data(dataset_name)

    # 2️⃣ Split and scale
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # 3️⃣ Model setup
    model = SGDClassifier(
        loss='log_loss', max_iter=10, warm_start=True, random_state=42, tol=1e-3
    )

    # 4️⃣ Apply global weights if available
    if global_model_weights:
        try:
            model.coef_ = np.array(global_model_weights[0])
            model.intercept_ = np.array(global_model_weights[1])
            model.classes_ = np.array([0, 1])
            print(f"🔄 Applied global weights.")
        except Exception as e:
            print(f"⚠️ Could not apply global weights: {e}")

    # 5️⃣ Train
    model.fit(X_train_scaled, y_train)

    # 6️⃣ Evaluate
    y_pred = model.predict(X_test_scaled)
    accuracy = round(float(accuracy_score(y_test, y_pred)), 3)

    # 7️⃣ Apply differential privacy
    noise_scale = 0.02
    coef = np.clip(model.coef_, -1, 1)
    intercept = np.clip(model.intercept_, -1, 1)

    coef += np.random.normal(0, noise_scale, coef.shape)
    intercept += np.random.normal(0, noise_scale, intercept.shape)

    # 8️⃣ Compute privacy–utility metric (for visualization)
    utility_score = round(float(100 * (1 - noise_scale)), 1)

    # 9️⃣ Prepare JSON-safe weights
    local_weights = [
        np.array(coef).astype(float).tolist(),
        np.array(intercept).astype(float).tolist()
    ]

    samples = int(len(X_train_scaled))
    print(f"✅ [{dataset_name}] Accuracy={accuracy}, Privacy σ={noise_scale}, Utility={utility_score}%")

    # 🔟 Return results (JSON-safe)
    return local_weights, accuracy, samples


# ---------------------------------------------------------
# 🧪 Standalone test
# ---------------------------------------------------------
if __name__ == "__main__":
    weights, acc, samples = train_on_local_data("heart_disease.csv")
    print(f"Test result: acc={acc}, samples={samples}")
