# ml_core/train_local.py

import numpy as np
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.datasets import make_classification
from datetime import datetime
import random

# ---------------------------------------------------------
# 🧱 DATA GENERATOR FOR EACH HOSPITAL
# ---------------------------------------------------------
def get_hospital_data(dataset_name: str):
    """
    Generates consistent yet slightly varied synthetic data for each hospital.
    Each dataset has its own difficulty level, helping simulate domain shift.
    """
    seed_offset = random.randint(0, 50)  # adds small randomization for realism

    if dataset_name == "heart_disease.csv":
        X, y = make_classification(
            n_samples=150,
            n_features=10,
            n_informative=6,
            n_redundant=0,
            random_state=42 + seed_offset
        )
    elif dataset_name == "diabetes.csv":
        X, y = make_classification(
            n_samples=120,
            n_features=10,
            n_informative=5,
            n_redundant=2,
            random_state=13 + seed_offset
        )
    elif dataset_name == "stroke.csv":
        X, y = make_classification(
            n_samples=90,
            n_features=10,
            n_informative=4,
            n_redundant=2,
            random_state=7 + seed_offset
        )
    else:
        X, y = make_classification(
            n_samples=80,
            n_features=10,
            n_informative=5,
            random_state=1 + seed_offset
        )

    return X, y


# ---------------------------------------------------------
# 🧠 LOCAL TRAINING FUNCTION
# ---------------------------------------------------------
def train_on_local_data(dataset_name: str, global_model_weights: list | None = None):
    """
    Simulates local model training at each hospital.
    Applies received global weights (if provided) for warm-start training.
    Returns updated weights, local accuracy, and number of samples.
    """

    print(f"\n🏥 [{dataset_name}] Starting local training at {datetime.now().strftime('%H:%M:%S')}")

    # 1️⃣ Load Local Data
    X, y = get_hospital_data(dataset_name)

    # 2️⃣ Preprocessing
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # 3️⃣ Initialize Model
    model = SGDClassifier(
        loss='log_loss',
        max_iter=15,        # 🔹 more iterations = stronger local learning
        warm_start=True,    # 🔹 reuse weights if provided
        random_state=42,
        tol=1e-3
    )

    # 4️⃣ Apply Global Model Weights (Warm Start)
    if global_model_weights:
        try:
            model.coef_ = np.array(global_model_weights[0])
            model.intercept_ = np.array(global_model_weights[1])
            model.classes_ = np.array([0, 1])
            print(f"🔄 [{dataset_name}] Global weights loaded successfully.")
        except Exception as e:
            print(f"⚠️ [{dataset_name}] Failed to apply global weights: {e}. Reinitializing model.")

    # 5️⃣ Train Model on Local Data
    model.fit(X_train_scaled, y_train)

    # 6️⃣ Evaluate
    y_pred = model.predict(X_test_scaled)
    accuracy = round(accuracy_score(y_test, y_pred), 3)
    num_samples = len(X_train_scaled)
    local_weights = [model.coef_.tolist(), model.intercept_.tolist()]

    # 7️⃣ Report
    print(f"✅ [{dataset_name}] Local training done → Accuracy={accuracy}, Samples={num_samples}")
    return local_weights, accuracy, num_samples


# ---------------------------------------------------------
# 🧪 LOCAL TEST (DEBUG)
# ---------------------------------------------------------
if __name__ == "__main__":
    print("🧪 Running standalone test for local training module...\n")

    # Hospital A: Heart Disease
    w1, a1, s1 = train_on_local_data("heart_disease.csv")
    print(f"Result → Heart: Accuracy={a1}, Samples={s1}")

    # Hospital B: Diabetes (using Heart’s global weights)
    w2, a2, s2 = train_on_local_data("diabetes.csv", w1)
    print(f"Result → Diabetes: Accuracy={a2}, Samples={s2}")

    # Hospital C: Stroke (using Diabetes’s global weights)
    w3, a3, s3 = train_on_local_data("stroke.csv", w2)
    print(f"Result → Stroke: Accuracy={a3}, Samples={s3}")

    print("\n🧠 Test complete — hospital training pipeline functioning properly.")
