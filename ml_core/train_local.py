# ml_core/train_local.py
import numpy as np
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.datasets import make_classification
from datetime import datetime
import json, os, random

def get_hospital_data(dataset_name: str):
    """Generates evolving synthetic data for each hospital."""
    base_seed = {"heart_disease.csv": 42, "diabetes.csv": 13, "stroke.csv": 7}.get(dataset_name, 1)
    evolving_seed = base_seed + random.randint(0, 999)

    if dataset_name == "heart_disease.csv":
        X, y = make_classification(n_samples=150, n_features=10, n_informative=6,
                                   n_redundant=0, random_state=evolving_seed)
    elif dataset_name == "diabetes.csv":
        X, y = make_classification(n_samples=120, n_features=10, n_informative=5,
                                   n_redundant=2, random_state=evolving_seed)
    elif dataset_name == "stroke.csv":
        X, y = make_classification(n_samples=90, n_features=10, n_informative=4,
                                   n_redundant=2, random_state=evolving_seed)
    else:
        X, y = make_classification(n_samples=60, n_features=10, random_state=evolving_seed)
    return X, y

def train_on_local_data(dataset_name: str, global_model_weights: list | None = None):
    hospital_name = os.getenv("HOSPITAL_NAME", dataset_name)
    print(f"\n🏥 [{hospital_name}] Training on {dataset_name} ({datetime.now().strftime('%H:%M:%S')})")

    X, y = get_hospital_data(dataset_name)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = SGDClassifier(loss='log_loss', max_iter=10, warm_start=True, random_state=42, tol=1e-3)

    if global_model_weights:
        try:
            model.coef_ = np.array(global_model_weights[0])
            model.intercept_ = np.array(global_model_weights[1])
            model.classes_ = np.array([0, 1])
            print(f"🔄 Global weights applied.")
        except Exception as e:
            print(f"⚠️ Could not apply global weights: {e}")

    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    accuracy = round(float(accuracy_score(y_test, y_pred)), 3)

    noise_scale = 0.02
    coef = np.clip(model.coef_, -1, 1)
    intercept = np.clip(model.intercept_, -1, 1)
    coef += np.random.normal(0, noise_scale, coef.shape)
    intercept += np.random.normal(0, noise_scale, intercept.shape)
    utility_score = round(float(100 * (1 - noise_scale)), 1)

    local_weights = [coef.astype(float).tolist(), intercept.astype(float).tolist()]
    samples = len(X_train_scaled)
    feature_names = [f"Feature_{i+1}" for i in range(X.shape[1])]

    print(f"✅ [{hospital_name}] Accuracy={accuracy}, Privacy σ={noise_scale}, Utility={utility_score}%")

    # Optional SHAP local explainability
    try:
        import shap, matplotlib.pyplot as plt
        os.makedirs("ml_core/plots", exist_ok=True)
        explainer = shap.LinearExplainer(model, X_train_scaled)
        shap_values = explainer.shap_values(X_test_scaled)
        shap.summary_plot(shap_values, X_test_scaled, feature_names=feature_names, show=False)
        plt.title(f"{hospital_name} Feature Importance")
        plt.savefig(f"ml_core/plots/{hospital_name}_shap.png")
        plt.close()
    except Exception as e:
        print(f"⚠️ SHAP skipped for {hospital_name}: {e}")

    return local_weights, accuracy, samples, feature_names

if __name__ == "__main__":
    w, a, s, f = train_on_local_data("heart_disease.csv")
    print(f"Test result → acc={a}, samples={s}")
