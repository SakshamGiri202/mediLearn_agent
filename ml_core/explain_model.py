# ml_core/explain_model.py
import numpy as np, shap, os, matplotlib.pyplot as plt
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_classification
from .train_local import get_hospital_data

PLOT_DIR = "ml_core/plots"
os.makedirs(PLOT_DIR, exist_ok=True)

def generate_explanation(global_model_weights, dataset_name="heart_disease.csv"):
    """
    Generate SHAP feature-importance plot for the global model.
    """
    if not global_model_weights:
        print("⚠️ No global weights provided, skipping explanation.")
        return None

    # --- 1️⃣ Prepare baseline data for SHAP ---
    X, y = get_hospital_data(dataset_name)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # --- 2️⃣ Load model using global weights ---
    model = SGDClassifier(loss="log_loss", random_state=42)
    model.coef_ = np.array(global_model_weights[0])
    model.intercept_ = np.array(global_model_weights[1])
    model.classes_ = np.array([0, 1])

    # --- 3️⃣ Explain model using SHAP KernelExplainer ---
    explainer = shap.Explainer(model.predict_proba, X_scaled[:50])  # sample subset
    shap_values = explainer(X_scaled[:50])

    # --- 4️⃣ Plot mean absolute feature importance ---
    plt.figure(figsize=(8, 6))
    shap.summary_plot(shap_values[..., 1], X_scaled[:50], show=False)
    save_path = os.path.join(PLOT_DIR, "importance.png")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

    print(f"🧩 SHAP feature-importance plot saved → {save_path}")
    return save_path
