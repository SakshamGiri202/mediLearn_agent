import numpy as np
import shap
import matplotlib.pyplot as plt
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler
from ml_core.train_local import get_hospital_data

def generate_explanation(global_model_weights, dataset_name="heart_disease.csv", output_path="feature_importance.png"):
    """Generates SHAP-based feature importance plot for global model."""
    X, y = get_hospital_data(dataset_name)
    scaler = StandardScaler().fit(X)
    X_scaled = scaler.transform(X)

    model = SGDClassifier(loss='log_loss', random_state=42)
    try:
        model.coef_ = np.array(global_model_weights[0])
        model.intercept_ = np.array(global_model_weights[1])
        model.classes_ = np.array([0, 1])
    except Exception as e:
        print(f"⚠️ Failed to load global weights: {e}")
        return None

    try:
        explainer = shap.Explainer(model.predict, X_scaled)
        shap_values = explainer(X_scaled)
        shap.summary_plot(shap_values, X_scaled, show=False)
    except Exception:
        weights = np.abs(model.coef_[0])
        plt.bar(range(len(weights)), weights)
        plt.title("Feature Importance (Coefficient Magnitude)")
        plt.xlabel("Feature Index")
        plt.ylabel("Importance")

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"✅ Feature importance saved → {output_path}")
    return output_path
