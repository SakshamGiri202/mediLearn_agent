import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import SGDClassifier
from ml_core.train_local import get_hospital_data

def predict_disease(input_data: list, global_model_weights: list):
    """Predicts disease likelihood + confidence from global weights."""
    if not global_model_weights or len(input_data) != 10:
        return "Error: Invalid input or model not trained."

    X_ref, _ = get_hospital_data("heart_disease.csv")
    scaler = StandardScaler().fit(X_ref)
    scaled_input = scaler.transform(np.array(input_data).reshape(1, -1))

    model = SGDClassifier(loss='log_loss', random_state=42, warm_start=True)
    try:
        model.coef_ = np.array(global_model_weights[0])
        model.intercept_ = np.array(global_model_weights[1])
        model.classes_ = np.array([0, 1])
    except Exception as e:
        return f"Error loading model: {e}"

    try:
        probs = model.predict_proba(scaled_input)[0]
        pred = np.argmax(probs)
        confidence = round(float(probs[pred]) * 100, 1)
    except Exception:
        pred = model.predict(scaled_input)[0]
        confidence = 50.0

    result = "Positive (Disease Detected)" if pred == 1 else "Negative (No Disease)"
    return f"{result} — {confidence}% Confidence"
