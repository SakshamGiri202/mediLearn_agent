import pandas as pd
import numpy as np
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.datasets import make_classification
import os

def get_hospital_data(dataset_name: str):
    """
    Generates synthetic, consistent data for each hospital.
    This is faster and more reliable for a hackathon.
    We ensure all datasets have 10 features for an easy 'warm-start'.
    """
    if dataset_name == "heart_disease.csv":
        # More samples, slightly easier to classify
        X, y = make_classification(
            n_samples=150, n_features=10, n_informative=5, n_redundant=0, random_state=42
        )
        return X, y
    elif dataset_name == "diabetes.csv":
        # Fewer samples, different data distribution
        X, y = make_classification(
            n_samples=100, n_features=10, n_informative=4, n_redundant=1, random_state=13
        )
        return X, y
    elif dataset_name == "stroke.csv":
        # Fewer samples, harder to classify
        X, y = make_classification(
            n_samples=80, n_features=10, n_informative=3, n_redundant=2, random_state=7
        )
        return X, y
    else:
        # A default case
        X, y = make_classification(
            n_samples=50, n_features=10, random_state=1
        )
        return X, y


def train_on_local_data(dataset_name: str, global_model_weights: list | None = None):
    """
    This is your main function.
    It simulates a hospital training its local model.
    """
    
    # 1. Load Data
    X, y = get_hospital_data(dataset_name)
    
    # 2. Preprocessing & Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # 3. Initialize Model
    # We use SGDClassifier(loss='log_loss') which is just Logistic Regression
    # trained with Gradient Descent. It's perfect for this because it supports
    # "warm_start" (using old weights) and "max_iter" (training a few steps).
    model = SGDClassifier(
        loss='log_loss',
        max_iter=5,          # "trains a few steps"
        warm_start=True,     # "use the global weights"
        random_state=42,
        tol=1e-3             # Stop if it's not improving
    )

    # 4. Crucially: Apply Global Weights (The "Warm-Up")
    if global_model_weights:
        try:
            # We must set the weights (coef_) AND the intercept (intercept_)
            # AND tell the model what the classes are (e.g., 0 and 1)
            model.coef_ = np.array(global_model_weights[0])
            model.intercept_ = np.array(global_model_weights[1])
            model.classes_ = np.array([0, 1]) # Hard-coding for binary classification
            print(f"[{dataset_name}]: Applied global weights successfully.")
        except Exception as e:
            print(f"[{dataset_name}]: Could not apply weights (Error: {e}). Training from scratch.")
    
    # 5. Train Model
    # If weights were applied, 'warm_start=True' improves them.
    # If not, it just trains a new model.
    model.fit(X_train_scaled, y_train)

    # 6. Calculate Accuracy
    y_pred = model.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)
    
    # 7. Return Weights and Metrics (as per the API contract)
    # Convert to simple lists for JSON.
    local_weights = [model.coef_.tolist(), model.intercept_.tolist()]
    num_samples = len(X_train_scaled)
    
    print(f"[{dataset_name}]: Local training complete. Accuracy: {accuracy:.4f}")

    # This return value is what Member 2 (Agent) will get
    return local_weights, accuracy, num_samples


# --- This "if __name__ == '__main__'" block is for YOU to test your code ---
if __name__ == "__main__":
    print("--- Testing train_local.py ---")
    
    # Test 1: Train from scratch
    print("\n--- Test 1: Training Heart Disease (from scratch) ---")
    weights1, acc1, samples1 = train_on_local_data("heart_disease.csv")
    print(f"Result: Accuracy {acc1:.4f}, Samples {samples1}")

    # Test 2: Train from previous weights
    # This proves your "warm-up" logic works.
    print("\n--- Test 2: Training Diabetes (using Heart weights) ---")
    weights2, acc2, samples2 = train_on_local_data("diabetes.csv", weights1)
    print(f"Result: Accuracy {acc2:.4f}, Samples {samples2}")